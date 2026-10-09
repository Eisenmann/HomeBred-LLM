using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;

namespace HomebredLLM.Services;

public record HfModelInfo(
    string RepoId,
    string ModelName,
    string? Author,
    long? Downloads,
    long? Likes,
    string[] Tags);

public record HfFileInfo(
    string Filename,
    long? SizeBytes,
    string? Quantization);

public sealed class HuggingFaceService
{
    private readonly HttpClient _http;
    private static readonly Regex _quantPattern =
        new(@"(Q\d+_[A-Z0-9_]+|IQ\d+_[A-Z0-9_]+|F16|F32|BF16)", RegexOptions.IgnoreCase | RegexOptions.Compiled);

    public HuggingFaceService()
    {
        _http = new HttpClient();
        _http.DefaultRequestHeaders.Add("User-Agent", "HomeBred-LLM/1.0");
        // Set HF token if available via env
        var token = Environment.GetEnvironmentVariable("HF_TOKEN");
        if (!string.IsNullOrEmpty(token))
            _http.DefaultRequestHeaders.Authorization = new("Bearer", token);
    }

    public async Task<List<HfModelInfo>> SearchModelsAsync(string query, int limit = 20, CancellationToken ct = default)
    {
        // Search for ONNX models on HuggingFace. The onnx filter returns models
        // that have ONNX files in their repository (exported via optimum-export).
        var url = $"https://huggingface.co/api/models?search={Uri.EscapeDataString(query)}&filter=onnx&sort=downloads&direction=-1&limit={limit}";
        var resp = await _http.GetFromJsonAsync<List<HfModelApiItem>>(url, ct) ?? [];
        return resp.Select(m => new HfModelInfo(
            m.Id,
            m.Id.Contains('/') ? m.Id[(m.Id.IndexOf('/') + 1)..] : m.Id,
            m.Author,
            m.Downloads,
            m.Likes,
            m.Tags ?? []
        )).ToList();
    }

    public async Task<List<HfFileInfo>> ListOnnxFilesAsync(string repoId, CancellationToken ct = default)
    {
        var url = $"https://huggingface.co/api/models/{repoId}";
        var resp = await _http.GetFromJsonAsync<HfModelDetail>(url, ct);
        if (resp?.Siblings is null) return [];

        // ONNX models are stored in subdirectories (e.g. onnx/model.onnx,
        // onnx/model.onnx.data). Group by directory to present each variant
        // as a single downloadable entry.
        var onnxDirs = resp.Siblings
            .Where(s => s.Rfilename.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase) ||
                        s.Rfilename.EndsWith(".onnx.data", StringComparison.OrdinalIgnoreCase))
            .GroupBy(s =>
            {
                var parts = s.Rfilename.Split('/');
                return parts.Length > 1 ? string.Join('/', parts[..^1]) : ".";
            })
            .Select(g => new HfFileInfo(
                g.Key,
                g.Sum(s => s.Size ?? 0),
                ParseQuantization(g.Key)))
            .ToList();

        return onnxDirs;
    }

    /// <summary>ONNX vision models may ship a separate projector directory.
    /// Find it so the caller can download it too.</summary>
    public async Task<HfFileInfo?> FindMmprojFileAsync(string repoId, CancellationToken ct = default)
    {
        var url = $"https://huggingface.co/api/models/{repoId}";
        var resp = await _http.GetFromJsonAsync<HfModelDetail>(url, ct);
        if (resp?.Siblings is null) return null;

        // Look for projector files in common ONNX vision model locations
        var mmproj = resp.Siblings.FirstOrDefault(s =>
            s.Rfilename.Contains("mmproj", StringComparison.OrdinalIgnoreCase) ||
            s.Rfilename.Contains("projector", StringComparison.OrdinalIgnoreCase));

        return mmproj is null ? null : new HfFileInfo(mmproj.Rfilename, mmproj.Size, null);
    }

    public async Task DownloadFileAsync(
        string repoId,
        string filename,
        string destPath,
        IProgress<(long downloaded, long total, double pct)>? progress = null,
        CancellationToken ct = default)
    {
        var url = $"https://huggingface.co/{repoId}/resolve/main/{filename}";
        using var resp = await _http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, ct);
        resp.EnsureSuccessStatusCode();

        var total = resp.Content.Headers.ContentLength ?? -1;
        using var src = await resp.Content.ReadAsStreamAsync(ct);
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(destPath))!);
        using var dst = new FileStream(destPath, FileMode.Create, FileAccess.Write, FileShare.None, 81920, true);

        var buf = new byte[81920];
        long downloaded = 0;
        int read;
        while ((read = await src.ReadAsync(buf, ct)) > 0)
        {
            await dst.WriteAsync(buf.AsMemory(0, read), ct);
            downloaded += read;
            var pct = total > 0 ? downloaded * 100.0 / total : 0;
            progress?.Report((downloaded, total, pct));
        }
    }

    /// <summary>
    /// Downloads one ONNX variant. <paramref name="variantDir"/> is the directory entry returned by
    /// <see cref="ListOnnxFilesAsync"/> ("onnx", "cpu_and_mobile/cpu-int4", or "." for the repo root).
    /// Files of that directory land flat in <paramref name="destDir"/>; the small root-level config /
    /// tokenizer files the runtime needs next to the model are downloaded too.
    /// </summary>
    public async Task DownloadDirectoryAsync(
        string repoId,
        string variantDir,
        string destDir,
        IProgress<(long downloaded, long total, double pct)>? progress = null,
        CancellationToken ct = default)
    {
        var detail = await _http.GetFromJsonAsync<HfModelDetail>(
            $"https://huggingface.co/api/models/{repoId}?blobs=true", ct);
        var siblings = detail?.Siblings ?? [];

        var prefix = variantDir is "." or "" ? "" : variantDir.TrimEnd('/') + "/";
        string[] sideExt = [".json", ".txt", ".model", ".jinja", ".tiktoken"];

        var files = new List<(string Remote, string Local, long Size)>();
        foreach (var s in siblings)
        {
            var name = s.Rfilename;
            if (prefix.Length > 0 && name.StartsWith(prefix, StringComparison.Ordinal))
                files.Add((name, name[prefix.Length..], s.Size ?? 0));
            else if (prefix.Length == 0 && !name.Contains('/') && IsRootFileForOnnx(name))
                files.Add((name, name, s.Size ?? 0));
            else if (prefix.Length > 0 && !name.Contains('/') &&
                     sideExt.Contains(Path.GetExtension(name), StringComparer.OrdinalIgnoreCase))
                files.Add((name, name, s.Size ?? 0)); // config.json, tokenizer.json ... next to the model
        }
        if (files.Count == 0)
            throw new InvalidOperationException($"No files found in '{variantDir}' of {repoId}.");

        var root = Path.GetFullPath(destDir);
        Directory.CreateDirectory(root);
        long grand = files.Sum(f => f.Size);
        long done = 0;
        foreach (var (remote, local, size) in files)
        {
            var target = Path.GetFullPath(Path.Combine(root, local));
            if (!target.StartsWith(root + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
                continue; // never write outside the model folder
            var baseDone = done;
            var inner = new Progress<(long downloaded, long total, double pct)>(p =>
            {
                var d = baseDone + p.downloaded;
                var t = grand > 0 ? grand : Math.Max(d, p.total);
                progress?.Report((d, t, t > 0 ? d * 100.0 / t : 0));
            });
            await DownloadFileAsync(repoId, remote, target, inner, ct);
            done += size > 0 ? size : new FileInfo(target).Length;
        }
    }

    // Root-level files of an ONNX-in-root repo: skip weights in other formats.
    private static bool IsRootFileForOnnx(string name) =>
        !name.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase) &&
        !name.EndsWith(".bin", StringComparison.OrdinalIgnoreCase) &&
        !name.EndsWith(".h5", StringComparison.OrdinalIgnoreCase) &&
        !name.EndsWith(".msgpack", StringComparison.OrdinalIgnoreCase) &&
        !name.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase) &&
        !name.EndsWith(".pt", StringComparison.OrdinalIgnoreCase) &&
        !name.EndsWith(".md", StringComparison.OrdinalIgnoreCase) &&
        !name.StartsWith(".", StringComparison.Ordinal);

    private static string? ParseQuantization(string filename)
    {
        var m = _quantPattern.Match(filename);
        return m.Success ? m.Value.ToUpperInvariant() : null;
    }

    // ── DTOs ────────────────────────────────────────────────────────────────

    private record HfModelApiItem(
        [property: JsonPropertyName("id")] string Id,
        [property: JsonPropertyName("author")] string? Author,
        [property: JsonPropertyName("downloads")] long? Downloads,
        [property: JsonPropertyName("likes")] long? Likes,
        [property: JsonPropertyName("tags")] string[]? Tags);

    private record HfModelDetail(
        [property: JsonPropertyName("siblings")] List<HfSibling>? Siblings);

    private record HfSibling(
        [property: JsonPropertyName("rfilename")] string Rfilename,
        [property: JsonPropertyName("size")] long? Size);
}
