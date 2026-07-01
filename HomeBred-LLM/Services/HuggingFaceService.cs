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
        var url = $"https://huggingface.co/api/models?search={Uri.EscapeDataString(query)}&filter=gguf&sort=downloads&direction=-1&limit={limit}";
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

    public async Task<List<HfFileInfo>> ListGgufFilesAsync(string repoId, CancellationToken ct = default)
    {
        var url = $"https://huggingface.co/api/models/{repoId}";
        var resp = await _http.GetFromJsonAsync<HfModelDetail>(url, ct);
        if (resp?.Siblings is null) return [];

        return resp.Siblings
            .Where(s => s.Rfilename.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
            .Where(s => !IsMmprojFile(s.Rfilename))
            .Select(s => new HfFileInfo(
                s.Rfilename,
                s.Size,
                ParseQuantization(s.Rfilename)))
            .ToList();
    }

    /// <summary>Vision models ship a companion "mmproj" GGUF (the CLIP/vision projector)
    /// alongside the text model — find it so the caller can download it too.</summary>
    public async Task<HfFileInfo?> FindMmprojFileAsync(string repoId, CancellationToken ct = default)
    {
        var url = $"https://huggingface.co/api/models/{repoId}";
        var resp = await _http.GetFromJsonAsync<HfModelDetail>(url, ct);
        if (resp?.Siblings is null) return null;

        var mmproj = resp.Siblings.FirstOrDefault(s =>
            s.Rfilename.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase) && IsMmprojFile(s.Rfilename));

        return mmproj is null ? null : new HfFileInfo(mmproj.Rfilename, mmproj.Size, null);
    }

    private static bool IsMmprojFile(string filename) =>
        filename.Contains("mmproj", StringComparison.OrdinalIgnoreCase);

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
        Directory.CreateDirectory(Path.GetDirectoryName(destPath)!);
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
