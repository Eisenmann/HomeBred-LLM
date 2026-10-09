using System.Formats.Tar;
using System.IO.Compression;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace HomebredLLM.Services.Tiering;

/// <summary>
/// One-click setup of the GPU backend: downloads the official llama.cpp release that matches the
/// bundled LLamaSharp (<see cref="GpuRequirementsChecker.LlamaCppTag"/>) and unpacks its libraries
/// into the native folder the app loads from. Nothing is bundled in the app itself.
/// </summary>
public sealed class GpuBackendInstaller
{
    private static readonly Uri ReleaseApi =
        new($"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{GpuRequirementsChecker.LlamaCppTag}");

    private sealed record Asset(string Name, long Size, string Url, string? Sha256);

    private readonly HttpClient _http = CreateClient();

    private static HttpClient CreateClient()
    {
        var http = new HttpClient { Timeout = TimeSpan.FromHours(2) };
        http.DefaultRequestHeaders.UserAgent.Add(new ProductInfoHeaderValue("HomeBred-LLM", "1.0"));
        http.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/vnd.github+json"));
        return http;
    }

    /// <summary>llama.cpp publishes Windows CUDA/Vulkan and Linux Vulkan builds (no Linux CUDA, no macOS GPU zips needed).</summary>
    public static bool CanInstall(GpuBackendKind kind) =>
        (OperatingSystem.IsWindows() && kind is GpuBackendKind.Cuda or GpuBackendKind.Vulkan) ||
        (OperatingSystem.IsLinux() && kind == GpuBackendKind.Vulkan);

    public static string Unsupported(GpuBackendKind kind) =>
        OperatingSystem.IsLinux() && kind == GpuBackendKind.Cuda
            ? "llama.cpp publishes no prebuilt CUDA build for Linux. Use the Vulkan backend, or build it yourself: native/llama.cpp-hbec/build.sh cuda."
            : "Automatic setup is not available for this platform — see docs/gpu-setup.md.";

    public async Task InstallAsync(GpuBackendKind kind, string targetDir, IProgress<string> status,
        IProgress<double> percent, CancellationToken ct = default)
    {
        if (!CanInstall(kind)) throw new NotSupportedException(Unsupported(kind));

        status.Report($"Looking up llama.cpp {GpuRequirementsChecker.LlamaCppTag}…");
        var assets = await ResolveAssetsAsync(kind, ct);
        var total = assets.Sum(a => a.Size);
        long done = 0;

        Directory.CreateDirectory(targetDir);
        var tmpRoot = Path.Combine(targetDir, ".download");
        Directory.CreateDirectory(tmpRoot);
        try
        {
            foreach (var a in assets)
            {
                var file = Path.Combine(tmpRoot, a.Name);
                status.Report($"Downloading {a.Name} ({a.Size / 1_000_000.0:F0} MB)…");
                var baseDone = done;
                await DownloadAsync(a, file, n => percent.Report(total > 0 ? (baseDone + n) * 100.0 / total : 0), ct);
                done += a.Size;

                status.Report($"Unpacking {a.Name}…");
                await Task.Run(() => Extract(file, targetDir), ct);
            }
            File.WriteAllText(Path.Combine(targetDir, "HOMEBRED-GPU-BACKEND.txt"),
                $"llama.cpp {GpuRequirementsChecker.LlamaCppTag} · {kind}\n{string.Join('\n', assets.Select(a => a.Name))}\n{DateTime.UtcNow:O}\n");
            percent.Report(100);
            status.Report("Installed. Restart the app to load the GPU backend.");
        }
        finally
        {
            try { Directory.Delete(tmpRoot, true); } catch { /* best effort */ }
        }
    }

    private async Task<List<Asset>> ResolveAssetsAsync(GpuBackendKind kind, CancellationToken ct)
    {
        using var resp = await _http.GetAsync(ReleaseApi, ct);
        if (!resp.IsSuccessStatusCode)
            throw new InvalidOperationException($"GitHub returned {(int)resp.StatusCode} for llama.cpp {GpuRequirementsChecker.LlamaCppTag}. Check your connection, or install manually (docs/gpu-setup.md).");
        using var doc = await JsonDocument.ParseAsync(await resp.Content.ReadAsStreamAsync(ct), cancellationToken: ct);

        var all = doc.RootElement.GetProperty("assets").EnumerateArray().Select(e =>
        {
            string? sha = null;
            if (e.TryGetProperty("digest", out var d) && d.ValueKind == JsonValueKind.String &&
                d.GetString() is { } ds && ds.StartsWith("sha256:", StringComparison.OrdinalIgnoreCase))
                sha = ds[7..];
            return new Asset(e.GetProperty("name").GetString()!, e.GetProperty("size").GetInt64(),
                e.GetProperty("browser_download_url").GetString()!, sha);
        }).ToList();

        var tag = Regex.Escape(GpuRequirementsChecker.LlamaCppTag);
        if (kind == GpuBackendKind.Vulkan)
        {
            var rx = OperatingSystem.IsWindows()
                ? new Regex($@"^llama-{tag}-bin-win-vulkan-x64\.zip$")
                : new Regex($@"^llama-{tag}-bin-ubuntu-vulkan-x64\.(zip|tar\.gz)$");
            return [Pick(all, rx, "Vulkan build")];
        }

        // CUDA 12.x on Windows: the build and the matching runtime DLLs (cudart archive).
        var build = new Regex($@"^llama-{tag}-bin-win-cuda-(12\.\d+)-x64\.zip$");
        var cudart = new Regex(@"^cudart-llama-bin-win-cuda-(12\.\d+)-x64\.zip$");
        var bestVersion = all.Select(a => build.Match(a.Name)).Where(m => m.Success)
            .Select(m => Version.Parse(m.Groups[1].Value)).OrderByDescending(v => v).FirstOrDefault()
            ?? throw new InvalidOperationException($"No CUDA 12 build found in llama.cpp {GpuRequirementsChecker.LlamaCppTag}; use Vulkan or install manually.");
        var v = bestVersion.ToString();
        return
        [
            Pick(all, new Regex($@"^llama-{tag}-bin-win-cuda-{Regex.Escape(v)}-x64\.zip$"), "CUDA build"),
            Pick(all, new Regex($@"^cudart-llama-bin-win-cuda-{Regex.Escape(v)}-x64\.zip$"), "CUDA runtime"),
        ];
    }

    private static Asset Pick(List<Asset> all, Regex rx, string what) =>
        all.FirstOrDefault(a => rx.IsMatch(a.Name))
        ?? throw new InvalidOperationException($"The {what} was not found in llama.cpp {GpuRequirementsChecker.LlamaCppTag} releases. Install manually (docs/gpu-setup.md).");

    private async Task DownloadAsync(Asset a, string dest, Action<long> progress, CancellationToken ct)
    {
        using var resp = await _http.GetAsync(a.Url, HttpCompletionOption.ResponseHeadersRead, ct);
        resp.EnsureSuccessStatusCode();
        using var sha = SHA256.Create();
        await using (var src = await resp.Content.ReadAsStreamAsync(ct))
        await using (var dst = new FileStream(dest, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 16, true))
        {
            var buf = new byte[1 << 16];
            long n = 0;
            int read;
            while ((read = await src.ReadAsync(buf, ct)) > 0)
            {
                await dst.WriteAsync(buf.AsMemory(0, read), ct);
                sha.TransformBlock(buf, 0, read, null, 0);
                n += read;
                progress(n);
            }
        }
        sha.TransformFinalBlock([], 0, 0);
        if (a.Sha256 is not null &&
            !string.Equals(Convert.ToHexString(sha.Hash!), a.Sha256, StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException($"Checksum mismatch for {a.Name}; the download is corrupt. Try again.");
    }

    private static bool IsLibrary(string name) =>
        name.EndsWith(".dll", StringComparison.OrdinalIgnoreCase) ||
        name.EndsWith(".so", StringComparison.Ordinal) || name.Contains(".so.", StringComparison.Ordinal);

    /// <summary>Flattens the archive's libraries (no tools/executables) into <paramref name="targetDir"/>.</summary>
    internal static void Extract(string archive, string targetDir)
    {
        if (archive.EndsWith(".zip", StringComparison.OrdinalIgnoreCase))
        {
            using var zip = ZipFile.OpenRead(archive);
            foreach (var e in zip.Entries)
            {
                var name = Path.GetFileName(e.FullName);
                if (name.Length == 0 || !IsLibrary(name)) continue;
                e.ExtractToFile(SafeTarget(targetDir, name), overwrite: true);
            }
            return;
        }

        using var gz = new GZipStream(File.OpenRead(archive), CompressionMode.Decompress);
        using var tar = new TarReader(gz);
        var links = new List<(string Name, string Target)>();
        while (tar.GetNextEntry() is { } entry)
        {
            var name = Path.GetFileName(entry.Name);
            if (name.Length == 0 || !IsLibrary(name)) continue;
            if (entry.EntryType == TarEntryType.SymbolicLink) { links.Add((name, Path.GetFileName(entry.LinkName))); continue; }
            if (entry.DataStream is null) continue;
            using var fs = File.Create(SafeTarget(targetDir, name));
            entry.DataStream.CopyTo(fs);
        }
        foreach (var (name, target) in links) // symlinks → plain copies (portable)
        {
            var src = Path.Combine(targetDir, target);
            if (File.Exists(src)) File.Copy(src, SafeTarget(targetDir, name), overwrite: true);
        }
    }

    private static string SafeTarget(string dir, string name)
    {
        var full = Path.GetFullPath(Path.Combine(dir, name));
        if (!full.StartsWith(Path.GetFullPath(dir), StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException("Archive entry escapes the target folder.");
        return full;
    }
}
