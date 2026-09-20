using HomebredLLM.Services.Gguf;

namespace HomebredLLM.Services;

/// <summary>
/// Validates a GGUF LoRA adapter file and copies it into the app's adapters
/// directory. Adapters are applied at model-load time by
/// <see cref="LlamaCppInferenceService"/> (LLamaSharp/llama.cpp has no
/// training API — the adapter must already be trained elsewhere, e.g. with
/// llama.cpp's finetune tooling or exported from a PEFT/LoRA training run
/// and converted to GGUF).
///
/// GGUF-only: ONNX Runtime GenAI has no adapter support, so LoRA adapters
/// can only be attached to models with <c>LocalModel.Format == ModelFormat.Gguf</c>.
/// </summary>
public sealed class LoraImportService
{
    public async Task<string> ImportAsync(string sourcePath, CancellationToken ct = default)
    {
        if (!File.Exists(sourcePath))
            throw new FileNotFoundException("LoRA adapter file not found.", sourcePath);
        if (!sourcePath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException("LoRA adapters must be GGUF files (.gguf).");

        GgufFile gguf;
        try
        {
            gguf = await GgufReader.ReadAsync(sourcePath, ct);
        }
        catch (Exception ex) when (ex is InvalidDataException or NotSupportedException)
        {
            throw new InvalidDataException(
                $"'{Path.GetFileName(sourcePath)}' is not a valid GGUF v2/v3 LoRA adapter: {ex.Message}", ex);
        }

        // llama.cpp's convention is to tag adapter GGUFs with adapter.type =
        // "lora". Tolerate its absence (some export tools omit it) but reject
        // anything explicitly tagged as something else, e.g. a control vector.
        if (gguf.GetString("adapter.type") is { } adapterType &&
            !adapterType.Equals("lora", StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException(
                $"'{Path.GetFileName(sourcePath)}' is a GGUF '{adapterType}' file, not a LoRA adapter.");

        Directory.CreateDirectory(AppPaths.AdaptersDirectory);
        var destPath = Path.Combine(AppPaths.AdaptersDirectory, Path.GetFileName(sourcePath));

        if (PathsRefEqual(sourcePath, destPath))
            return destPath;

        if (File.Exists(destPath))
            destPath = Path.Combine(
                AppPaths.AdaptersDirectory,
                $"{Path.GetFileNameWithoutExtension(sourcePath)}_{Guid.NewGuid():N}.gguf");

        await CopyFileAsync(sourcePath, destPath, ct);
        return destPath;
    }

    private static bool PathsRefEqual(string a, string b) =>
        string.Equals(Path.GetFullPath(a), Path.GetFullPath(b), StringComparison.OrdinalIgnoreCase);

    private static async Task CopyFileAsync(string source, string dest, CancellationToken ct)
    {
        await using var src = new FileStream(
            source, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, useAsync: true);
        await using var dst = new FileStream(
            dest, FileMode.CreateNew, FileAccess.Write, FileShare.None, 1 << 20, useAsync: true);
        await src.CopyToAsync(dst, ct);
    }
}
