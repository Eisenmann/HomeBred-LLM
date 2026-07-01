namespace HomebredLLM.Services;

/// <summary>
/// Validates a user-picked GGUF LoRA adapter and copies it into the app's adapters
/// directory so it survives even if the user deletes the original. Training adapters
/// is out of scope (needs an external Python toolchain) — this only imports files
/// that were produced elsewhere and converted to GGUF.
/// </summary>
public sealed class LoraImportService
{
    // GGUF files start with the magic bytes 0x47 0x47 0x55 0x46 ("GGUF").
    private static readonly byte[] GgufMagic = "GGUF"u8.ToArray();

    /// <summary>
    /// Copies the adapter into the adapters directory and returns the stored path.
    /// Throws if the file is missing or is not a GGUF file.
    /// </summary>
    public async Task<string> ImportAsync(string sourcePath, CancellationToken ct = default)
    {
        if (!File.Exists(sourcePath))
            throw new FileNotFoundException("Adapter file not found.", sourcePath);

        if (!await IsGgufAsync(sourcePath, ct))
            throw new InvalidDataException(
                "This file is not a GGUF LoRA adapter. Convert PEFT/safetensors adapters " +
                "with llama.cpp's convert_lora_to_gguf.py first.");

        var fileName = Path.GetFileName(sourcePath);
        var destPath = Path.Combine(AppPaths.AdaptersDirectory, fileName);

        // Avoid clobbering an existing adapter of the same name.
        if (File.Exists(destPath) && !PathsEqual(sourcePath, destPath))
        {
            var stem = Path.GetFileNameWithoutExtension(fileName);
            var ext = Path.GetExtension(fileName);
            var n = 1;
            do { destPath = Path.Combine(AppPaths.AdaptersDirectory, $"{stem} ({n++}){ext}"); }
            while (File.Exists(destPath));
        }

        if (!PathsEqual(sourcePath, destPath))
        {
            await using var src = File.OpenRead(sourcePath);
            await using var dst = File.Create(destPath);
            await src.CopyToAsync(dst, ct);
        }

        return destPath;
    }

    private static async Task<bool> IsGgufAsync(string path, CancellationToken ct)
    {
        try
        {
            await using var fs = File.OpenRead(path);
            var header = new byte[4];
            var read = await fs.ReadAsync(header, ct);
            return read == 4 && header.AsSpan().SequenceEqual(GgufMagic);
        }
        catch
        {
            return false;
        }
    }

    private static bool PathsEqual(string a, string b) =>
        string.Equals(Path.GetFullPath(a), Path.GetFullPath(b), StringComparison.OrdinalIgnoreCase);
}
