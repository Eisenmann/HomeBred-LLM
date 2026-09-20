using HomebredLLM.Data;
using HomebredLLM.Models;
using HomebredLLM.Services.Gguf;
using Microsoft.EntityFrameworkCore;

namespace HomebredLLM.Services;

/// <summary>
/// Validates a user-picked model — an ONNX export directory, or a single
/// GGUF v2/v3 file — and registers it in the Model Library. Both are used
/// in-place (no copy), since model files are often many gigabytes.
/// </summary>
public sealed class LocalModelImportService(
    IDbContextFactory<AppDbContext> dbFactory,
    OnnxModelMetadataReader onnxMetadataReader,
    GgufMetadataReader ggufMetadataReader)
{
    /// <summary>
    /// Imports a GGUF file (by its .gguf path) or an ONNX export (by the
    /// model.onnx file or its containing directory). Throws
    /// <see cref="FileNotFoundException"/>/<see cref="DirectoryNotFoundException"/>
    /// or <see cref="InvalidDataException"/> if the path isn't a valid model.
    /// </summary>
    public Task<LocalModel> ImportAsync(string sourcePath, CancellationToken ct = default) =>
        File.Exists(sourcePath) && sourcePath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)
            ? ImportGgufAsync(sourcePath, ct)
            : ImportOnnxAsync(sourcePath, ct);

    private async Task<LocalModel> ImportGgufAsync(string filePath, CancellationToken ct)
    {
        var fullPath = Path.GetFullPath(filePath);

        await using var db = await dbFactory.CreateDbContextAsync(ct);
        if (await db.Models.AnyAsync(m => m.LocalPath == fullPath, ct))
            throw new InvalidDataException("This GGUF file is already imported.");

        var metadata = await ggufMetadataReader.ReadMetadataAsync(fullPath, ct)
            ?? throw new InvalidDataException($"'{Path.GetFileName(fullPath)}' is not a valid GGUF v2/v3 file.");

        // Import-time diagnostics, mirroring what the load-time checks
        // (ValidateTensorNaming / ValidateVocabSize) will enforce. These are
        // non-blocking — the file is still importable so it shows up in the
        // library with a warning badge; loading it will produce the specific
        // error. This is consistent with how HasHfStyleTensorNames was
        // decided for the naming case (non-blocking, surfaced in the UI).
        var warnings = new List<string>();
        if (metadata.HasHfStyleTensorNames)
            warnings.Add("HuggingFace-style tensor names — cannot be loaded by llama.cpp; repair and re-import");
        if (metadata.VocabSizeMismatch)
            warnings.Add($"Tokenizer/embedding vocab size mismatch — tokenizer.ggml.tokens has {metadata.VocabSize} entries but token_embd.weight differs; regenerate the file with a corrected converter");

        var model = new LocalModel
        {
            Name = metadata.Name,
            LocalPath = fullPath,
            Format = ModelFormat.Gguf,
            FileSizeBytes = new FileInfo(fullPath).Length,
            Quantization = metadata.Quantization,
            Architecture = metadata.Architecture,
            ParameterCount = metadata.ParameterCount,
            ContextLength = metadata.ContextLength is > 0 and < int.MaxValue
                ? (int)metadata.ContextLength.Value
                : null,
            Warnings = warnings.Count > 0 ? string.Join("; ", warnings) : null,
            Status = ModelStatus.Ready,
        };

        var cfg = new ModelConfiguration
        {
            ModelId = model.Id,
            ContextSize = model.ContextLength is > 512 and <= 131072 ? model.ContextLength.Value : 4096,
        };

        db.Models.Add(model);
        db.ModelConfigurations.Add(cfg);
        await db.SaveChangesAsync(ct);
        return model;
    }

    private async Task<LocalModel> ImportOnnxAsync(string sourcePath, CancellationToken ct)
    {
        // sourcePath can be a directory or a file within the model directory
        var modelDir = File.Exists(sourcePath) ? Path.GetDirectoryName(sourcePath)! : sourcePath;
        if (!Directory.Exists(modelDir))
            throw new DirectoryNotFoundException($"Model directory not found: {modelDir}");

        var hasOnnxFile = File.Exists(Path.Combine(modelDir, "model.onnx")) ||
                          File.Exists(Path.Combine(modelDir, "model.onnx.data"));
        if (!hasOnnxFile)
            throw new InvalidDataException(
                "The selected path is not a valid ONNX model export or a .gguf file. " +
                "An ONNX export must contain a model.onnx (or model.onnx.data) file, " +
                "along with config.json and tokenizer files. GGUF models are imported " +
                "by picking the .gguf file directly.");

        var metadata = await onnxMetadataReader.ReadMetadataAsync(modelDir, ct)
            ?? throw new InvalidDataException(
                "This directory is not a valid ONNX model. " +
                "No config.json was found alongside the model.onnx file. " +
                "Ensure the directory contains a complete ONNX export with config.json.");

        var name = string.IsNullOrWhiteSpace(metadata.Name) ? Path.GetFileName(modelDir) : metadata.Name;
        var localPath = Path.GetFullPath(modelDir);

        await using var db = await dbFactory.CreateDbContextAsync(ct);
        if (await db.Models.AnyAsync(m => m.LocalPath == localPath, ct))
            throw new InvalidDataException("This model directory is already imported.");

        long totalSize = 0;
        foreach (var f in Directory.EnumerateFiles(modelDir, "*.onnx", SearchOption.TopDirectoryOnly))
        {
            try { totalSize += new FileInfo(f).Length; } catch { /* ignore */ }
        }
        foreach (var f in Directory.EnumerateFiles(modelDir, "*.onnx.data", SearchOption.TopDirectoryOnly))
        {
            try { totalSize += new FileInfo(f).Length; } catch { /* ignore */ }
        }
        if (totalSize == 0) totalSize = new DirectoryInfo(modelDir).EnumerateFiles().Sum(f => f.Length);

        var model = new LocalModel
        {
            Name = name,
            LocalPath = localPath,
            Format = ModelFormat.Onnx,
            FileSizeBytes = totalSize,
            Quantization = metadata.Quantization,
            Architecture = metadata.Architecture,
            ParameterCount = metadata.ParameterCount,
            ContextLength = metadata.ContextLength is > 0 and < int.MaxValue
                ? (int)metadata.ContextLength
                : null,
            Status = ModelStatus.Ready,
        };

        var cfg = new ModelConfiguration
        {
            ModelId = model.Id,
            ContextSize = model.ContextLength is > 512 and <= 131072
                ? model.ContextLength.Value
                : 4096,
        };

        db.Models.Add(model);
        db.ModelConfigurations.Add(cfg);
        await db.SaveChangesAsync(ct);

        return model;
    }
}
