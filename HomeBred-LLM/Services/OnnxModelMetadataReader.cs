using System.Text.Json;
using System.Text.Json.Serialization;

namespace HomebredLLM.Services;

/// <summary>
/// Metadata about an ONNX model directory, parsed from config.json and
/// tokenizer files without loading the model weights.
/// </summary>
public record OnnxModelMetadata(
    string Name,
    string? Architecture,
    string? Quantization,
    long? ParameterCount,
    long? ContextLength,
    string? FileType,
    int NumLayers,
    int HiddenSize,
    int NumAttentionHeads,
    string? ModelType);

/// <summary>
/// Reads metadata from an ONNX model directory. ONNX models exported via
/// HuggingFace's optimum-export pipeline ship with a config.json that contains
/// architecture hyperparameters, plus tokenizer files. This reader parses
/// config.json to extract enough information to populate the Model Library UI.
/// </summary>
public sealed class OnnxModelMetadataReader
{
    /// <summary>
    /// Reads metadata from an ONNX model directory. Returns null if the directory
    /// does not contain a valid ONNX model (no config.json or model.onnx).
    /// Throws <see cref="IOException"/> for unreadable files.
    /// </summary>
    public async Task<OnnxModelMetadata?> ReadMetadataAsync(string modelPath, CancellationToken ct = default)
    {
        // modelPath can be either a directory or a file path
        var modelDir = File.Exists(modelPath)
            ? Path.GetDirectoryName(modelPath)!
            : Directory.Exists(modelPath)
                ? modelPath
                : modelPath; // let the checks below handle invalid paths

        if (!Directory.Exists(modelDir))
            throw new DirectoryNotFoundException($"Model directory not found: {modelDir}");

        // Must have model.onnx (or model.onnx.data) to be a valid ONNX export
        var hasOnnxFile = File.Exists(Path.Combine(modelDir, "model.onnx")) ||
                          File.Exists(Path.Combine(modelDir, "model.onnx.data"));
        if (!hasOnnxFile)
            return null;

        var configPath = Path.Combine(modelDir, "config.json");
        if (!File.Exists(configPath))
            return null;

        var json = await File.ReadAllTextAsync(configPath, ct);
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        string? GetString(string key) =>
            root.TryGetProperty(key, out var el) && el.ValueKind == JsonValueKind.String
                ? el.GetString()
                : null;

        long? GetLong(string key)
        {
            if (root.TryGetProperty(key, out var el))
            {
                if (el.ValueKind == JsonValueKind.Number)
                    return el.GetInt64();
                if (el.ValueKind == JsonValueKind.String && long.TryParse(el.GetString(), out var val))
                    return val;
            }
            return null;
        }

        var name = GetString("name") ??
                   GetString("_name") ??
                   Path.GetFileName(Path.GetDirectoryName(modelDir));

        var modelType = GetString("model_type");
        var arch = modelType ?? GetString("architectures") ?? "unknown";
        // architectures field can be an array like ["LlamaForCausalLM"]
        if (root.TryGetProperty("architectures", out var archEl) && archEl.ValueKind == JsonValueKind.Array)
        {
            var first = archEl.EnumerateArray().FirstOrDefault();
            if (first.ValueKind == JsonValueKind.String)
                arch = first.GetString() ?? arch;
        }

        var contextLength = GetLong("max_position_embeddings") ?? GetLong("n_positions") ?? GetLong("seq_length");
        var paramCount = GetLong("num_params") ?? GetLong("total_params");
        var numLayers = GetLong("num_hidden_layers") ?? GetLong("n_layers") ?? GetLong("num_layers");
        var hiddenSize = GetLong("hidden_size") ?? GetLong("n_embd") ?? GetLong("d_model");
        var numHeads = GetLong("num_attention_heads") ?? GetLong("n_head") ?? GetLong("num_heads");

        // Quantization is not typically in config.json for ONNX models.
        // Try to infer from directory name or a separate quantization file.
        var quantization = InferQuantization(modelDir);

        return new OnnxModelMetadata(
            Name: name ?? "Unnamed model",
            Architecture: arch,
            Quantization: quantization,
            ParameterCount: paramCount,
            ContextLength: contextLength,
            FileType: "ONNX",
            NumLayers: (int)(numLayers ?? 0),
            HiddenSize: (int)(hiddenSize ?? 0),
            NumAttentionHeads: (int)(numHeads ?? 0),
            ModelType: modelType);
    }

    private static string? InferQuantization(string modelDir)
    {
        var dirName = Path.GetFileName(modelDir);
        // Common quantization patterns in ONNX model directory names
        var match = System.Text.RegularExpressions.Regex.Match(
            dirName,
            @"(Q\d+_[A-Z0-9_]+|INT8|INT4|FP16|FP32|BF16|q\d+|quantized|quant)",
            System.Text.RegularExpressions.RegexOptions.IgnoreCase);
        return match.Success ? match.Value.ToUpperInvariant() : null;
    }
}