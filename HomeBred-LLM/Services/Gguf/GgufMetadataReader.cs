namespace HomebredLLM.Services.Gguf;

/// <summary>
/// Metadata about a GGUF model file, read from its header/KV block without
/// loading tensor weights. Field names deliberately mirror
/// <see cref="HomebredLLM.Services.OnnxModelMetadata"/> so both formats can
/// populate the same Model Library UI.
/// </summary>
public sealed record GgufModelMetadata(
    string Name,
    string? Architecture,
    string? Quantization,
    long? ParameterCount,
    long? ContextLength,
    long EmbeddingLength,
    long BlockCount,
    long AttentionHeadCount,
    long VocabSize,
    bool HasChatTemplate,
    uint GgufVersion,
    bool HasHfStyleTensorNames,
    bool VocabSizeMismatch);

/// <summary>
/// Reads metadata from a .gguf file using <see cref="GgufReader"/>. Cheap
/// enough to run on import and whenever the Model Library needs to refresh
/// a card, since it never reads tensor weight bytes.
/// </summary>
public sealed class GgufMetadataReader
{
    /// <summary>
    /// Reads metadata from a GGUF file. Returns null if the path doesn't exist
    /// or isn't a .gguf file. Throws <see cref="InvalidDataException"/> or
    /// <see cref="NotSupportedException"/> if the file exists but isn't a
    /// valid GGUF v2/v3 file.
    /// </summary>
    public async Task<GgufModelMetadata?> ReadMetadataAsync(string filePath, CancellationToken ct = default)
    {
        if (!File.Exists(filePath) || !filePath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
            return null;

        GgufFile gguf;
        try
        {
            gguf = await GgufReader.ReadAsync(filePath, ct);
        }
        catch (Exception ex) when (ex is InvalidDataException or NotSupportedException)
        {
            throw new InvalidDataException(
                $"'{Path.GetFileName(filePath)}' is not a valid GGUF v2/v3 file: {ex.Message}", ex);
        }

        var arch = gguf.GetString("general.architecture");
        string Prefixed(string suffix) => arch is null ? suffix : $"{arch}.{suffix}";

        var name = gguf.GetString("general.name");
        if (string.IsNullOrWhiteSpace(name))
            name = Path.GetFileNameWithoutExtension(filePath);

        var vocabSize = gguf.Metadata.TryGetValue("tokenizer.ggml.tokens", out var toks) && toks is List<object?> tokList
            ? tokList.Count
            : gguf.GetInt(Prefixed("vocab_size")) ?? 0;

        long paramCount = 0;
        foreach (var t in gguf.Tensors)
            paramCount += t.ElementCount;

        return new GgufModelMetadata(
            Name: name!,
            Architecture: arch,
            Quantization: InferQuantization(gguf),
            ParameterCount: paramCount > 0 ? paramCount : null,
            ContextLength: gguf.GetInt(Prefixed("context_length")),
            EmbeddingLength: gguf.GetInt(Prefixed("embedding_length")) ?? 0,
            BlockCount: gguf.GetInt(Prefixed("block_count")) ?? 0,
            AttentionHeadCount: gguf.GetInt(Prefixed("attention.head_count")) ?? 0,
            VocabSize: vocabSize,
            HasChatTemplate: gguf.Metadata.ContainsKey("tokenizer.chat_template"),
            GgufVersion: gguf.Version,
            // Same heuristic LLamaCppInferenceService.ValidateTensorNaming uses at
            // load time, surfaced here (non-blocking) so the import step can warn
            // about a file that will never load. Deliberately NOT used to block
            // import: the heuristic is conservative (only flags the self_attn./mlp.
            // substrings plus a "model." prefix) and must not reject a genuinely
            // loadable file on a false positive. See the class header of
            // LlamaCppInferenceService for the reasoning decision.
            HasHfStyleTensorNames: HasHfStyleTensorNames(gguf),
            // A file whose tokenizer vocabulary count (tokenizer.ggml.tokens) does
            // not match its token_embd.weight tensor width will fail llama.cpp's
            // check_tensor_dims at load time. Surfaced (non-blocking) just like
            // HasHfStyleTensorNames — a mismatch means the model needs regeneration,
            // not that this app's loader is at fault.
            VocabSizeMismatch: vocabSize > 0 && TokenEmbeddingWidth(gguf) != vocabSize);
    }

    private static long TokenEmbeddingWidth(GgufFile gguf)
    {
        foreach (var t in gguf.Tensors)
        {
            if (t.Name == "token_embd.weight" && t.Shape.Length >= 2)
                return t.Shape[1];
        }
        return 0;
    }

    private static bool HasHfStyleTensorNames(GgufFile gguf)
    {
        const int sampleLimit = 5000; // same cap as the load-time check
        var count = 0;
        foreach (var t in gguf.Tensors)
        {
            if (++count > sampleLimit) break;
            var n = t.Name;
            if (n.StartsWith("model.", StringComparison.Ordinal) ||
                n.Contains(".self_attn.", StringComparison.Ordinal) ||
                n.Contains(".mlp.", StringComparison.Ordinal))
                return true;
        }
        return false;
    }

    private static string? InferQuantization(GgufFile gguf)
    {
        // general.file_type is llama.cpp's own ggml_ftype enum and is the most
        // reliable label when present.
        if (gguf.Metadata.TryGetValue("general.file_type", out var ft) && ft is not null)
            return FileTypeName(Convert.ToUInt32(ft));

        // Fall back to a majority vote across weight tensors (2+ dims), since
        // 1D tensors like norms/biases are almost always kept at F32
        // regardless of the model's overall quantization.
        var dominant = gguf.Tensors
            .Where(t => t.Shape.Length > 1)
            .GroupBy(t => t.Type)
            .OrderByDescending(g => g.Count())
            .FirstOrDefault();
        return dominant?.Key.ToString();
    }

    // Matches llama.cpp's LLAMA_FTYPE_* enum (a superset of ggml_ftype),
    // which is what models typically store in general.file_type.
    private static string FileTypeName(uint ft) => ft switch
    {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "Q2_K_S",
        22 => "IQ3_XS",
        23 => "IQ3_XXS",
        24 => "IQ1_S",
        25 => "IQ4_NL",
        26 => "IQ3_S",
        27 => "IQ3_M",
        28 => "IQ2_S",
        29 => "IQ2_M",
        30 => "IQ4_XS",
        31 => "IQ1_M",
        32 => "BF16",
        36 => "TQ1_0",
        37 => "TQ2_0",
        _ => $"ftype{ft}",
    };
}
