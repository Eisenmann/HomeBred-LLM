using HomebredLLM.Services.Gguf;

namespace HomebredLLM.Services.Tiering;

/// <summary>
/// Storage sizes of ggml tensor types (mirrors ggml's <c>type_traits</c>
/// block size / type size table) plus effective bits-per-weight for the
/// common llama.cpp quantization labels ("Q4_K_M", ...), used when only a
/// parameter count is known (capacity calculator, Hugging Face listings).
/// </summary>
public static class GgmlTypeInfo
{
    // (elements per block, bytes per block)
    private static readonly Dictionary<GgmlType, (int Block, int Bytes)> Traits = new()
    {
        [GgmlType.F32] = (1, 4),
        [GgmlType.F16] = (1, 2),
        [GgmlType.BF16] = (1, 2),
        [GgmlType.F64] = (1, 8),
        [GgmlType.I8] = (1, 1),
        [GgmlType.I16] = (1, 2),
        [GgmlType.I32] = (1, 4),
        [GgmlType.I64] = (1, 8),
        [GgmlType.Q4_0] = (32, 18),
        [GgmlType.Q4_1] = (32, 20),
        [GgmlType.Q5_0] = (32, 22),
        [GgmlType.Q5_1] = (32, 24),
        [GgmlType.Q8_0] = (32, 34),
        [GgmlType.Q8_1] = (32, 36),
        [GgmlType.Q2_K] = (256, 84),
        [GgmlType.Q3_K] = (256, 110),
        [GgmlType.Q4_K] = (256, 144),
        [GgmlType.Q5_K] = (256, 176),
        [GgmlType.Q6_K] = (256, 210),
        [GgmlType.Q8_K] = (256, 292),
        [GgmlType.IQ2_XXS] = (256, 66),
        [GgmlType.IQ2_XS] = (256, 74),
        [GgmlType.IQ3_XXS] = (256, 98),
        [GgmlType.IQ1_S] = (256, 50),
        [GgmlType.IQ4_NL] = (32, 18),
        [GgmlType.IQ3_S] = (256, 110),
        [GgmlType.IQ2_S] = (256, 82),
        [GgmlType.IQ4_XS] = (256, 136),
        [GgmlType.IQ1_M] = (256, 56),
        [GgmlType.TQ1_0] = (256, 54),
        [GgmlType.TQ2_0] = (256, 66),
        [GgmlType.MXFP4] = (32, 17),
    };

    /// <summary>True when the type's storage size is known.</summary>
    public static bool IsKnown(GgmlType type) => Traits.ContainsKey(type);

    /// <summary>Bytes needed to store <paramref name="elements"/> values of <paramref name="type"/>.</summary>
    public static long BytesFor(GgmlType type, long elements)
    {
        if (!Traits.TryGetValue(type, out var t))
            t = (1, 2); // unknown future type: assume 16-bit so estimates err on the large side
        var blocks = (elements + t.Block - 1) / t.Block;
        return blocks * t.Bytes;
    }

    /// <summary>Bytes per element (fractional for block-quantized types).</summary>
    public static double BytesPerElement(GgmlType type) =>
        Traits.TryGetValue(type, out var t) ? (double)t.Bytes / t.Block : 2.0;

    /// <summary>
    /// Effective bits per weight of a whole model file for llama.cpp's
    /// quantization presets (includes the higher-precision tensors those
    /// presets keep, e.g. Q6_K output heads in Q4_K_M).
    /// </summary>
    private static readonly Dictionary<string, double> PresetBitsPerWeight = new(StringComparer.OrdinalIgnoreCase)
    {
        ["F32"] = 32.0,
        ["F16"] = 16.0,
        ["BF16"] = 16.0,
        ["Q8_0"] = 8.5,
        ["Q6_K"] = 6.56,
        ["Q5_K_M"] = 5.69,
        ["Q5_K_S"] = 5.54,
        ["Q5_1"] = 6.0,
        ["Q5_0"] = 5.5,
        ["Q4_K_M"] = 4.85,
        ["Q4_K_S"] = 4.58,
        ["Q4_1"] = 5.0,
        ["Q4_0"] = 4.55,
        ["IQ4_NL"] = 4.5,
        ["IQ4_XS"] = 4.25,
        ["MXFP4"] = 4.25,
        ["Q3_K_L"] = 4.27,
        ["Q3_K_M"] = 3.91,
        ["Q3_K_S"] = 3.5,
        ["IQ3_M"] = 3.66,
        ["IQ3_S"] = 3.44,
        ["IQ3_XS"] = 3.3,
        ["IQ3_XXS"] = 3.06,
        ["Q2_K"] = 2.96,
        ["Q2_K_S"] = 2.6,
        ["IQ2_M"] = 2.7,
        ["IQ2_S"] = 2.5,
        ["IQ2_XS"] = 2.31,
        ["IQ2_XXS"] = 2.06,
        ["IQ1_M"] = 1.75,
        ["IQ1_S"] = 1.56,
        ["TQ2_0"] = 2.06,
        ["TQ1_0"] = 1.69,
        // ONNX export labels
        ["INT4"] = 4.5,
        ["INT8"] = 8.5,
        ["FP16"] = 16.0,
        ["FP32"] = 32.0,
    };

    /// <summary>Quantization labels offered by the calculator, largest first.</summary>
    public static readonly string[] CommonQuantLabels =
        ["F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_K_S", "IQ4_XS", "Q3_K_M", "IQ3_XXS", "Q2_K", "IQ2_XXS", "IQ1_S"];

    /// <summary>Bits per weight for a preset label; null when unknown.</summary>
    public static double? BitsPerWeight(string? quantLabel)
    {
        if (string.IsNullOrWhiteSpace(quantLabel)) return null;
        var key = quantLabel.Trim().ToUpperInvariant().Replace('-', '_');
        if (PresetBitsPerWeight.TryGetValue(key, out var bpw)) return bpw;
        // Tolerate labels with suffixes like "Q4_K_M_imatrix" or "UD-Q4_K_XL".
        foreach (var (label, value) in PresetBitsPerWeight.OrderByDescending(p => p.Key.Length))
            if (key.Contains(label, StringComparison.OrdinalIgnoreCase)) return value;
        // Unknown variants ("Q4_K_XL", "IQ5_K"…): base bits + typical K-quant overhead.
        var m = System.Text.RegularExpressions.Regex.Match(key, @"I?Q([1-8])_");
        return m.Success ? int.Parse(m.Groups[1].Value) + 0.85 : null;
    }
}
