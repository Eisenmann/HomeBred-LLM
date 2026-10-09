using System.Globalization;
using HomebredLLM.Services.Gguf;

namespace HomebredLLM.Services.Tiering;

/// <summary>What role a weight tensor plays — decides how often it is read per token.</summary>
public enum TensorKind
{
    /// <summary>token_embd: a row lookup per token, not a matmul; llama.cpp keeps it on CPU.</summary>
    Embedding,
    /// <summary>Output head (lm_head): read fully every token.</summary>
    Output,
    OutputNorm,
    Attention,
    Norm,
    /// <summary>MoE router (ffn_gate_inp) and its biases.</summary>
    Router,
    /// <summary>Shared experts (always active in DeepSeek/Qwen-style MoE).</summary>
    SharedExpert,
    /// <summary>Dense FFN (dense models, or leading dense layers of MoE models).</summary>
    DenseFfn,
    /// <summary>Fused routed-expert tensor: only n_used of n_expert slices are read per token.</summary>
    ExpertFfn,
    Other,
}

/// <summary>One weight tensor with everything the placement planner needs.</summary>
public sealed record CatalogTensor(
    string Name,
    int Layer,
    TensorKind Kind,
    GgmlType Type,
    long Elements,
    long Bytes,
    long FileOffset,
    int ExpertCount,
    long BytesPerExpert)
{
    public bool IsExpert => Kind == TensorKind.ExpertFfn && ExpertCount > 1;

    /// <summary>Read on every token regardless of routing.</summary>
    public bool IsAlwaysOn => Kind is not (TensorKind.ExpertFfn or TensorKind.Embedding);

    /// <summary>Absolute file offset of one expert's contiguous slice.</summary>
    public long ExpertOffset(int expert) => FileOffset + expert * BytesPerExpert;
}

/// <summary>
/// Classified list of a model's weight tensors with byte sizes and file
/// offsets. Built from a GGUF tensor directory (exact) or synthesized from a
/// <see cref="ModelShape"/> (capacity calculator).
/// </summary>
public sealed class TensorCatalog
{
    public required ModelShape Shape { get; init; }
    public required IReadOnlyList<CatalogTensor> Tensors { get; init; }

    /// <summary>GGUF path for real files (needed by the warm tier); null when synthetic.</summary>
    public string? FilePath { get; init; }

    public long TotalBytes => Tensors.Sum(t => t.Bytes);
    public long ExpertBytes => Tensors.Where(t => t.IsExpert).Sum(t => t.Bytes);
    public bool IsMoe => Shape.IsMoe && Tensors.Any(t => t.IsExpert);

    public static bool IsExpertTensorName(string name) =>
        name.Contains("_exps", StringComparison.Ordinal) && !name.Contains("norm", StringComparison.Ordinal);

    public static TensorCatalog FromGguf(GgufFile gguf, string? filePath = null)
    {
        var shape = ModelShape.FromGguf(gguf);
        var list = new List<CatalogTensor>(gguf.Tensors.Count);
        foreach (var t in gguf.Tensors)
        {
            var (layer, kind) = Classify(t.Name);
            var bytes = GgmlTypeInfo.BytesFor(t.Type, t.ElementCount);
            var experts = 0;
            long perExpert = 0;
            if (kind == TensorKind.ExpertFfn && t.Shape.Length >= 2)
            {
                var outer = (int)t.Shape[^1];
                if (outer > 1 && (shape.ExpertCount == 0 || outer == shape.ExpertCount))
                {
                    experts = outer;
                    perExpert = bytes / outer;
                }
            }
            if (kind == TensorKind.ExpertFfn && experts == 0)
                kind = TensorKind.DenseFfn; // fused tensor we can't slice: treat as always-on

            list.Add(new CatalogTensor(t.Name, layer, kind, t.Type, t.ElementCount, bytes,
                gguf.TensorDataStartOffset + (long)t.RelativeOffset, experts, perExpert));
        }

        return new TensorCatalog { Shape = shape, Tensors = list, FilePath = filePath };
    }

    public static async Task<TensorCatalog> FromFileAsync(string ggufPath, CancellationToken ct = default)
    {
        var gguf = await GgufReader.ReadAsync(ggufPath, ct);
        return FromGguf(gguf, ggufPath);
    }

    /// <summary>
    /// Builds a catalog for a hypothetical model. Tensor names follow
    /// llama.cpp's conventions so the same planner code paths apply; byte
    /// sizes come from <paramref name="bitsPerWeight"/>.
    /// </summary>
    public static TensorCatalog Synthetic(ModelShape shape, double bitsPerWeight)
    {
        double B(long parameters) => parameters * bitsPerWeight / 8.0;
        var list = new List<CatalogTensor>();
        long offset = 0;
        void Add(string name, int layer, TensorKind kind, long parameters, int experts = 0)
        {
            var bytes = (long)B(parameters);
            if (experts > 0) bytes = bytes / experts * experts; // whole expert slices, like real GGUF tensors
            list.Add(new CatalogTensor(name, layer, kind, GgmlType.Q4_K, parameters, bytes, offset,
                experts, experts > 0 ? bytes / experts : 0));
            offset += bytes;
        }

        var d = (long)shape.EmbeddingLength;
        var kvDim = shape.LayerCount > 0 ? shape.KvHeadsTotal / shape.LayerCount * shape.HeadDimK : d;
        var qDim = (long)shape.HeadCount * shape.HeadDimK;
        Add("token_embd.weight", -1, TensorKind.Embedding, shape.VocabSize * d);

        long attn = d * qDim * 2 + d * kvDim * 2;
        for (var i = 0; i < shape.LayerCount; i++)
        {
            Add($"blk.{i}.attn_qkvo.weight", i, TensorKind.Attention, attn);
            Add($"blk.{i}.attn_norm.weight", i, TensorKind.Norm, 2 * d);
            if (shape.IsMoe)
            {
                Add($"blk.{i}.ffn_gate_inp.weight", i, TensorKind.Router, d * shape.ExpertCount);
                var perTensor = d * shape.ExpertFeedForwardLength * shape.ExpertCount;
                Add($"blk.{i}.ffn_gate_exps.weight", i, TensorKind.ExpertFfn, perTensor, shape.ExpertCount);
                Add($"blk.{i}.ffn_up_exps.weight", i, TensorKind.ExpertFfn, perTensor, shape.ExpertCount);
                Add($"blk.{i}.ffn_down_exps.weight", i, TensorKind.ExpertFfn, perTensor, shape.ExpertCount);
            }
            else
            {
                Add($"blk.{i}.ffn_gate_up_down.weight", i, TensorKind.DenseFfn, 3 * d * shape.FeedForwardLength);
            }
        }

        // Remaining parameters (shared experts, biases, rounding) go into the
        // always-on part of each layer so the synthetic total matches.
        long counted = list.Sum(t => t.Elements) + shape.VocabSize * d;
        long remainder = shape.TotalParameters - counted;
        if (remainder > 0 && shape.LayerCount > 0 && shape.IsMoe)
        {
            var perLayer = remainder / shape.LayerCount;
            for (var i = 0; i < shape.LayerCount; i++)
                Add($"blk.{i}.ffn_shexp.weight", i, TensorKind.SharedExpert, perLayer);
        }

        Add("output_norm.weight", -1, TensorKind.OutputNorm, d);
        Add("output.weight", -1, TensorKind.Output, shape.VocabSize * d);

        return new TensorCatalog { Shape = shape, Tensors = list };
    }

    internal static (int Layer, TensorKind Kind) Classify(string name)
    {
        if (name.Contains("token_embd", StringComparison.Ordinal))
            return (-1, name.Contains("norm", StringComparison.Ordinal) ? TensorKind.Norm : TensorKind.Embedding);
        if (name.StartsWith("output_norm", StringComparison.Ordinal)) return (-1, TensorKind.OutputNorm);
        if (name.StartsWith("output.", StringComparison.Ordinal)) return (-1, TensorKind.Output);

        if (!name.StartsWith("blk.", StringComparison.Ordinal)) return (-1, TensorKind.Other);
        var dot = name.IndexOf('.', 4);
        if (dot < 0 || !int.TryParse(name.AsSpan(4, dot - 4), NumberStyles.None, CultureInfo.InvariantCulture, out var layer))
            return (-1, TensorKind.Other);
        var rest = name[(dot + 1)..];

        if (rest.Contains("norm", StringComparison.Ordinal)) return (layer, TensorKind.Norm);
        if (rest.Contains("_exps", StringComparison.Ordinal)) return (layer, TensorKind.ExpertFfn);
        if (rest.Contains("shexp", StringComparison.Ordinal)) return (layer, TensorKind.SharedExpert);
        if (rest.StartsWith("ffn_gate_inp", StringComparison.Ordinal) || rest.StartsWith("exp_probs", StringComparison.Ordinal))
            return (layer, TensorKind.Router);
        if (rest.StartsWith("attn", StringComparison.Ordinal)) return (layer, TensorKind.Attention);
        if (rest.StartsWith("ffn_", StringComparison.Ordinal)) return (layer, TensorKind.DenseFfn);
        return (layer, TensorKind.Other);
    }
}
