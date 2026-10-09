using HomebredLLM.Services.Gguf;

namespace HomebredLLM.Services.Tiering;

/// <summary>
/// The architecture numbers the memory/speed estimates need. Read from GGUF
/// metadata for a real file, or synthesized from a parameter count for the
/// capacity calculator ("what if I had a 70B model?").
/// </summary>
public sealed record ModelShape
{
    public string Architecture { get; init; } = "unknown";
    public int LayerCount { get; init; }
    public int EmbeddingLength { get; init; }
    public int HeadCount { get; init; }

    /// <summary>Sum over layers of KV heads (handles per-layer head_count_kv arrays).</summary>
    public long KvHeadsTotal { get; init; }

    public int HeadDimK { get; init; }
    public int HeadDimV { get; init; }
    public int VocabSize { get; init; }
    public int FeedForwardLength { get; init; }
    public int ExpertCount { get; init; }
    public int ExpertUsedCount { get; init; }
    public int ExpertFeedForwardLength { get; init; }
    public int LeadingDenseLayers { get; init; }

    /// <summary>Multi-head latent attention (DeepSeek-V2/V3 style) compressed KV rank; 0 if not MLA.</summary>
    public int KvLoraRank { get; init; }
    public int RopeDim { get; init; }

    public long TotalParameters { get; init; }
    public long ActiveParameters { get; init; }

    /// <summary>True when the shape came from a parameter count, not a real file.</summary>
    public bool IsSynthetic { get; init; }

    public bool IsMoe => ExpertCount > 1 && ExpertUsedCount > 0;

    /// <summary>Fraction of expert weights read per token (n_used / n_expert); 1 for dense models.</summary>
    public double ExpertReadFraction => IsMoe ? (double)ExpertUsedCount / ExpertCount : 1.0;

    public static ModelShape FromGguf(GgufFile gguf)
    {
        var arch = gguf.GetString("general.architecture") ?? "unknown";
        long? I(string suffix) => gguf.GetInt($"{arch}.{suffix}");

        var layers = (int)(I("block_count") ?? 0);
        var embd = (int)(I("embedding_length") ?? 0);
        var heads = (int)(I("attention.head_count") ?? MaxOfArray(gguf, $"{arch}.attention.head_count") ?? 0);

        // head_count_kv may be a scalar or a per-layer array (e.g. OpenELM, hybrid models).
        long kvTotal;
        if (gguf.Metadata.TryGetValue($"{arch}.attention.head_count_kv", out var kvRaw) && kvRaw is List<object?> kvList)
            kvTotal = kvList.Where(v => v is not null).Sum(v => Convert.ToInt64(v));
        else
            kvTotal = (I("attention.head_count_kv") ?? heads) * (long)layers;

        var headDimK = (int)(I("attention.key_length") ?? (heads > 0 ? embd / heads : 0));
        var headDimV = (int)(I("attention.value_length") ?? headDimK);

        var vocab = gguf.Metadata.TryGetValue("tokenizer.ggml.tokens", out var toks) && toks is List<object?> tokList
            ? tokList.Count
            : (int)(I("vocab_size") ?? 0);

        var ff = (int)(I("feed_forward_length") ?? MaxOfArray(gguf, $"{arch}.feed_forward_length") ?? 0);
        var experts = (int)(I("expert_count") ?? 0);
        var used = (int)(I("expert_used_count") ?? 0);
        var expFf = (int)(I("expert_feed_forward_length") ?? 0);
        var leadDense = (int)(I("leading_dense_block_count") ?? 0);
        var kvLora = (int)(I("attention.kv_lora_rank") ?? 0);
        var rope = (int)(I("rope.dimension_count") ?? 0);

        long total = 0, expertElements = 0;
        foreach (var t in gguf.Tensors)
        {
            total += t.ElementCount;
            if (TensorCatalog.IsExpertTensorName(t.Name)) expertElements += t.ElementCount;
        }

        var readFraction = experts > 1 && used > 0 ? (double)used / experts : 1.0;
        var active = total - expertElements + (long)(expertElements * readFraction);

        return new ModelShape
        {
            Architecture = arch,
            LayerCount = layers,
            EmbeddingLength = embd,
            HeadCount = heads,
            KvHeadsTotal = kvTotal,
            HeadDimK = headDimK,
            HeadDimV = headDimV,
            VocabSize = vocab,
            FeedForwardLength = ff,
            ExpertCount = experts,
            ExpertUsedCount = used,
            ExpertFeedForwardLength = expFf,
            LeadingDenseLayers = leadDense,
            KvLoraRank = kvLora,
            RopeDim = rope,
            TotalParameters = total,
            ActiveParameters = active,
        };
    }

    private static long? MaxOfArray(GgufFile gguf, string key) =>
        gguf.Metadata.TryGetValue(key, out var v) && v is List<object?> list && list.Count > 0
            ? list.Where(x => x is not null).Max(x => Convert.ToInt64(x))
            : null;

    // Typical (params B → layers, hidden) shapes of recent open-weight dense
    // models. Used only to synthesize a believable architecture for "what if"
    // sizes; interpolated in log(params).
    private static readonly (double ParamsB, int Layers, int Hidden)[] ReferenceShapes =
    [
        (0.5, 24, 896),
        (1.5, 28, 1536),
        (3, 36, 2048),
        (8, 32, 4096),
        (14, 48, 5120),
        (32, 64, 5120),
        (70, 80, 8192),
        (123, 88, 12288),
        (405, 126, 16384),
        (1000, 160, 20480),
    ];

    /// <summary>
    /// Synthesizes a shape for a model with <paramref name="totalParamsB"/>
    /// billion parameters. For MoE pass <paramref name="activeParamsB"/>
    /// (parameters used per token) plus the expert counts; the layer count
    /// and hidden size follow the active size, which is what decides them in
    /// practice (e.g. 235B-A22B models look like ~22B dense models with wide
    /// expert FFNs).
    /// </summary>
    public static ModelShape Synthetic(double totalParamsB, double? activeParamsB = null,
        int expertCount = 128, int expertUsedCount = 8, int vocab = 128_000)
    {
        totalParamsB = Math.Max(0.05, totalParamsB);
        var isMoe = activeParamsB is > 0 && activeParamsB < totalParamsB && expertCount > 1 && expertUsedCount > 0;
        var sizeForShape = isMoe ? activeParamsB!.Value : totalParamsB;
        var (layers, hidden) = Interpolate(sizeForShape);

        const int headDim = 128;
        const int kvHeads = 8;
        var heads = Math.Max(1, hidden / headDim);
        var total = (long)(totalParamsB * 1e9);
        var active = isMoe ? (long)(activeParamsB!.Value * 1e9) : total;

        // FFN width from what is left after embeddings + attention.
        long embedParams = 2L * vocab * hidden;
        long attnPerLayer = 2L * hidden * hidden + 2L * hidden * kvHeads * headDim;
        long ffnPerLayer = Math.Max(0, (active - embedParams) / Math.Max(1, layers) - attnPerLayer);
        var ff = (int)Math.Max(hidden, ffnPerLayer / (3L * hidden));

        int expFf = 0;
        if (isMoe)
        {
            // Expert weights total E solves: total = shared + E, active = shared + E*k/n.
            var r = (double)expertUsedCount / expertCount;
            var expertParams = (total - active) / (1 - r);
            expFf = (int)Math.Max(64, expertParams / ((double)layers * expertCount * 3 * hidden));
        }

        return new ModelShape
        {
            Architecture = isMoe ? "synthetic-moe" : "synthetic-dense",
            LayerCount = layers,
            EmbeddingLength = hidden,
            HeadCount = heads,
            KvHeadsTotal = (long)kvHeads * layers,
            HeadDimK = headDim,
            HeadDimV = headDim,
            VocabSize = vocab,
            FeedForwardLength = ff,
            ExpertCount = isMoe ? expertCount : 0,
            ExpertUsedCount = isMoe ? expertUsedCount : 0,
            ExpertFeedForwardLength = expFf,
            TotalParameters = total,
            ActiveParameters = active,
            IsSynthetic = true,
        };
    }

    private static (int Layers, int Hidden) Interpolate(double paramsB)
    {
        var pts = ReferenceShapes;
        if (paramsB <= pts[0].ParamsB) return (pts[0].Layers, pts[0].Hidden);
        if (paramsB >= pts[^1].ParamsB) return (pts[^1].Layers, pts[^1].Hidden);
        for (var i = 1; i < pts.Length; i++)
        {
            if (paramsB > pts[i].ParamsB) continue;
            var a = pts[i - 1];
            var b = pts[i];
            var t = (Math.Log(paramsB) - Math.Log(a.ParamsB)) / (Math.Log(b.ParamsB) - Math.Log(a.ParamsB));
            var layers = (int)Math.Round(a.Layers + t * (b.Layers - a.Layers));
            var hidden = (int)Math.Round((a.Hidden + t * (b.Hidden - a.Hidden)) / 128.0) * 128;
            return (layers, Math.Max(128, hidden));
        }
        return (pts[^1].Layers, pts[^1].Hidden);
    }
}
