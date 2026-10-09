namespace HomebredLLM.Services.Tiering;

/// <summary>KV cache element type (llama.cpp <c>type_k</c>/<c>type_v</c>).</summary>
public enum KvCacheType { F16, Q8_0, Q4_0, F32 }

/// <summary>Context/runtime settings that change memory needs.</summary>
public sealed record RuntimeSettings
{
    public int ContextSize { get; init; } = 4096;
    public int ParallelSequences { get; init; } = 1;
    public KvCacheType KvType { get; init; } = KvCacheType.F16;
    public bool KvOnGpu { get; init; } = true;
    public bool FlashAttention { get; init; } = true;
    public int UBatch { get; init; } = 512;

    /// <summary>Multiplier learned from real loads (measured / estimated compute buffer).</summary>
    public double ComputeBufferCalibration { get; init; } = 1.0;
}

/// <summary>How much of each tier one model may use.</summary>
public sealed record TierBudget
{
    /// <summary>Total VRAM this model may occupy (weights + KV + compute buffers).</summary>
    public long VramBytes { get; init; }

    /// <summary>RAM this model may pin as its warm tier (plus CPU-side KV/compute).</summary>
    public long RamBytes { get; init; }

    /// <summary>Whether weights may stay on disk (memory-mapped, paged in on demand).</summary>
    public bool AllowDisk { get; init; } = true;

    /// <summary>VRAM consumed outside the model's own buffers (CUDA context, driver).</summary>
    public long VramOverheadBytes { get; init; } = 450L << 20;
}

/// <summary>Measured (or default) machine capabilities used by the estimators.</summary>
public sealed record HardwareSpec
{
    public bool HasGpuBackend { get; init; }
    /// <summary>A GPU backend is loaded but the user chose CPU mode.</summary>
    public bool GpuDisabledByUser { get; init; }
    public string? GpuName { get; init; }
    public long VramTotalBytes { get; init; }
    public long VramFreeBytes { get; init; }
    public double VramBandwidthGBs { get; init; } = 400;
    public double PcieBandwidthGBs { get; init; } = 16;

    public long RamTotalBytes { get; init; }
    public long RamAvailableBytes { get; init; }
    public double RamBandwidthGBs { get; init; } = 40;
    public int CpuCores { get; init; } = Environment.ProcessorCount;

    public double DiskSequentialMBs { get; init; } = 1500;
    public double DiskRandomMBs { get; init; } = 600;
    public double DiskLatencyMs { get; init; } = 0.1;
    public bool DiskLooksRotational => DiskLatencyMs >= 2 || DiskSequentialMBs < 300;

    public bool IsMeasured { get; init; }

    /// <summary>Patched llama.cpp with per-expert VRAM slots (Phase 2) is loaded.</summary>
    public bool HasExpertCache { get; init; }

    public static HardwareSpec Fallback => new()
    {
        HasGpuBackend = false,
        RamTotalBytes = 16L << 30,
        RamAvailableBytes = 12L << 30,
    };
}

/// <summary>Byte estimates for the non-weight allocations llama.cpp makes.</summary>
public static class MemoryEstimator
{
    public static double KvBytesPerElement(KvCacheType t) => t switch
    {
        KvCacheType.F32 => 4,
        KvCacheType.F16 => 2,
        KvCacheType.Q8_0 => 34.0 / 32,
        KvCacheType.Q4_0 => 18.0 / 32,
        _ => 2,
    };

    /// <summary>KV cache for a full context window across all sequences.</summary>
    public static long KvCacheBytes(ModelShape s, RuntimeSettings r)
    {
        var tokens = (long)Math.Max(1, r.ContextSize) * Math.Max(1, r.ParallelSequences);
        var b = KvBytesPerElement(r.KvType);
        if (s.KvLoraRank > 0)
        {
            // MLA caches the compressed latent + rope part once per layer (no separate V).
            return (long)(tokens * s.LayerCount * (double)(s.KvLoraRank + s.RopeDim) * b);
        }
        return (long)(tokens * (double)s.KvHeadsTotal * (s.HeadDimK + s.HeadDimV) * b);
    }

    /// <summary>
    /// Scratch buffer for one micro-batch: logits + activations, plus the
    /// attention score matrix when flash attention is off. Calibrated by
    /// <see cref="RuntimeSettings.ComputeBufferCalibration"/>.
    /// </summary>
    public static long ComputeBufferBytes(ModelShape s, RuntimeSettings r)
    {
        double ub = Math.Max(1, Math.Min(r.UBatch, r.ContextSize));
        var ff = Math.Max(s.FeedForwardLength, s.ExpertFeedForwardLength * Math.Max(1, s.ExpertUsedCount));
        double bytes = 4 * ub * (s.VocabSize + 4.0 * s.EmbeddingLength + 3.0 * ff);
        if (!r.FlashAttention)
            bytes += 4 * ub * (double)r.ContextSize * Math.Max(1, s.HeadCount);
        bytes = Math.Max(bytes, 128d * (1 << 20));
        return (long)(bytes * Math.Max(0.25, r.ComputeBufferCalibration));
    }

    /// <summary>Bytes of KV cache read per generated token (assumes the context is about half full on average).</summary>
    public static long KvReadPerToken(ModelShape s, RuntimeSettings r) =>
        KvCacheBytes(s, r with { ParallelSequences = 1 }) / 2;
}
