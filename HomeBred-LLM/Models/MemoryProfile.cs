using System.ComponentModel.DataAnnotations;
using HomebredLLM.Services.Tiering;

namespace HomebredLLM.Models;

/// <summary>How a GGUF model's weights are placed when it is started.</summary>
public enum TieringMode
{
    /// <summary>Legacy behaviour: ModelConfiguration.GpuLayerCount decides, nothing else.</summary>
    Simple,
    /// <summary>Budget-driven VRAM / RAM / disk placement (see docs/tiered-memory-architecture.md).</summary>
    Tiered,
}

public enum RebalancePolicy
{
    Off,
    /// <summary>Recompute the warm tier and show a hint when it would improve the hit rate.</summary>
    Suggest,
    /// <summary>Re-target the warm tier automatically while the model is idle.</summary>
    Auto,
}

/// <summary>Per-model tier budgets and runtime-memory settings (1:1 with LocalModel).</summary>
public class MemoryProfile
{
    [Key] public Guid Id { get; set; } = Guid.NewGuid();
    public Guid ModelId { get; set; }
    public LocalModel? Model { get; set; }

    public TieringMode Mode { get; set; } = TieringMode.Tiered;

    /// <summary>VRAM this model may use, in MB. -1 = automatic (free VRAM minus a safety margin).</summary>
    public int VramBudgetMb { get; set; } = -1;

    /// <summary>RAM for the warm tier + CPU-side buffers, in MB. -1 = automatic (available RAM minus OS reserve).</summary>
    public int RamBudgetMb { get; set; } = -1;

    public bool AllowDiskTier { get; set; } = true;

    /// <summary>Pin warm-tier pages (mlock / VirtualLock) instead of only prefetching them.</summary>
    public bool LockWarmTier { get; set; }

    public KvCacheType KvCacheType { get; set; } = KvCacheType.F16;
    public bool KvOnGpu { get; set; } = true;
    public bool FlashAttention { get; set; } = true;
    public int ParallelSequences { get; set; } = 1;

    /// <summary>Learn the expert hot set from real routing (MoE only).</summary>
    public bool RoutingProfilerEnabled { get; set; } = true;

    /// <summary>Tokens replayed per profiling pass (side-context window).</summary>
    public int ProfilerWindowTokens { get; set; } = 2048;

    public RebalancePolicy RebalancePolicy { get; set; } = RebalancePolicy.Auto;

    /// <summary>Minimum expected warm-tier hit-rate gain (0..1) before the rebalancer acts.</summary>
    public float RebalanceThreshold { get; set; } = 0.03f;

    public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
}
