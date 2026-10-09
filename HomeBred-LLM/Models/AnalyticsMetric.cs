using System.ComponentModel.DataAnnotations;

namespace HomebredLLM.Models;

public class AnalyticsMetric
{
    [Key] public Guid Id { get; set; } = Guid.NewGuid();
    public Guid ModelId { get; set; }
    public LocalModel? Model { get; set; }
    public DateTime RecordedAt { get; set; } = DateTime.UtcNow;

    // GPU
    public float? GpuUtilizationPct { get; set; }
    public float? GpuMemoryUsedMb { get; set; }
    public float? GpuMemoryTotalMb { get; set; }
    public float? GpuTemperatureC { get; set; }

    // CPU / RAM
    public float? CpuUtilizationPct { get; set; }
    public float? RamUsedMb { get; set; }
    public float? RamTotalMb { get; set; }

    // Inference
    public float? TokensPerSecond { get; set; }
    public float? TimeToFirstTokenMs { get; set; }
    public float? TotalInferenceTimeMs { get; set; }
    public int? PromptTokens { get; set; }
    public int? OutputTokens { get; set; }
    public int? ActiveRequests { get; set; }

    // Prefill / decode split (from TTFT): prefill = prompt tokens / TTFT,
    // decode = (output tokens - 1) / (total - TTFT).
    public float? PrefillTokensPerSecond { get; set; }
    public float? DecodeTokensPerSecond { get; set; }

    // Memory tiers (planned placement for this run)
    public float? TierVramMb { get; set; }
    public float? TierWarmMb { get; set; }
    public float? TierColdMb { get; set; }

    /// <summary>Warm-tier bytes actually resident in RAM (Linux mincore; null where not measurable).</summary>
    public float? WarmResidentMb { get; set; }

    /// <summary>Expected share of CPU-side expert reads served from RAM, under the current routing profile.</summary>
    public float? WarmHitRate { get; set; }

    public float? EstimatedTokensPerSecond { get; set; }
    public float? EstGpuMsPerToken { get; set; }
    public float? EstCpuMsPerToken { get; set; }
    public float? EstDiskMsPerToken { get; set; }
    public float? EstSyncMsPerToken { get; set; }

    // I/O
    public float? DiskReadMbps { get; set; }
    public float? MajorFaultsPerSec { get; set; }
    public float? PcieRxMbps { get; set; }
    public float? PcieTxMbps { get; set; }

    // Phase 2 expert cache (per-expert VRAM slots)
    /// <summary>Expected share of routed picks served from VRAM slots.</summary>
    public float? ExpertCacheHitRate { get; set; }
    /// <summary>Experts uploaded into VRAM slots during the sample interval.</summary>
    public float? ExpertPromotions { get; set; }
    public float? ExpertUploadMb { get; set; }
}
