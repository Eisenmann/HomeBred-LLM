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
}
