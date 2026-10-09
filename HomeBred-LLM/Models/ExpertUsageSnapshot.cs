using System.ComponentModel.DataAnnotations;

namespace HomebredLLM.Models;

/// <summary>A saved MoE routing histogram (see ExpertUsageProfile.Serialize).</summary>
public class ExpertUsageSnapshot
{
    [Key] public Guid Id { get; set; } = Guid.NewGuid();
    public Guid ModelId { get; set; }
    public LocalModel? Model { get; set; }
    public DateTime RecordedAt { get; set; } = DateTime.UtcNow;
    public int Layers { get; set; }
    public int Experts { get; set; }
    public double ObservedTokens { get; set; }

    /// <summary>Share of routing that goes to the top 20% of experts (skew).</summary>
    public double Concentration { get; set; }

    public byte[] Data { get; set; } = [];
}
