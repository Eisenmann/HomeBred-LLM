using System.ComponentModel.DataAnnotations;

namespace HomebredLLM.Models;

public class ModelConfiguration
{
    [Key] public Guid Id { get; set; } = Guid.NewGuid();
    public Guid ModelId { get; set; }
    public LocalModel? Model { get; set; }

    // Inference params
    public int ContextSize { get; set; } = 4096;
    public float Temperature { get; set; } = 0.7f;
    public float TopP { get; set; } = 0.9f;
    public int TopK { get; set; } = 40;
    public float RepeatPenalty { get; set; } = 1.1f;
    public int GpuLayerCount { get; set; } = -1;         // -1 = all on GPU
    public int ThreadCount { get; set; } = 4;
    public int BatchSize { get; set; } = 512;
    public int MaxTokens { get; set; } = 2048;
    public string SystemPrompt { get; set; } = "";
    public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
}
