using System.ComponentModel.DataAnnotations;

namespace HomebredLLM.Models;

/// <summary>
/// A pre-trained GGUF LoRA adapter attached to a base model. Adapters are
/// <b>applied</b> at load time (LLamaSharp has no training API) — training the
/// adapter itself is done outside the app. Multiple adapters can be stacked on
/// one model, each with its own blend scale.
/// </summary>
public class LoraAdapterConfig
{
    [Key] public Guid Id { get; set; } = Guid.NewGuid();
    public Guid ModelId { get; set; }
    public LocalModel? Model { get; set; }

    public string Name { get; set; } = "";

    /// <summary>Absolute path to the GGUF LoRA file inside the app's adapters directory.</summary>
    public string FilePath { get; set; } = "";

    /// <summary>Blend strength. 1.0 = full effect, 0 = disabled, &gt;1 over-applies.</summary>
    public float Scale { get; set; } = 1.0f;

    /// <summary>When false the adapter is kept on record but skipped at load time.</summary>
    public bool Enabled { get; set; } = true;

    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    public string DisplayScale => $"{Scale:F2}";
}
