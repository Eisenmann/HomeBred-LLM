using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace HomebredLLM.Models;

public enum ModelStatus { Pending, Downloading, Ready, Running, Error }

/// <summary>Which inference engine a model's weights are loaded by.</summary>
public enum ModelFormat { Onnx, Gguf }

public class LocalModel
{
    [Key] public Guid Id { get; set; } = Guid.NewGuid();
    public string Name { get; set; } = "";
    public string? HfRepoId { get; set; }
    public string? HfFilename { get; set; }
    public string? LocalPath { get; set; }
    public ModelFormat Format { get; set; } = ModelFormat.Onnx;
    public string? MmprojPath { get; set; }
    public long? FileSizeBytes { get; set; }
    public string? Quantization { get; set; }
    public string? Architecture { get; set; }
    public long? ParameterCount { get; set; }
    public int? ContextLength { get; set; }

    /// <summary>
    /// Import-time diagnostics for files that will not load (or may not load)
    /// cleanly but are still importable. Semicolon-separated human-readable
    /// warnings, e.g. "Tokenizer/embedding vocab size mismatch (151643 vs 151936)".
    /// The Model Library UI renders a warning badge whenever this is non-empty.
    /// </summary>
    public string? Warnings { get; set; }

    public ModelStatus Status { get; set; } = ModelStatus.Pending;
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;

    // Navigation
    public ModelConfiguration? Config { get; set; }
    public List<AnalyticsMetric> Metrics { get; set; } = [];
    public List<ChatSession> ChatSessions { get; set; } = [];
    public List<DownloadJob> DownloadJobs { get; set; } = [];
    public List<LoraAdapterConfig> LoraAdapters { get; set; } = [];
    public MemoryProfile? MemoryProfile { get; set; }
    public List<ExpertUsageSnapshot> ExpertUsageSnapshots { get; set; } = [];

    // Hardware-fit badge, computed by the Model Library (not persisted).
    [NotMapped] public string? FitLabel { get; set; }
    [NotMapped] public string FitColor { get; set; } = "#6B7280";
    [NotMapped] public string? FitDetail { get; set; }
    [NotMapped] public bool HasFit => !string.IsNullOrEmpty(FitLabel);

    // Computed for Avalonia IsVisible bindings (replaces WPF DataTrigger)
    public bool HasWarning => !string.IsNullOrWhiteSpace(Warnings);
    public bool IsRunning    => Status == ModelStatus.Running;
    public bool IsNotRunning => Status != ModelStatus.Running;
    public bool IsStartable  => Status is ModelStatus.Ready or ModelStatus.Error;
    public bool IsMultimodal => MmprojPath is not null;

    public string DisplaySize => FileSizeBytes.HasValue
        ? FileSizeBytes.Value >= 1_000_000_000
            ? $"{FileSizeBytes.Value / 1_000_000_000.0:F1} GB"
            : $"{FileSizeBytes.Value / 1_000_000.0:F0} MB"
        : "—";
}