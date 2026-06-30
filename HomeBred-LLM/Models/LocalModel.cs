using System.ComponentModel.DataAnnotations;

namespace HomebredLLM.Models;

public enum ModelStatus { Pending, Downloading, Ready, Running, Error }

public class LocalModel
{
    [Key] public Guid Id { get; set; } = Guid.NewGuid();
    public string Name { get; set; } = "";
    public string? HfRepoId { get; set; }
    public string? HfFilename { get; set; }
    public string? LocalPath { get; set; }
    public long? FileSizeBytes { get; set; }
    public string? Quantization { get; set; }
    public string? Architecture { get; set; }
    public long? ParameterCount { get; set; }
    public int? ContextLength { get; set; }
    public ModelStatus Status { get; set; } = ModelStatus.Pending;
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;

    // Navigation
    public ModelConfiguration? Config { get; set; }
    public List<AnalyticsMetric> Metrics { get; set; } = [];
    public List<ChatSession> ChatSessions { get; set; } = [];
    public List<DownloadJob> DownloadJobs { get; set; } = [];

    // Computed for Avalonia IsVisible bindings (replaces WPF DataTrigger)
    public bool IsRunning    => Status == ModelStatus.Running;
    public bool IsNotRunning => Status != ModelStatus.Running;
    public bool IsStartable  => Status is ModelStatus.Ready or ModelStatus.Error;

    public string DisplaySize => FileSizeBytes.HasValue
        ? FileSizeBytes.Value >= 1_000_000_000
            ? $"{FileSizeBytes.Value / 1_000_000_000.0:F1} GB"
            : $"{FileSizeBytes.Value / 1_000_000.0:F0} MB"
        : "—";
}
