using System.ComponentModel.DataAnnotations;

namespace HomebredLLM.Models;

public enum DownloadStatus { Pending, Downloading, Completed, Failed, Cancelled }

public class DownloadJob
{
    [Key] public Guid Id { get; set; } = Guid.NewGuid();
    public Guid ModelId { get; set; }
    public LocalModel? Model { get; set; }
    public string HfRepoId { get; set; } = "";
    public string HfFilename { get; set; } = "";
    public DownloadStatus Status { get; set; } = DownloadStatus.Pending;
    public double ProgressPct { get; set; }
    public long? TotalBytes { get; set; }
    public long? DownloadedBytes { get; set; }
    public string? ErrorMessage { get; set; }
    public DateTime? StartedAt { get; set; }
    public DateTime? CompletedAt { get; set; }
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
}
