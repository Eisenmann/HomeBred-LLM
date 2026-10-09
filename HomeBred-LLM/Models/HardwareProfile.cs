using System.ComponentModel.DataAnnotations;
using HomebredLLM.Services.Tiering;

namespace HomebredLLM.Models;

/// <summary>Measured capabilities of this machine (single row) plus learned calibration factors.</summary>
public class HardwareProfile
{
    public static readonly Guid SingletonId = new("6b1f0d2e-6c47-4f0e-9b6a-2a8f0c5d7e11");

    [Key] public Guid Id { get; set; } = SingletonId;

    public bool HasGpuBackend { get; set; }
    public string? GpuName { get; set; }
    public string? BackendDevices { get; set; }
    public long VramTotalBytes { get; set; }
    public double VramBandwidthGBs { get; set; }
    public double PcieBandwidthGBs { get; set; }

    public long RamTotalBytes { get; set; }
    public double RamBandwidthGBs { get; set; }
    public int CpuCores { get; set; }

    public string? DiskPath { get; set; }
    public double DiskSequentialMBs { get; set; }
    public double DiskRandomMBs { get; set; }
    public double DiskLatencyMs { get; set; }

    /// <summary>Measured / estimated decode speed, averaged over recent runs (1 = estimator is right).</summary>
    public double SpeedCalibration { get; set; } = 1.0;

    /// <summary>Measured / estimated compute buffer size.</summary>
    public double ComputeBufferCalibration { get; set; } = 1.0;

    public DateTime MeasuredAt { get; set; } = DateTime.UtcNow;

    public HardwareSpec ToSpec(long vramFree, long ramAvailable) => new()
    {
        HasGpuBackend = HasGpuBackend,
        GpuName = GpuName,
        VramTotalBytes = VramTotalBytes,
        VramFreeBytes = vramFree,
        VramBandwidthGBs = VramBandwidthGBs > 0 ? VramBandwidthGBs : 400,
        PcieBandwidthGBs = PcieBandwidthGBs > 0 ? PcieBandwidthGBs : 16,
        RamTotalBytes = RamTotalBytes,
        RamAvailableBytes = ramAvailable,
        RamBandwidthGBs = RamBandwidthGBs > 0 ? RamBandwidthGBs : 40,
        CpuCores = CpuCores > 0 ? CpuCores : Environment.ProcessorCount,
        DiskSequentialMBs = DiskSequentialMBs > 0 ? DiskSequentialMBs : 1500,
        DiskRandomMBs = DiskRandomMBs > 0 ? DiskRandomMBs : 600,
        DiskLatencyMs = DiskLatencyMs > 0 ? DiskLatencyMs : 0.1,
        IsMeasured = true,
    };
}
