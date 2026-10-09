namespace HomebredLLM.Services.Tiering;

/// <summary>
/// Turns a placement plan into a per-token decode time. Generation is
/// memory-bandwidth bound, so each tier contributes bytes-read ÷ effective
/// bandwidth; disk adds a latency per cold slice read; each GPU↔CPU graph
/// split adds a synchronisation cost. Deliberately simple and explainable —
/// the analytics page compares it with measured tok/s, and the ratio is fed
/// back as a calibration factor.
/// </summary>
public static class PerformanceEstimator
{
    /// <summary>Fraction of peak VRAM bandwidth quantized matmul-vec kernels reach.</summary>
    public const double GpuEfficiency = 0.75;

    /// <summary>Fraction of measured RAM bandwidth llama.cpp's CPU kernels reach.</summary>
    public const double CpuEfficiency = 0.6;

    public const double SyncMsPerSplit = 0.04;

    public static PerTokenEstimate Estimate(PlacementPlan plan, HardwareSpec hw, double calibration = 1.0)
    {
        var gpuMs = plan.UsesGpu || plan.KvOnGpu
            ? plan.GpuBytesPerToken / (Math.Max(1, hw.VramBandwidthGBs) * 1e9 * GpuEfficiency) * 1000
            : 0;
        var cpuMs = plan.CpuBytesPerToken / (Math.Max(0.5, hw.RamBandwidthGBs) * 1e9 * CpuEfficiency) * 1000;

        var diskMs = 0.0;
        if (plan.ColdBytesPerToken > 0)
        {
            var avgRead = plan.ColdReadsPerToken > 0 ? plan.ColdBytesPerToken / plan.ColdReadsPerToken : plan.ColdBytesPerToken;
            // Large slices stream near sequential speed; small ones are dominated by random-read speed.
            var mbs = avgRead >= 4 << 20 ? hw.DiskSequentialMBs : hw.DiskRandomMBs;
            diskMs = plan.ColdBytesPerToken / (Math.Max(1, mbs) * 1e6) * 1000
                     + plan.ColdReadsPerToken * hw.DiskLatencyMs;
        }

        var syncMs = plan.GraphSplits * SyncMsPerSplit;
        calibration = calibration is > 0.05 and < 20 ? calibration : 1.0;
        return new PerTokenEstimate(gpuMs * calibration, cpuMs * calibration, diskMs, syncMs);
    }
}
