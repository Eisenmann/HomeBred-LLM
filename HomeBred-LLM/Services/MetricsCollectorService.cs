using HomebredLLM.Data;
using HomebredLLM.Models;
using HomebredLLM.Services.Tiering;
using Microsoft.EntityFrameworkCore;
using System.Threading;

namespace HomebredLLM.Services;

/// <summary>
/// Samples GPU/CPU/RAM every N seconds for all running models and persists to SQLite.
/// </summary>
public sealed class MetricsCollectorService(
    GpuMetricsService gpu,
    AnalyticsRepository repo,
    IDbContextFactory<AppDbContext> dbFactory,
    TieringCoordinator tiering,
    ProcessIoSampler io)
{
    private System.Timers.Timer? _timer;
    private int _isCollecting;
    public int IntervalSeconds { get; set; } = 5;

    // Externally set to record the latest inference stats into the next metric row
    public InferenceStats? LastInferenceStats { get; set; }

    public void Start()
    {
        _timer = new System.Timers.Timer(IntervalSeconds * 1000);
        _timer.Elapsed += async (_, _) => await CollectAsync();
        _timer.AutoReset = true;
        _timer.Start();
    }

    public void Stop() => _timer?.Stop();

    private async Task CollectAsync()
    {
        if (Interlocked.Exchange(ref _isCollecting, 1) == 1) return;
        try
        {
            await using var db = await dbFactory.CreateDbContextAsync();
            var runningModels = await db.Models
                .Where(m => m.Status == ModelStatus.Running)
                .Select(m => m.Id)
                .ToListAsync();

            if (runningModels.Count == 0) return;

            var snap = gpu.Sample();
            // Each chat turn's stats are recorded once (not repeated in every sample).
            var stats = LastInferenceStats;
            LastInferenceStats = null;
            var now = DateTime.UtcNow;
            var pcie = gpu.SamplePcie();
            var (diskMbps, faults) = io.Sample();

            float? prefill = null, decode = null;
            if (stats is { TimeToFirstTokenMs: > 0 })
            {
                prefill = stats.PromptTokens / (stats.TimeToFirstTokenMs / 1000f);
                var decodeMs = stats.TotalMs - stats.TimeToFirstTokenMs;
                if (stats.OutputTokens > 1 && decodeMs > 0)
                    decode = (stats.OutputTokens - 1) / (decodeMs / 1000f);
            }

            foreach (var modelId in runningModels)
            {
                var t = tiering.GetSnapshot(modelId);
                const float Mb = 1 << 20;
                await repo.SaveAsync(new AnalyticsMetric
                {
                    PrefillTokensPerSecond = prefill,
                    DecodeTokensPerSecond = decode,
                    TierVramMb = t is null ? null : t.Plan.VramTotalBytes / Mb,
                    TierWarmMb = t is null ? null : t.Plan.WarmBytes / Mb,
                    TierColdMb = t is null ? null : t.Plan.ColdBytes / Mb,
                    WarmResidentMb = t?.WarmResidentBytes is { } res ? res / Mb : null,
                    WarmHitRate = t is null ? null : (float)t.WarmHitRate,
                    EstimatedTokensPerSecond = t is null ? null : (float)t.CalibratedEstimate.TokensPerSecond,
                    EstGpuMsPerToken = t is null ? null : (float)t.CalibratedEstimate.GpuMs,
                    EstCpuMsPerToken = t is null ? null : (float)t.CalibratedEstimate.CpuMs,
                    EstDiskMsPerToken = t is null ? null : (float)t.CalibratedEstimate.DiskMs,
                    EstSyncMsPerToken = t is null ? null : (float)t.CalibratedEstimate.SyncMs,
                    DiskReadMbps = diskMbps,
                    MajorFaultsPerSec = faults,
                    PcieRxMbps = pcie?.RxMbps,
                    PcieTxMbps = pcie?.TxMbps,
                    ExpertCacheHitRate = t?.ExpertCache is { } ec ? (float)ec.ExpectedHitRate : null,
                    ExpertPromotions = t?.ExpertCache is { } ec2 ? ec2.PromotionsSinceLastSample : null,
                    ExpertUploadMb = t?.ExpertCache is { } ec3 ? ec3.UploadBytesSinceLastSample / Mb : null,
                    ModelId = modelId,
                    RecordedAt = now,
                    GpuUtilizationPct = snap.GpuUtilPct,
                    GpuMemoryUsedMb = snap.GpuMemUsedMb,
                    GpuMemoryTotalMb = snap.GpuMemTotalMb,
                    GpuTemperatureC = snap.GpuTempC,
                    CpuUtilizationPct = snap.CpuUtilPct,
                    RamUsedMb = snap.RamUsedMb,
                    RamTotalMb = snap.RamTotalMb,
                    TokensPerSecond = stats?.TokensPerSecond,
                    TimeToFirstTokenMs = stats?.TimeToFirstTokenMs,
                    TotalInferenceTimeMs = stats?.TotalMs,
                    PromptTokens = stats?.PromptTokens,
                    OutputTokens = stats?.OutputTokens,
                });
                tiering.MarkSampled(modelId);
            }
        }
        catch { /* Don't crash the collector on transient errors */ }
        finally
        {
            Interlocked.Exchange(ref _isCollecting, 0);
        }
    }
}
