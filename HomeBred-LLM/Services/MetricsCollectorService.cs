using HomebredLLM.Data;
using HomebredLLM.Models;
using Microsoft.EntityFrameworkCore;
using System.Threading;

namespace HomebredLLM.Services;

/// <summary>
/// Samples GPU/CPU/RAM every N seconds for all running models and persists to SQLite.
/// </summary>
public sealed class MetricsCollectorService(
    GpuMetricsService gpu,
    AnalyticsRepository repo,
    IDbContextFactory<AppDbContext> dbFactory)
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
            var stats = LastInferenceStats;
            var now = DateTime.UtcNow;

            foreach (var modelId in runningModels)
            {
                await repo.SaveAsync(new AnalyticsMetric
                {
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
            }
        }
        catch { /* Don't crash the collector on transient errors */ }
        finally
        {
            Interlocked.Exchange(ref _isCollecting, 0);
        }
    }
}
