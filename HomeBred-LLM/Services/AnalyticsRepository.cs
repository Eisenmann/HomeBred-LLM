using HomebredLLM.Data;
using HomebredLLM.Models;
using Microsoft.EntityFrameworkCore;

namespace HomebredLLM.Services;

public record AnalyticsSummary(
    double? AvgGpuUtilPct,
    double? MaxGpuUtilPct,
    double? AvgGpuMemUsedMb,
    double? PeakGpuMemUsedMb,
    double? AvgTokensPerSecond,
    double? MaxTokensPerSecond,
    double? AvgTtftMs,
    long TotalPromptTokens,
    long TotalOutputTokens,
    double? AvgCpuUtilPct,
    int TotalPoints);

public sealed class AnalyticsRepository(IDbContextFactory<AppDbContext> dbFactory)
{
    public async Task SaveAsync(AnalyticsMetric metric)
    {
        await using var db = await dbFactory.CreateDbContextAsync();
        db.AnalyticsMetrics.Add(metric);
        await db.SaveChangesAsync();
    }

    public async Task<List<AnalyticsMetric>> QueryAsync(
        Guid modelId, DateTime from, DateTime to)
    {
        await using var db = await dbFactory.CreateDbContextAsync();
        return await db.AnalyticsMetrics
            .Where(m => m.ModelId == modelId && m.RecordedAt >= from && m.RecordedAt <= to)
            .OrderBy(m => m.RecordedAt)
            .ToListAsync();
    }

    public async Task<AnalyticsSummary> GetSummaryAsync(
        Guid modelId, DateTime from, DateTime to)
    {
        await using var db = await dbFactory.CreateDbContextAsync();
        var result = await db.AnalyticsMetrics
            .Where(m => m.ModelId == modelId && m.RecordedAt >= from && m.RecordedAt <= to)
            .GroupBy(_ => 1)
            .Select(g => new
            {
                AvgGpuUtilPct      = g.Average(m => (double?)m.GpuUtilizationPct),
                MaxGpuUtilPct      = g.Max(m => (double?)m.GpuUtilizationPct),
                AvgGpuMemUsedMb    = g.Average(m => (double?)m.GpuMemoryUsedMb),
                PeakGpuMemUsedMb   = g.Max(m => (double?)m.GpuMemoryUsedMb),
                AvgTokensPerSecond = g.Average(m => (double?)m.TokensPerSecond),
                MaxTokensPerSecond = g.Max(m => (double?)m.TokensPerSecond),
                AvgTtftMs          = g.Average(m => (double?)m.TimeToFirstTokenMs),
                TotalPromptTokens  = g.Sum(m => (long?)m.PromptTokens),
                TotalOutputTokens  = g.Sum(m => (long?)m.OutputTokens),
                AvgCpuUtilPct      = g.Average(m => (double?)m.CpuUtilizationPct),
                TotalPoints        = g.Count(),
            })
            .FirstOrDefaultAsync();

        return new AnalyticsSummary(
            AvgGpuUtilPct:       result?.AvgGpuUtilPct,
            MaxGpuUtilPct:       result?.MaxGpuUtilPct,
            AvgGpuMemUsedMb:     result?.AvgGpuMemUsedMb,
            PeakGpuMemUsedMb:    result?.PeakGpuMemUsedMb,
            AvgTokensPerSecond:  result?.AvgTokensPerSecond,
            MaxTokensPerSecond:  result?.MaxTokensPerSecond,
            AvgTtftMs:           result?.AvgTtftMs,
            TotalPromptTokens:   result?.TotalPromptTokens ?? 0,
            TotalOutputTokens:   result?.TotalOutputTokens ?? 0,
            AvgCpuUtilPct:       result?.AvgCpuUtilPct,
            TotalPoints:         result?.TotalPoints ?? 0
        );
    }

    public async Task<int> DeleteAsync(Guid modelId, DateTime from, DateTime to)
    {
        await using var db = await dbFactory.CreateDbContextAsync();
        return await db.AnalyticsMetrics
            .Where(m => m.ModelId == modelId && m.RecordedAt >= from && m.RecordedAt <= to)
            .ExecuteDeleteAsync();
    }
}
