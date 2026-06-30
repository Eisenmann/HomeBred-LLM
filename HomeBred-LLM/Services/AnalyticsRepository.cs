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
        var q = db.AnalyticsMetrics
            .Where(m => m.ModelId == modelId && m.RecordedAt >= from && m.RecordedAt <= to);

        return new AnalyticsSummary(
            AvgGpuUtilPct:       await q.AverageAsync(m => (double?)m.GpuUtilizationPct),
            MaxGpuUtilPct:       await q.MaxAsync(m => (double?)m.GpuUtilizationPct),
            AvgGpuMemUsedMb:     await q.AverageAsync(m => (double?)m.GpuMemoryUsedMb),
            PeakGpuMemUsedMb:    await q.MaxAsync(m => (double?)m.GpuMemoryUsedMb),
            AvgTokensPerSecond:  await q.AverageAsync(m => (double?)m.TokensPerSecond),
            MaxTokensPerSecond:  await q.MaxAsync(m => (double?)m.TokensPerSecond),
            AvgTtftMs:           await q.AverageAsync(m => (double?)m.TimeToFirstTokenMs),
            TotalPromptTokens:   await q.SumAsync(m => (long?)m.PromptTokens) ?? 0,
            TotalOutputTokens:   await q.SumAsync(m => (long?)m.OutputTokens) ?? 0,
            AvgCpuUtilPct:       await q.AverageAsync(m => (double?)m.CpuUtilizationPct),
            TotalPoints:         await q.CountAsync()
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
