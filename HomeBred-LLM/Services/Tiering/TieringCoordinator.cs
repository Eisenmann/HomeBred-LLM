using System.Collections.Concurrent;
using HomebredLLM.Data;
using HomebredLLM.Models;
using LLama;
using LLama.Abstractions;
using Microsoft.EntityFrameworkCore;

namespace HomebredLLM.Services.Tiering;

/// <summary>Everything decided before llama.cpp is asked to load a tiered model.</summary>
public sealed record PreparedLoad(
    PlacementPlan Plan,
    MemoryProfile Settings,
    ExpertUsageProfile? Usage,
    HardwareSpec Hardware,
    double SpeedCalibration);

/// <summary>Live tiering state of a running model, for metrics and the UI.</summary>
public sealed record TieringSnapshot(
    PlacementPlan Plan,
    PerTokenEstimate CalibratedEstimate,
    WarmTierStatus? Warm,
    long? WarmResidentBytes,
    double WarmHitRate,
    string? Suggestion,
    string ProfilerStatus,
    long TokensProfiled,
    double? RoutingConcentration);

/// <summary>
/// Owns the tiered-memory lifecycle of every loaded GGUF model: plans the
/// load, runs the warm tier and the routing profiler, rebalances the warm
/// tier as routing statistics change, persists profiles and exposes the
/// numbers the analytics collector records.
/// </summary>
public sealed class TieringCoordinator : IDisposable
{
    private sealed class Runtime
    {
        public required Guid ModelId { get; init; }
        public required PreparedLoad Prepared { get; init; }
        public required PlacementPlan Plan { get; set; }
        public WarmTierManager? Warm { get; set; }
        public ExpertRoutingProfiler? Profiler { get; set; }
        public string ProfilerStatus { get; set; } = "Off";
        public Func<Func<Task>, Task>? Exclusive { get; set; }
        public string? PendingText { get; set; }
        public DateTime LastActivity { get; set; } = DateTime.UtcNow;
        public DateTime LastPersist { get; set; } = DateTime.UtcNow;
        public DateTime UsageSeenAt { get; set; }
        public string? Suggestion { get; set; }
        public long? ResidentBytes { get; set; }
        public DateTime ResidentMeasuredAt { get; set; }
        public int Busy;
    }

    private const long AutoVramMargin = 768L << 20;
    private const long AutoRamReserve = 4L << 30;
    private static readonly TimeSpan IdleBeforeBackgroundWork = TimeSpan.FromSeconds(3);
    private static readonly TimeSpan PersistEvery = TimeSpan.FromMinutes(5);

    private readonly IDbContextFactory<AppDbContext> _dbFactory;
    private readonly HardwareProbe _probe;
    private readonly ConcurrentDictionary<Guid, Runtime> _runtimes = new();
    private readonly ConcurrentDictionary<string, (DateTime Stamp, TensorCatalog Catalog)> _catalogs = new();
    private readonly System.Threading.Timer _timer;

    public IExpertCacheBackend ExpertCache { get; } = new NativeExpertCacheBackend();

    /// <summary>Raised (on a background thread) when a model's plan, warm tier or suggestion changes.</summary>
    public event Action<Guid>? StateChanged;

    public TieringCoordinator(IDbContextFactory<AppDbContext> dbFactory, HardwareProbe probe)
    {
        _dbFactory = dbFactory;
        _probe = probe;
        _timer = new System.Threading.Timer(_ => _ = TickAsync(), null, TimeSpan.FromSeconds(15), TimeSpan.FromSeconds(15));
    }

    // ── Settings & planning ─────────────────────────────────────────────────

    public async Task<MemoryProfile> GetOrCreateMemoryProfileAsync(Guid modelId)
    {
        await using var db = await _dbFactory.CreateDbContextAsync();
        var p = await db.MemoryProfiles.AsNoTracking().FirstOrDefaultAsync(x => x.ModelId == modelId);
        if (p is not null) return p;
        p = new MemoryProfile { ModelId = modelId };
        db.MemoryProfiles.Add(p);
        await db.SaveChangesAsync();
        return p;
    }

    public async Task SaveMemoryProfileAsync(MemoryProfile profile)
    {
        await using var db = await _dbFactory.CreateDbContextAsync();
        var existing = await db.MemoryProfiles.FirstOrDefaultAsync(x => x.Id == profile.Id);
        profile.UpdatedAt = DateTime.UtcNow;
        if (existing is null) db.MemoryProfiles.Add(profile);
        else db.Entry(existing).CurrentValues.SetValues(profile);
        await db.SaveChangesAsync();
    }

    public async Task<TensorCatalog> GetCatalogAsync(string ggufPath, CancellationToken ct = default)
    {
        var stamp = File.GetLastWriteTimeUtc(ggufPath);
        if (_catalogs.TryGetValue(ggufPath, out var c) && c.Stamp == stamp) return c.Catalog;
        var catalog = await TensorCatalog.FromFileAsync(ggufPath, ct);
        _catalogs[ggufPath] = (stamp, catalog);
        return catalog;
    }

    public static RuntimeSettings SettingsFor(ModelConfiguration cfg, MemoryProfile p, double computeCalibration = 1) => new()
    {
        ContextSize = Math.Max(512, cfg.ContextSize),
        ParallelSequences = Math.Max(1, p.ParallelSequences),
        KvType = p.KvCacheType,
        KvOnGpu = p.KvOnGpu,
        FlashAttention = p.FlashAttention || p.KvCacheType is KvCacheType.Q8_0 or KvCacheType.Q4_0,
        UBatch = Math.Clamp(cfg.BatchSize, 32, 512),
        ComputeBufferCalibration = computeCalibration,
    };

    /// <summary>
    /// Turns the stored budgets (MB, -1 = auto) into bytes for this machine,
    /// reserving room for the routing profiler's side context when it will run.
    /// </summary>
    public static TierBudget ResolveBudget(MemoryProfile p, HardwareSpec hw, ModelShape shape, RuntimeSettings settings)
    {
        long vram = p.VramBudgetMb >= 0
            ? (long)p.VramBudgetMb << 20
            : Math.Max(0, (hw.VramFreeBytes > 0 ? hw.VramFreeBytes : hw.VramTotalBytes) - AutoVramMargin);
        long ram = p.RamBudgetMb >= 0
            ? (long)p.RamBudgetMb << 20
            : Math.Max(1L << 30, hw.RamAvailableBytes - AutoRamReserve);

        if (shape.IsMoe && p.RoutingProfilerEnabled && ExpertRoutingProfiler.IsAvailable)
        {
            var side = settings with { UBatch = 64, ContextSize = p.ProfilerWindowTokens, ParallelSequences = 1 };
            vram -= MemoryEstimator.ComputeBufferBytes(shape, side);
            ram -= MemoryEstimator.KvCacheBytes(shape, side);
        }

        return new TierBudget
        {
            VramBytes = Math.Max(0, vram),
            RamBytes = Math.Max(0, ram),
            AllowDisk = p.AllowDiskTier,
        };
    }

    /// <summary>Plan for a model's current settings without loading it (Config preview, library badges).</summary>
    public async Task<FitReport?> PreviewAsync(LocalModel model, ModelConfiguration cfg, MemoryProfile p,
        CancellationToken ct = default)
    {
        if (model.Format != ModelFormat.Gguf || model.LocalPath is null || !File.Exists(model.LocalPath)) return null;
        var catalog = await GetCatalogAsync(model.LocalPath, ct);
        var hwp = await _probe.GetProfileAsync(ct);
        var hw = await _probe.GetCurrentSpecAsync(ct);
        if (_runtimes.TryGetValue(model.Id, out var rt))
            hw = hw with { VramFreeBytes = hw.VramFreeBytes + rt.Plan.VramTotalBytes }; // its own VRAM is "free" for it
        var settings = SettingsFor(cfg, p, hwp.ComputeBufferCalibration);
        var budget = ResolveBudget(p, hw, catalog.Shape, settings);
        var usage = _runtimes.TryGetValue(model.Id, out var r2) ? r2.Prepared.Usage : await LoadUsageAsync(model.Id, catalog.Shape);
        var report = CapacityCalculator.Assess(catalog, settings, budget, hw, usage, model.Quantization);
        var calibrated = PerformanceEstimator.Estimate(report.Plan, hw, hwp.SpeedCalibration);
        return report with { Plan = report.Plan.WithEstimate(calibrated) };
    }

    /// <summary>
    /// Plans a load. Returns null when the model should load the legacy way
    /// (Simple mode, non-GGUF, or a GPU backend whose VRAM size is unknown).
    /// </summary>
    public async Task<PreparedLoad?> PrepareAsync(Guid modelId, string modelPath, ModelConfiguration cfg,
        IProgress<string>? progress = null, CancellationToken ct = default)
    {
        var p = await GetOrCreateMemoryProfileAsync(modelId);
        if (p.Mode != TieringMode.Tiered) return null;

        progress?.Report("Planning memory tiers…");
        var catalog = await GetCatalogAsync(modelPath, ct);
        var hwp = await _probe.GetProfileAsync(ct);
        var hw = await _probe.GetCurrentSpecAsync(ct);

        if (hw.HasGpuBackend && hw.VramTotalBytes == 0 && p.VramBudgetMb < 0)
        {
            progress?.Report("GPU VRAM size unknown (no NVML) — using GPU layer count. Set an explicit VRAM budget to enable tiering.");
            return null;
        }

        var settings = SettingsFor(cfg, p, hwp.ComputeBufferCalibration);
        var budget = ResolveBudget(p, hw, catalog.Shape, settings);
        var usage = catalog.IsMoe ? await LoadUsageAsync(modelId, catalog.Shape) : null;
        var plan = PlacementPlanner.Plan(catalog, settings, budget, hw, usage);

        if (!plan.Fits)
            throw new InvalidOperationException(
                "The model does not fit the memory budgets: " + string.Join(" ", plan.Warnings) +
                " Raise the RAM budget or allow the disk tier (Config → Memory tiers).");

        progress?.Report(
            $"Tiers: VRAM {PlacementPlanner.Gb(plan.VramWeightBytes)} · RAM {PlacementPlanner.Gb(plan.WarmBytes)} · " +
            $"disk {PlacementPlanner.Gb(plan.ColdBytes)} · est. {PerformanceEstimator.Estimate(plan, hw, hwp.SpeedCalibration).TokensPerSecond:F1} tok/s");
        return new PreparedLoad(plan, p, usage, hw, hwp.SpeedCalibration);
    }

    private async Task<ExpertUsageProfile?> LoadUsageAsync(Guid modelId, ModelShape shape)
    {
        if (!shape.IsMoe) return null;
        await using var db = await _dbFactory.CreateDbContextAsync();
        var snap = await db.ExpertUsageSnapshots.AsNoTracking()
            .Where(s => s.ModelId == modelId && s.Layers == shape.LayerCount && s.Experts == shape.ExpertCount)
            .OrderByDescending(s => s.RecordedAt)
            .FirstOrDefaultAsync();
        return ExpertUsageProfile.Deserialize(snap?.Data) ?? new ExpertUsageProfile(shape.LayerCount, shape.ExpertCount);
    }

    // ── Lifecycle ───────────────────────────────────────────────────────────

    public void OnLoaded(Guid modelId, PreparedLoad prep, LLamaWeights weights, IContextParams contextParams,
        Func<Func<Task>, Task> exclusive)
    {
        var rt = new Runtime { ModelId = modelId, Prepared = prep, Plan = prep.Plan, Exclusive = exclusive };

        var path = prep.Plan.Catalog.FilePath;
        if (path is not null && prep.Plan.CpuWeightBytes > 0)
        {
            try
            {
                rt.Warm = new WarmTierManager(path);
                _ = rt.Warm.ApplyAsync(prep.Plan.WarmRanges, prep.Settings.LockWarmTier);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                rt.Suggestion = $"Warm tier unavailable: {ex.Message}";
            }
        }

        if (prep.Usage is not null && prep.Settings.RoutingProfilerEnabled)
        {
            if (!ExpertRoutingProfiler.IsAvailable)
                rt.ProfilerStatus = $"Unavailable: {ExpertRoutingProfiler.UnavailableReason}";
            else
            {
                try
                {
                    rt.Profiler = new ExpertRoutingProfiler(weights, contextParams, prep.Plan.Catalog.Shape,
                        prep.Usage, prep.Settings.ProfilerWindowTokens);
                    rt.ProfilerStatus = "Waiting for chat activity";
                }
                catch (Exception ex)
                {
                    rt.ProfilerStatus = $"Failed to start: {ex.Message}";
                }
            }
        }
        rt.UsageSeenAt = prep.Usage?.UpdatedAt ?? DateTime.MinValue;

        _runtimes[modelId] = rt;
        StateChanged?.Invoke(modelId);
    }

    public void OnUnloaded(Guid modelId)
    {
        if (!_runtimes.TryRemove(modelId, out var rt)) return;
        try { Task.Run(() => PersistUsageAsync(rt, force: true)).Wait(TimeSpan.FromSeconds(3)); }
        catch { /* losing one profile snapshot is acceptable */ }
        rt.Profiler?.Dispose();
        rt.Warm?.Dispose();
        StateChanged?.Invoke(modelId);
    }

    /// <summary>Called after each chat turn with the full text the model just processed.</summary>
    public void OnChatCompleted(Guid modelId, string processedText, InferenceStats stats)
    {
        if (!_runtimes.TryGetValue(modelId, out var rt)) return;
        rt.LastActivity = DateTime.UtcNow;
        if (rt.Profiler is not null) rt.PendingText = processedText;

        // Calibrate the speed model against measured decode speed.
        var decodeMs = stats.TotalMs - stats.TimeToFirstTokenMs;
        if (stats.OutputTokens >= 16 && decodeMs > 0)
        {
            var measured = (stats.OutputTokens - 1) / (decodeMs / 1000.0);
            var estimated = rt.Plan.Estimate.TokensPerSecond; // uncalibrated model
            if (estimated > 0) _ = _probe.RecordSpeedSampleAsync(measured / estimated);
        }
    }

    public void NotifyActivity(Guid modelId)
    {
        if (_runtimes.TryGetValue(modelId, out var rt)) rt.LastActivity = DateTime.UtcNow;
    }

    // ── Background: profiling, rebalancing, persistence ─────────────────────

    private async Task TickAsync()
    {
        foreach (var rt in _runtimes.Values)
        {
            if (Interlocked.Exchange(ref rt.Busy, 1) == 1) continue;
            try
            {
                await ProfileIfIdleAsync(rt);
                Rebalance(rt);
                await PersistUsageAsync(rt, force: false);
            }
            catch
            {
                // Background housekeeping must never take the app down.
            }
            finally
            {
                Interlocked.Exchange(ref rt.Busy, 0);
            }
        }
    }

    private async Task ProfileIfIdleAsync(Runtime rt)
    {
        if (rt.Profiler is null || rt.PendingText is null || rt.Exclusive is null) return;
        if (DateTime.UtcNow - rt.LastActivity < IdleBeforeBackgroundWork) return;

        var text = rt.PendingText;
        rt.PendingText = null;
        rt.ProfilerStatus = "Profiling…";
        await rt.Exclusive(() => Task.Run(() =>
        {
            rt.Prepared.Usage!.Decay(0.97f);
            var n = rt.Profiler.ProfileText(text);
            rt.ProfilerStatus = $"Last pass: {n:N0} tokens at {DateTime.Now:HH:mm:ss}";
        }));
        StateChanged?.Invoke(rt.ModelId);
    }

    private void Rebalance(Runtime rt)
    {
        var usage = rt.Prepared.Usage;
        var policy = rt.Prepared.Settings.RebalancePolicy;
        if (usage is null || rt.Warm is null || policy == RebalancePolicy.Off) return;
        if (usage.UpdatedAt <= rt.UsageSeenAt) return;
        if (DateTime.UtcNow - rt.LastActivity < IdleBeforeBackgroundWork) return;
        rt.UsageSeenAt = usage.UpdatedAt;

        var p = rt.Prepared;
        var current = PlacementPlanner.ExpectedWarmHitRate(rt.Plan, usage);
        var candidate = PlacementPlanner.Plan(rt.Plan.Catalog, rt.Plan.Settings, rt.Plan.Budget, p.Hardware, usage);
        var gain = candidate.WarmHitRate - current;
        if (gain <= Math.Max(p.Settings.RebalanceThreshold, 1e-4))
        {
            rt.Suggestion = null;
            return;
        }

        if (policy == RebalancePolicy.Auto)
        {
            rt.Plan = candidate;
            _ = rt.Warm.ApplyAsync(candidate.WarmRanges, p.Settings.LockWarmTier);
            rt.Suggestion = $"Warm tier re-targeted: hit rate {current:P0} → {candidate.WarmHitRate:P0}.";
        }
        else
        {
            rt.Suggestion = $"Re-targeting the warm tier would raise its hit rate {current:P0} → {candidate.WarmHitRate:P0}.";
        }
        StateChanged?.Invoke(rt.ModelId);
    }

    /// <summary>Applies a pending rebalance suggestion now (Suggest policy).</summary>
    public void ApplySuggestedRebalance(Guid modelId)
    {
        if (!_runtimes.TryGetValue(modelId, out var rt) || rt.Warm is null || rt.Prepared.Usage is null) return;
        var candidate = PlacementPlanner.Plan(rt.Plan.Catalog, rt.Plan.Settings, rt.Plan.Budget, rt.Prepared.Hardware, rt.Prepared.Usage);
        rt.Plan = candidate;
        _ = rt.Warm.ApplyAsync(candidate.WarmRanges, rt.Prepared.Settings.LockWarmTier);
        rt.Suggestion = null;
        StateChanged?.Invoke(modelId);
    }

    private async Task PersistUsageAsync(Runtime rt, bool force)
    {
        var usage = rt.Prepared.Usage;
        if (usage is null || !usage.HasAnyData) return;
        if (!force && (DateTime.UtcNow - rt.LastPersist < PersistEvery || usage.UpdatedAt <= rt.LastPersist)) return;
        rt.LastPersist = DateTime.UtcNow;

        await using var db = await _dbFactory.CreateDbContextAsync();
        db.ExpertUsageSnapshots.Add(new ExpertUsageSnapshot
        {
            ModelId = rt.ModelId,
            Layers = usage.Layers,
            Experts = usage.Experts,
            ObservedTokens = usage.ObservedTokens,
            Concentration = usage.Concentration(),
            Data = usage.Serialize(),
        });
        await db.SaveChangesAsync();

        // Keep the 20 most recent snapshots per model.
        var stale = await db.ExpertUsageSnapshots.Where(s => s.ModelId == rt.ModelId)
            .OrderByDescending(s => s.RecordedAt).Skip(20).Select(s => s.Id).ToListAsync();
        if (stale.Count > 0)
            await db.ExpertUsageSnapshots.Where(s => stale.Contains(s.Id)).ExecuteDeleteAsync();
    }

    // ── Read side ───────────────────────────────────────────────────────────

    public bool IsTiered(Guid modelId) => _runtimes.ContainsKey(modelId);

    public TieringSnapshot? GetSnapshot(Guid modelId)
    {
        if (!_runtimes.TryGetValue(modelId, out var rt)) return null;
        var plan = rt.Plan;
        if (rt.Warm is not null && DateTime.UtcNow - rt.ResidentMeasuredAt > TimeSpan.FromSeconds(15))
        {
            rt.ResidentBytes = rt.Warm.MeasureResidentBytes(plan.WarmRanges);
            rt.ResidentMeasuredAt = DateTime.UtcNow;
        }
        var usage = rt.Prepared.Usage;
        return new TieringSnapshot(
            plan,
            PerformanceEstimator.Estimate(plan, rt.Prepared.Hardware, rt.Prepared.SpeedCalibration),
            rt.Warm?.Status,
            rt.ResidentBytes,
            usage is null ? plan.WarmHitRate : PlacementPlanner.ExpectedWarmHitRate(plan, usage),
            rt.Suggestion,
            rt.Profiler is null && rt.ProfilerStatus == "Off" ? (plan.Catalog.IsMoe ? "Off" : "Dense model — not needed") : rt.ProfilerStatus,
            rt.Profiler?.TokensProfiled ?? 0,
            usage is { HasAnyData: true } ? usage.Concentration() : null);
    }

    /// <summary>Routing histogram for the analytics heatmap: live if running, else the latest saved one.</summary>
    public async Task<ExpertUsageProfile?> GetUsageAsync(Guid modelId)
    {
        if (_runtimes.TryGetValue(modelId, out var rt) && rt.Prepared.Usage is { } live) return live.Clone();
        await using var db = await _dbFactory.CreateDbContextAsync();
        var snap = await db.ExpertUsageSnapshots.AsNoTracking()
            .Where(s => s.ModelId == modelId).OrderByDescending(s => s.RecordedAt).FirstOrDefaultAsync();
        return ExpertUsageProfile.Deserialize(snap?.Data);
    }

    public void Dispose()
    {
        _timer.Dispose();
        foreach (var id in _runtimes.Keys.ToArray()) OnUnloaded(id);
    }
}
