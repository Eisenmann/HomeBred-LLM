using System.Text.RegularExpressions;

namespace HomebredLLM.Services.Tiering;

public enum CapacityStrategy
{
    /// <summary>Everything (weights, KV, buffers) in VRAM.</summary>
    GpuOnly,
    /// <summary>VRAM + the RAM budget; nothing read from disk.</summary>
    GpuAndRam,
    /// <summary>VRAM + RAM + memory-mapped disk (MoE: cold experts).</summary>
    GpuRamDisk,
}

public enum FitVerdict { FitsGpu, FitsGpuAndRam, NeedsDisk, TooLarge }

/// <summary>"What model sizes can I run?" input.</summary>
public sealed record CapacityQuery
{
    public required HardwareSpec Hardware { get; init; }
    public required TierBudget Budget { get; init; }
    public RuntimeSettings Settings { get; init; } = new();
    public string Quantization { get; init; } = "Q4_K_M";

    /// <summary>MoE when set: active parameters as a fraction of total (e.g. 0.1 for 235B-A22B).</summary>
    public double? MoeActiveRatio { get; init; }
    public int ExpertCount { get; init; } = 128;
    public int ExpertUsedCount { get; init; } = 8;

    /// <summary>Answers "the largest model that still runs at ≥ this speed" (0 = no limit).</summary>
    public double MinTokensPerSecond { get; init; }
}

public sealed record CapacityRow(
    CapacityStrategy Strategy,
    double MaxParamsB,
    double? TokensPerSecond,
    long WeightBytes,
    string Note);

/// <summary>"Can I run this model?" answer.</summary>
public sealed record FitReport(
    FitVerdict Verdict,
    PlacementPlan Plan,
    int? MaxContextFullyOnGpu,
    string? LargestQuantFullyOnGpu,
    IReadOnlyList<string> Suggestions)
{
    public string VerdictLabel => Verdict switch
    {
        FitVerdict.FitsGpu => "Fits on GPU",
        FitVerdict.FitsGpuAndRam when !Plan.UsesGpu => "Fits in RAM (CPU)",
        FitVerdict.FitsGpuAndRam => "Fits with RAM offload",
        FitVerdict.NeedsDisk => "Needs disk tier",
        _ => "Too large",
    };
}

/// <summary>
/// Inverse of the planner: binary-searches the parameter count that still
/// fits each strategy (and optional minimum speed), and assesses concrete
/// models with actionable suggestions.
/// </summary>
public static class CapacityCalculator
{
    private const double MinB = 0.1;
    private const double MaxB = 4000;

    /// <summary>Disk strategy needs a speed floor, otherwise "any size fits" (just slowly).</summary>
    public const double DefaultDiskMinTps = 0.5;

    public static IReadOnlyList<CapacityRow> MaxModelSizes(CapacityQuery q)
    {
        var bpw = GgmlTypeInfo.BitsPerWeight(q.Quantization) ?? 4.85;
        var rows = new List<CapacityRow>();
        foreach (var strategy in Enum.GetValues<CapacityStrategy>())
        {
            var minTps = q.MinTokensPerSecond;
            if (strategy == CapacityStrategy.GpuRamDisk) minTps = Math.Max(minTps, DefaultDiskMinTps);

            if (strategy == CapacityStrategy.GpuOnly && !q.Hardware.HasGpuBackend)
            {
                rows.Add(new CapacityRow(strategy, 0, null, 0, "No GPU device in the llama.cpp backend."));
                continue;
            }

            var best = Search(p => Evaluate(q, strategy, p, bpw) is { } plan && plan.Estimate.TokensPerSecond >= minTps);
            if (best <= 0)
            {
                rows.Add(new CapacityRow(strategy, 0, null, 0, "Nothing fits with these budgets/settings."));
                continue;
            }

            var plan = Evaluate(q, strategy, best, bpw)!;
            var note = strategy switch
            {
                CapacityStrategy.GpuOnly => "Fastest: everything in VRAM.",
                CapacityStrategy.GpuAndRam when q.MoeActiveRatio is null =>
                    "Dense: CPU layers run at RAM bandwidth.",
                CapacityStrategy.GpuAndRam => "Experts not in VRAM are computed on the CPU from RAM.",
                _ when q.MoeActiveRatio is null =>
                    $"Dense on disk streams layers every token; limited by the {minTps:0.#} tok/s floor.",
                _ => $"Cold experts paged from disk; limited by the {minTps:0.#} tok/s floor.",
            };
            if (plan.Estimate.TokensPerSecond > 0 && strategy != CapacityStrategy.GpuOnly)
                note += $" Bottleneck: {plan.Estimate.Bottleneck}.";
            rows.Add(new CapacityRow(strategy, Math.Round(best, best < 10 ? 1 : 0), plan.Estimate.TokensPerSecond,
                plan.Catalog.TotalBytes, note));
        }
        return rows;
    }

    /// <summary>Estimated tok/s per strategy for a sweep of model sizes (null = doesn't fit).</summary>
    public static IReadOnlyList<(double ParamsB, double? GpuOnly, double? GpuAndRam, double? GpuRamDisk)> SpeedCurve(
        CapacityQuery q, IEnumerable<double> sizesB)
    {
        var bpw = GgmlTypeInfo.BitsPerWeight(q.Quantization) ?? 4.85;
        double? Tps(CapacityStrategy s, double p) => Evaluate(q, s, p, bpw)?.Estimate.TokensPerSecond;
        return sizesB.Select(p => (p,
            q.Hardware.HasGpuBackend ? Tps(CapacityStrategy.GpuOnly, p) : null,
            Tps(CapacityStrategy.GpuAndRam, p),
            Tps(CapacityStrategy.GpuRamDisk, p))).ToList();
    }

    /// <summary>Plan for a synthetic model under one strategy; null when it doesn't fit.</summary>
    public static PlacementPlan? Evaluate(CapacityQuery q, CapacityStrategy strategy, double paramsB, double bpw)
    {
        var shape = q.MoeActiveRatio is > 0 and < 1
            ? ModelShape.Synthetic(paramsB, paramsB * q.MoeActiveRatio, q.ExpertCount, q.ExpertUsedCount)
            : ModelShape.Synthetic(paramsB);
        var catalog = TensorCatalog.Synthetic(shape, bpw);

        var budget = strategy switch
        {
            CapacityStrategy.GpuOnly => q.Budget with { RamBytes = q.Budget.RamBytes, AllowDisk = false },
            CapacityStrategy.GpuAndRam => q.Budget with { AllowDisk = false },
            _ => q.Budget with { AllowDisk = true },
        };
        var plan = PlacementPlanner.Plan(catalog, q.Settings, budget, q.Hardware);

        return strategy switch
        {
            CapacityStrategy.GpuOnly => plan.CpuTensors.All(t => t.Kind == TensorKind.Embedding) && plan.KvOnGpu ? plan : null,
            CapacityStrategy.GpuAndRam => plan.Fits && plan.ColdBytes == 0 ? plan : null,
            _ => plan.Fits ? plan : null,
        };
    }

    /// <summary>Largest p in [MinB, MaxB] with predicate true (assumes monotone), 0 if none.</summary>
    private static double Search(Func<double, bool> ok)
    {
        if (!ok(MinB)) return 0;
        if (ok(MaxB)) return MaxB;
        double lo = Math.Log(MinB), hi = Math.Log(MaxB);
        for (var i = 0; i < 36; i++)
        {
            var mid = (lo + hi) / 2;
            if (ok(Math.Exp(mid))) lo = mid; else hi = mid;
        }
        return Math.Exp(lo);
    }

    /// <summary>Full assessment of a concrete (or synthetic) model under the user's budgets.</summary>
    public static FitReport Assess(TensorCatalog catalog, RuntimeSettings settings, TierBudget budget,
        HardwareSpec hw, ExpertUsageProfile? profile = null, string? currentQuant = null)
    {
        var plan = PlacementPlanner.Plan(catalog, settings, budget, hw, profile);
        var verdict = VerdictOf(plan);
        var suggestions = new List<string>();

        int? maxCtx = null;
        if (hw.HasGpuBackend)
        {
            bool AllGpu(int ctx)
            {
                var p = PlacementPlanner.Plan(catalog, settings with { ContextSize = ctx }, budget with { AllowDisk = false }, hw, profile);
                return VerdictOf(p) == FitVerdict.FitsGpu;
            }
            if (AllGpu(512))
            {
                int lo = 512, hi = Math.Max(settings.ContextSize, 512) * 64;
                if (AllGpu(hi)) maxCtx = hi;
                else
                {
                    while (hi - lo > 256)
                    {
                        var mid = (lo + hi) / 2;
                        if (AllGpu(mid)) lo = mid; else hi = mid;
                    }
                    maxCtx = lo / 256 * 256;
                }
            }
        }

        string? bestQuant = null;
        var currentBpw = GgmlTypeInfo.BitsPerWeight(currentQuant)
                         ?? (catalog.Shape.TotalParameters > 0 ? catalog.TotalBytes * 8.0 / catalog.Shape.TotalParameters : (double?)null);
        if (hw.HasGpuBackend && currentBpw is > 0)
        {
            foreach (var label in GgmlTypeInfo.CommonQuantLabels)
            {
                var bpw = GgmlTypeInfo.BitsPerWeight(label)!.Value;
                var scaled = TensorCatalog.Synthetic(catalog.Shape with { IsSynthetic = true }, bpw);
                var p = PlacementPlanner.Plan(scaled, settings, budget with { AllowDisk = false }, hw);
                if (VerdictOf(p) == FitVerdict.FitsGpu) { bestQuant = label; break; }
            }
        }

        if (verdict != FitVerdict.FitsGpu && maxCtx is { } c && c < settings.ContextSize)
            suggestions.Add($"Context ≤ {c:N0} tokens would keep everything on the GPU.");
        if (verdict != FitVerdict.FitsGpu && bestQuant is not null && bestQuant != currentQuant)
            suggestions.Add($"A {bestQuant} build of this model would fit fully in VRAM.");
        if (plan.Settings.KvType == KvCacheType.F16 && verdict is not FitVerdict.FitsGpu && plan.UsesGpu)
            suggestions.Add("A Q8_0 KV cache halves KV memory with negligible quality loss.");
        if (catalog.IsMoe && plan.ColdBytes > 0)
            suggestions.Add($"Raising the RAM budget by {PlacementPlanner.Gb(plan.ColdBytes)} removes all disk reads.");
        if (catalog.IsMoe && profile is null && plan.ColdBytes > 0)
            suggestions.Add("Enable the routing profiler: a learned hot set usually beats the uniform guess.");

        return new FitReport(verdict, plan, maxCtx, bestQuant, suggestions);
    }

    public static FitVerdict VerdictOf(PlacementPlan plan)
    {
        if (!plan.Fits) return FitVerdict.TooLarge;
        if (plan.ColdBytes > 0) return FitVerdict.NeedsDisk;
        var onlyEmbeddingOnCpu = plan.CpuTensors.All(t => t.Kind == TensorKind.Embedding);
        return plan.UsesGpu && onlyEmbeddingOnCpu && plan.KvOnGpu ? FitVerdict.FitsGpu : FitVerdict.FitsGpuAndRam;
    }

    private static readonly Regex ParamsPattern = new(@"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)[Bb](?![A-Za-z])", RegexOptions.Compiled);
    private static readonly Regex ActivePattern = new(@"[-_]A(\d+(?:\.\d+)?)B", RegexOptions.Compiled | RegexOptions.IgnoreCase);
    private static readonly Regex MixtralPattern = new(@"(\d+)x(\d+(?:\.\d+)?)B", RegexOptions.Compiled | RegexOptions.IgnoreCase);

    /// <summary>
    /// Quick verdict from just a file size and name (Hugging Face listings):
    /// parameters are derived from size ÷ bits-per-weight, MoE-ness from
    /// naming conventions like "-A3B" or "8x7B".
    /// </summary>
    public static FitReport? AssessFile(long sizeBytes, string? quantLabel, string? fileName,
        RuntimeSettings settings, TierBudget budget, HardwareSpec hw)
    {
        if (sizeBytes <= 0) return null;
        var bpw = GgmlTypeInfo.BitsPerWeight(quantLabel) ?? GgmlTypeInfo.BitsPerWeight(fileName) ?? 4.85;
        var paramsB = sizeBytes * 8.0 / bpw / 1e9;

        double? activeB = null;
        int experts = 128, used = 8;
        if (fileName is not null)
        {
            var a = ActivePattern.Match(fileName);
            if (a.Success && double.TryParse(a.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture, out var av))
                activeB = av;
            var m = MixtralPattern.Match(fileName);
            if (activeB is null && m.Success)
            {
                experts = int.Parse(m.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture);
                used = 2;
                activeB = paramsB * 0.28; // Mixtral-style: 2 of 8 experts + shared attention
            }
        }

        var shape = activeB is > 0 && activeB < paramsB
            ? ModelShape.Synthetic(paramsB, activeB, experts, used)
            : ModelShape.Synthetic(paramsB);
        var catalog = TensorCatalog.Synthetic(shape, bpw);
        return Assess(catalog, settings, budget, hw, null, quantLabel);
    }

    /// <summary>Parameter count (billions) parsed from a model name like "Llama-3.1-70B"; null if absent.</summary>
    public static double? ParamsFromName(string? name)
    {
        if (string.IsNullOrEmpty(name)) return null;
        var mm = MixtralPattern.Match(name);
        if (mm.Success) return null; // ambiguous for MoE naming; size-based estimate is better
        var m = ParamsPattern.Match(name);
        return m.Success && double.TryParse(m.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture, out var v) ? v : null;
    }
}
