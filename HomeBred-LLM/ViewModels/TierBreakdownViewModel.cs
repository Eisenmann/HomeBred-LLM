using HomebredLLM.Services.Tiering;

namespace HomebredLLM.ViewModels;

/// <summary>
/// Display model for one placement plan: a stacked VRAM / RAM / disk bar,
/// the speed estimate with its per-token breakdown, verdict, warnings and
/// suggestions. Immutable — view models swap in a new instance.
/// </summary>
public sealed class TierBreakdownViewModel
{
    public const double BarWidth = 560;

    public string Verdict { get; init; } = "";
    public string VerdictColor { get; init; } = "#6B7280";

    public string VramText { get; init; } = "";
    public string WarmText { get; init; } = "";
    public string ColdText { get; init; } = "";
    public double VramWidth { get; init; }
    public double WarmWidth { get; init; }
    public double ColdWidth { get; init; }

    public string SpeedText { get; init; } = "";
    public string BreakdownText { get; init; } = "";
    public string DetailText { get; init; } = "";
    public IReadOnlyList<string> Warnings { get; init; } = [];
    public IReadOnlyList<string> Suggestions { get; init; } = [];
    public bool HasWarnings => Warnings.Count > 0;
    public bool HasSuggestions => Suggestions.Count > 0;

    public static TierBreakdownViewModel From(FitReport report) =>
        From(report.Plan, report.Plan.Estimate, report.VerdictLabel, VerdictColorOf(report.Verdict), report.Suggestions);

    public static string VerdictColorOf(FitVerdict v) => v switch
    {
        FitVerdict.FitsGpu => "#10B981",
        FitVerdict.FitsGpuAndRam => "#0EA5E9",
        FitVerdict.NeedsDisk => "#F59E0B",
        _ => "#EF4444",
    };

    public static TierBreakdownViewModel From(PlacementPlan plan, PerTokenEstimate est, string verdict, string color,
        IReadOnlyList<string>? suggestions = null)
    {
        double vram = plan.VramWeightBytes, warm = plan.WarmBytes, cold = plan.ColdBytes;
        var total = Math.Max(1, vram + warm + cold);
        var catalog = plan.Catalog;
        var detail = $"{catalog.Shape.TotalParameters / 1e9:F1}B params" +
                     (catalog.IsMoe
                         ? $" · MoE {catalog.Shape.ExpertUsedCount}/{catalog.Shape.ExpertCount} experts · {catalog.Shape.ActiveParameters / 1e9:F1}B active"
                         : " · dense") +
                     $" · weights {Gb(catalog.TotalBytes)}" +
                     $" · KV {Gb(plan.VramKvBytes + plan.RamKvBytes)} ({(plan.KvOnGpu ? "VRAM" : "RAM")})" +
                     (catalog.IsMoe && plan.CpuTensors.Any(t => t.IsExpert) ? $" · warm-tier hit rate {plan.WarmHitRate:P0}" : "") +
                     (plan.ExpertSlotsPerLayer > 0
                         ? $" · VRAM expert cache {plan.ExpertSlotsPerLayer} slots/layer ({Gb(plan.ExpertCacheBytes)}), hit rate {plan.ExpertCacheHitRate:P0}"
                         : "");

        return new TierBreakdownViewModel
        {
            Verdict = verdict,
            VerdictColor = color,
            VramText = $"VRAM {Gb(plan.VramTotalBytes)} (weights {Gb(plan.VramWeightBytes)})",
            WarmText = $"RAM {Gb(plan.RamTotalBytes)} (warm weights {Gb(plan.WarmBytes)})",
            ColdText = $"Disk {Gb(plan.ColdBytes)}",
            VramWidth = BarWidth * vram / total,
            WarmWidth = BarWidth * warm / total,
            ColdWidth = BarWidth * cold / total,
            SpeedText = est.TokensPerSecond > 0 ? $"≈ {est.TokensPerSecond:F1} tok/s" : "—",
            BreakdownText = $"per token: GPU {est.GpuMs:F1} ms · CPU {est.CpuMs:F1} ms · disk {est.DiskMs:F1} ms · sync {est.SyncMs:F1} ms — bottleneck: {est.Bottleneck}",
            DetailText = detail,
            Warnings = plan.Warnings,
            Suggestions = suggestions ?? [],
        };
    }

    public static string Gb(double bytes) => bytes >= 1L << 30
        ? $"{bytes / (1L << 30):F1} GB"
        : $"{bytes / (1L << 20):F0} MB";
}
