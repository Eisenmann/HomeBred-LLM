using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using HomebredLLM.Data;
using HomebredLLM.Models;
using HomebredLLM.Services;
using HomebredLLM.Services.Tiering;
using LiveChartsCore;
using LiveChartsCore.Defaults;
using LiveChartsCore.Drawing;
using LiveChartsCore.SkiaSharpView;
using LiveChartsCore.SkiaSharpView.Painting;
using Microsoft.EntityFrameworkCore;
using SkiaSharp;
using System.Collections.ObjectModel;

namespace HomebredLLM.ViewModels;

public partial class AnalyticsViewModel(
    IDbContextFactory<AppDbContext> dbFactory,
    AnalyticsRepository analyticsRepo,
    GpuMetricsService gpuService,
    TieringCoordinator tiering) : ObservableObject
{
    // ── Memory tiers & MoE routing ──────────────────────────────────────────
    [ObservableProperty] private TierBreakdownViewModel? _liveTiers;
    [ObservableProperty] private string _liveTierStatus = "";
    [ObservableProperty] private ISeries[] _tierSeries = [];
    [ObservableProperty] private ISeries[] _speedSplitSeries = [];
    [ObservableProperty] private ISeries[] _ioSeries = [];
    [ObservableProperty] private ISeries[] _stallSeries = [];
    [ObservableProperty] private ISeries[] _heatSeries = [];
    // LiveCharts throws on empty axis arrays — always keep one axis.
    [ObservableProperty] private Axis[] _heatXAxis = [new Axis()];
    [ObservableProperty] private Axis[] _heatYAxis = [new Axis()];
    [ObservableProperty] private string _routingSummary = "No routing data yet (MoE models with the profiler enabled).";
    [ObservableProperty] private bool _hasRouting;
    [ObservableProperty] private string _avgDecodeTps = "—";
    [ObservableProperty] private string _avgPrefillTps = "—";

    partial void OnSelectedModelChanged(LocalModel? value)
    {
        if (value is null) return;
        RefreshLiveTiers();
        _ = RefreshHistoricalAsync();
    }

    private void RefreshLiveTiers()
    {
        var model = SelectedModel;
        var snap = model is null ? null : tiering.GetSnapshot(model.Id);
        if (snap is null)
        {
            LiveTiers = null;
            LiveTierStatus = model?.Status == ModelStatus.Running
                ? "Running without tiered placement (Simple mode)."
                : "Start this model to see its live tier placement.";
            return;
        }
        var verdict = CapacityCalculator.VerdictOf(snap.Plan);
        LiveTiers = TierBreakdownViewModel.From(snap.Plan, snap.CalibratedEstimate,
            "Live · " + verdict switch
            {
                FitVerdict.FitsGpu => "all on GPU",
                FitVerdict.FitsGpuAndRam => snap.Plan.UsesGpu ? "GPU + RAM" : "RAM (CPU)",
                FitVerdict.NeedsDisk => "GPU + RAM + disk",
                _ => "over budget",
            },
            TierBreakdownViewModel.VerdictColorOf(verdict));
        var w = snap.Warm;
        LiveTierStatus =
            (w is null ? "No warm tier." :
                $"Warm tier: {TierBreakdownViewModel.Gb(w.PrefetchedBytes)} of {TierBreakdownViewModel.Gb(w.TargetBytes)} prefetched" +
                (w.Locking ? $", {TierBreakdownViewModel.Gb(w.LockedBytes)} locked" : "") +
                (snap.WarmResidentBytes is { } r ? $", {TierBreakdownViewModel.Gb(r)} resident" : "") +
                (w.LockError is { } e ? $" — {e}" : "") + ". ") +
            $"Expected warm hit rate {snap.WarmHitRate:P0}. Profiler: {snap.ProfilerStatus}." +
            (snap.Suggestion is { } sug ? $" {sug}" : "");
    }

    private async Task RefreshRoutingAsync(Guid modelId)
    {
        var usage = await tiering.GetUsageAsync(modelId);
        if (usage is null || !usage.HasAnyData)
        {
            HasRouting = false;
            HeatSeries = [];
            RoutingSummary = "No routing data yet (MoE models with the profiler enabled).";
            return;
        }

        // Weight = share relative to uniform (1.0 = average expert); capped so a few
        // very hot experts don't wash out the colour scale.
        var points = new List<WeightedPoint>(usage.Layers * usage.Experts);
        for (var l = 0; l < usage.Layers; l++)
        {
            if (!usage.HasData(l)) continue;
            var shares = usage.LayerShares(l);
            for (var e = 0; e < usage.Experts; e++)
                points.Add(new WeightedPoint(e, l, Math.Min(4, shares[e] * usage.Experts)));
        }

        HasRouting = points.Count > 0;
        HeatSeries =
        [
            new HeatSeries<WeightedPoint>
            {
                Values = points,
                HeatMap =
                [
                    new LvcColor(17, 24, 39),
                    new LvcColor(14, 165, 233),
                    new LvcColor(245, 158, 11),
                    new LvcColor(239, 68, 68),
                ],
                XToolTipLabelFormatter = p => $"expert {p.Coordinate.SecondaryValue:0}",
                YToolTipLabelFormatter = p => $"layer {p.Coordinate.PrimaryValue:0} · {p.Coordinate.TertiaryValue:0.00}× average",
            }
        ];
        HeatXAxis = [ChartTheme.Axis("expert")];
        HeatYAxis = [ChartTheme.Axis("layer")];
        RoutingSummary = $"{usage.ObservedTokens:N0} routed tokens observed (decayed). " +
                         $"Top 20% of experts receive {usage.Concentration(0.2):P0} of routing " +
                         $"(uniform would be 20%) — the higher this is, the more the RAM tier helps.";
    }
    [ObservableProperty] private ObservableCollection<LocalModel> _models = [];
    [ObservableProperty] private LocalModel? _selectedModel;
    [ObservableProperty] private DateTime _fromDate = DateTime.UtcNow.AddHours(-1);
    [ObservableProperty] private DateTime _toDate = DateTime.UtcNow;

    // Summary
    [ObservableProperty] private string _avgGpuUtil = "—";
    [ObservableProperty] private string _peakVram = "—";
    [ObservableProperty] private string _avgTps = "—";
    [ObservableProperty] private string _avgTtft = "—";
    [ObservableProperty] private string _totalTokens = "—";

    // Live metric for header gauges
    [ObservableProperty] private float _liveGpuPct;
    [ObservableProperty] private float _liveCpuPct;
    [ObservableProperty] private float _liveRamMb;
    [ObservableProperty] private float _liveGpuTempC;

    // Charts
    [ObservableProperty] private ISeries[] _gpuSeries = [];
    [ObservableProperty] private ISeries[] _vramSeries = [];
    [ObservableProperty] private ISeries[] _tpsSeries = [];
    [ObservableProperty] private ISeries[] _cpuSeries = [];
    [ObservableProperty] private Axis[] _timeAxis = [];

    [ObservableProperty] private string _deleteStatus = "";

    private System.Timers.Timer? _liveTimer;
    private readonly List<float?> _gpuPoints = [];
    private readonly List<float?> _tpsPoints = [];

    public async Task InitializeAsync()
    {
        await using var db = await dbFactory.CreateDbContextAsync();
        var models = await db.Models.OrderBy(m => m.Name).ToListAsync();
        Models = new ObservableCollection<LocalModel>(models);
        SelectedModel = models.FirstOrDefault();

        _liveTimer = new System.Timers.Timer(5000);
        _liveTimer.Elapsed += (_, _) =>
        {
            RefreshLiveMetrics();
            Dispatcher.UIThread.Post(RefreshLiveTiers);
        };
        _liveTimer.Start();

        if (SelectedModel is not null)
            await RefreshHistoricalAsync();
    }

    private void RefreshLiveMetrics()
    {
        var snap = gpuService.Sample();
        Dispatcher.UIThread.Post(() =>
        {
            LiveGpuPct   = snap.GpuUtilPct ?? 0;
            LiveCpuPct   = snap.CpuUtilPct;
            LiveRamMb    = snap.RamUsedMb;
            LiveGpuTempC = snap.GpuTempC ?? 0;

            // Rolling 60-point live chart
            _gpuPoints.Add(snap.GpuUtilPct);
            if (_gpuPoints.Count > 60) _gpuPoints.RemoveAt(0);

            GpuSeries =
            [
                new LineSeries<float?>
                {
                    Values       = _gpuPoints.ToArray(),
                    Name         = "GPU %",
                    Stroke       = new SolidColorPaint(SKColor.Parse("#0EA5E9"), 2),
                    Fill         = new SolidColorPaint(SKColor.Parse("#0EA5E91A")),
                    GeometrySize = 0,
                }
            ];
        });
    }

    [RelayCommand]
    private async Task RefreshAsync()
    {
        if (SelectedModel is null) return;
        await RefreshHistoricalAsync();
    }

    private async Task RefreshHistoricalAsync()
    {
        if (SelectedModel is null) return;

        var summary = await analyticsRepo.GetSummaryAsync(SelectedModel.Id, FromDate, ToDate);
        AvgGpuUtil = summary.AvgGpuUtilPct.HasValue ? $"{summary.AvgGpuUtilPct:F1}%" : "—";
        PeakVram = summary.PeakGpuMemUsedMb.HasValue ? $"{summary.PeakGpuMemUsedMb:F0} MB" : "—";
        AvgTps = summary.AvgTokensPerSecond.HasValue ? $"{summary.AvgTokensPerSecond:F1} t/s" : "—";
        AvgTtft = summary.AvgTtftMs.HasValue ? $"{summary.AvgTtftMs:F0} ms" : "—";
        TotalTokens = (summary.TotalPromptTokens + summary.TotalOutputTokens).ToString("N0");

        await RefreshRoutingAsync(SelectedModel.Id);

        var points = await analyticsRepo.QueryAsync(SelectedModel.Id, FromDate, ToDate);
        if (points.Count == 0) return;

        var decodeVals = points.Where(p => p.DecodeTokensPerSecond.HasValue).Select(p => p.DecodeTokensPerSecond!.Value).ToList();
        var prefillVals = points.Where(p => p.PrefillTokensPerSecond.HasValue).Select(p => p.PrefillTokensPerSecond!.Value).ToList();
        AvgDecodeTps = decodeVals.Count > 0 ? $"{decodeVals.Average():F1} t/s" : "—";
        AvgPrefillTps = prefillVals.Count > 0 ? $"{prefillVals.Average():F0} t/s" : "—";

        static LineSeries<float?> L(string name, float?[] v, string color, float width = 2) => new()
        {
            Values = v, Name = name, GeometrySize = 0, Fill = null,
            Stroke = new SolidColorPaint(SKColor.Parse(color), width),
        };

        TierSeries =
        [
            L("VRAM (planned)", points.Select(p => p.TierVramMb).ToArray(), "#8B5CF6"),
            L("RAM warm (planned)", points.Select(p => p.TierWarmMb).ToArray(), "#0EA5E9"),
            L("RAM warm (resident)", points.Select(p => p.WarmResidentMb).ToArray(), "#10B981", 1.5f),
            L("Disk (cold)", points.Select(p => p.TierColdMb).ToArray(), "#F59E0B"),
        ];
        SpeedSplitSeries =
        [
            L("Decode tok/s", points.Select(p => p.DecodeTokensPerSecond).ToArray(), "#10B981"),
            L("Estimated tok/s", points.Select(p => p.EstimatedTokensPerSecond).ToArray(), "#6B7280", 1.5f),
            L("Prefill tok/s ÷ 10", points.Select(p => p.PrefillTokensPerSecond / 10).ToArray(), "#0EA5E9", 1.5f),
        ];
        IoSeries =
        [
            L("Disk read MB/s", points.Select(p => p.DiskReadMbps).ToArray(), "#F59E0B"),
            L("PCIe RX MB/s", points.Select(p => p.PcieRxMbps).ToArray(), "#8B5CF6"),
            L("PCIe TX MB/s", points.Select(p => p.PcieTxMbps).ToArray(), "#EC4899", 1.5f),
            L("Major faults/s", points.Select(p => p.MajorFaultsPerSec).ToArray(), "#EF4444", 1.5f),
        ];
        static StackedAreaSeries<float?> S(string name, float?[] v, string color) => new()
        {
            Values = v, Name = name, GeometrySize = 0,
            Fill = new SolidColorPaint(SKColor.Parse(color).WithAlpha(160)),
            Stroke = null,
        };
        StallSeries =
        [
            S("GPU ms", points.Select(p => p.EstGpuMsPerToken).ToArray(), "#8B5CF6"),
            S("CPU ms", points.Select(p => p.EstCpuMsPerToken).ToArray(), "#0EA5E9"),
            S("Disk ms", points.Select(p => p.EstDiskMsPerToken).ToArray(), "#F59E0B"),
            S("Sync ms", points.Select(p => p.EstSyncMsPerToken).ToArray(), "#6B7280"),
        ];

        var gpuVals = points.Select(p => (float?)p.GpuUtilizationPct).ToArray();
        var vramUsed = points.Select(p => (float?)p.GpuMemoryUsedMb).ToArray();
        var vramTotal = points.Select(p => (float?)p.GpuMemoryTotalMb).ToArray();
        var tpsVals = points.Select(p => (float?)p.TokensPerSecond).ToArray();
        var cpuVals = points.Select(p => (float?)p.CpuUtilizationPct).ToArray();

        GpuSeries =
        [
            new LineSeries<float?> { Values = gpuVals, Name = "GPU %", Stroke = new SolidColorPaint(SKColor.Parse("#0EA5E9"), 2), GeometrySize = 0, Fill = new SolidColorPaint(SKColor.Parse("#0EA5E914")) }
        ];

        VramSeries =
        [
            new LineSeries<float?> { Values = vramUsed, Name = "VRAM Used", Stroke = new SolidColorPaint(SKColor.Parse("#8B5CF6"), 2), GeometrySize = 0 },
            new LineSeries<float?> { Values = vramTotal, Name = "VRAM Total", Stroke = new SolidColorPaint(SKColor.Parse("#374151"), 1.5f), GeometrySize = 0 },
        ];

        TpsSeries =
        [
            new LineSeries<float?> { Values = tpsVals, Name = "Tokens/s", Stroke = new SolidColorPaint(SKColor.Parse("#10B981"), 2), GeometrySize = 0, Fill = new SolidColorPaint(SKColor.Parse("#10B98114")) }
        ];

        CpuSeries =
        [
            new LineSeries<float?> { Values = cpuVals, Name = "CPU %", Stroke = new SolidColorPaint(SKColor.Parse("#F59E0B"), 2), GeometrySize = 0 }
        ];
    }

    [RelayCommand]
    private async Task DeleteMetricsAsync()
    {
        if (SelectedModel is null) return;
        var count = await analyticsRepo.DeleteAsync(SelectedModel.Id, FromDate, ToDate);
        DeleteStatus = $"Deleted {count} records.";
        await Task.Delay(3000);
        DeleteStatus = "";
        await RefreshHistoricalAsync();
    }

    public void Cleanup() => _liveTimer?.Stop();
}
