using System.Collections.ObjectModel;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using HomebredLLM.Data;
using HomebredLLM.Models;
using HomebredLLM.Services.Tiering;
using LiveChartsCore;
using LiveChartsCore.SkiaSharpView;
using LiveChartsCore.SkiaSharpView.Painting;
using Microsoft.EntityFrameworkCore;
using SkiaSharp;

namespace HomebredLLM.ViewModels;

public sealed record CapacityRowView(string Strategy, string MaxSize, string Speed, string Weights, string Note);

/// <summary>
/// "What can I run?" — inverse of the tier planner: for the configured
/// budgets and runtime settings, the largest model per strategy and the
/// expected speed, plus "can I run this?" for any local GGUF model.
/// </summary>
public partial class CalculatorViewModel(
    HardwareProbe probe,
    TieringCoordinator tiering,
    IDbContextFactory<AppDbContext> dbFactory) : ObservableObject
{
    public IReadOnlyList<string> QuantOptions { get; } = GgmlTypeInfo.CommonQuantLabels;
    public IReadOnlyList<KvCacheType> KvOptions { get; } = Enum.GetValues<KvCacheType>();
    public IReadOnlyList<string> KindOptions { get; } = ["Dense", "MoE"];

    // Hardware
    [ObservableProperty] private string _hardwareSummary = "Detecting hardware…";
    [ObservableProperty] private string _benchmarkStatus = "";
    [ObservableProperty] private bool _isBenchmarking;
    [ObservableProperty] private double _vramTotalGb;
    [ObservableProperty] private double _ramTotalGb;

    // Inputs
    [ObservableProperty] private double _vramBudgetGb;
    [ObservableProperty] private double _ramBudgetGb;
    [ObservableProperty] private string _quantization = "Q4_K_M";
    [ObservableProperty] private int _contextSize = 8192;
    [ObservableProperty] private KvCacheType _kvType = KvCacheType.F16;
    [ObservableProperty] private bool _flashAttention = true;
    [ObservableProperty] private int _parallelSequences = 1;
    [ObservableProperty] private string _modelKind = "Dense";
    [ObservableProperty] private double _moeActivePercent = 10;
    [ObservableProperty] private int _expertCount = 128;
    [ObservableProperty] private int _expertsUsed = 8;
    [ObservableProperty] private double _minTokensPerSecond;

    // Results
    [ObservableProperty] private ObservableCollection<CapacityRowView> _rows = [];
    [ObservableProperty] private ISeries[] _speedSeries = [];
    // LiveCharts throws on empty axis arrays — always keep one axis.
    [ObservableProperty] private Axis[] _sizeAxis = [new Axis()];
    [ObservableProperty] private Axis[] _speedAxis = [new Axis()];
    [ObservableProperty] private bool _isMoe;

    // "Can I run this?"
    [ObservableProperty] private ObservableCollection<LocalModel> _localModels = [];
    [ObservableProperty] private LocalModel? _selectedModel;
    [ObservableProperty] private TierBreakdownViewModel? _modelFit;
    [ObservableProperty] private string _modelFitStatus = "";

    private HardwareSpec _hw = HardwareSpec.Fallback;
    private CancellationTokenSource? _pending;
    private bool _initialized;

    private static readonly double[] CurveSizes = [1, 3, 8, 14, 32, 70, 123, 235, 405, 671, 1000];

    public async Task InitializeAsync()
    {
        if (_initialized) { await RefreshModelsAsync(); return; }
        _initialized = true;
        await LoadHardwareAsync();
        await RefreshModelsAsync();
        Schedule();
    }

    private async Task LoadHardwareAsync()
    {
        _hw = await probe.GetCurrentSpecAsync();
        var profile = await probe.GetProfileAsync();
        VramTotalGb = Math.Round(_hw.VramTotalBytes / (double)(1L << 30), 1);
        RamTotalGb = Math.Round(_hw.RamTotalBytes / (double)(1L << 30), 1);
        if (VramBudgetGb <= 0) VramBudgetGb = Math.Max(0, Math.Round(VramTotalGb - 1, 1));
        if (RamBudgetGb <= 0) RamBudgetGb = Math.Max(1, Math.Round(Math.Min(RamTotalGb - 4, RamTotalGb * 0.8)));

        var gpu = _hw.HasGpuBackend
            ? $"{_hw.GpuName ?? "GPU"} · {VramTotalGb:F1} GB @ {_hw.VramBandwidthGBs:F0} GB/s"
            : $"No GPU device in llama.cpp backend ({profile.BackendDevices})";
        var disk = profile.DiskSequentialMBs > 0
            ? $"disk {profile.DiskSequentialMBs:F0} MB/s seq, {profile.DiskLatencyMs:F2} ms latency{(_hw.DiskLooksRotational ? " (HDD-like)" : "")}"
            : "disk not benchmarked yet";
        if (_hw.HasExpertCache) gpu += " · per-expert VRAM cache available";
        HardwareSummary = $"{gpu}\nRAM {RamTotalGb:F0} GB @ {_hw.RamBandwidthGBs:F0} GB/s · {_hw.CpuCores} threads · {disk}" +
                          $"\nMeasured {profile.MeasuredAt.ToLocalTime():g} · speed calibration ×{profile.SpeedCalibration:F2}";
    }

    public async Task RefreshModelsAsync()
    {
        await using var db = await dbFactory.CreateDbContextAsync();
        var models = await db.Models.AsNoTracking()
            .Where(m => m.Format == ModelFormat.Gguf && m.LocalPath != null)
            .OrderBy(m => m.Name).ToListAsync();
        var keep = SelectedModel?.Id;
        LocalModels = new ObservableCollection<LocalModel>(models);
        SelectedModel = models.FirstOrDefault(m => m.Id == keep) ?? models.FirstOrDefault();
    }

    [RelayCommand]
    private async Task RunBenchmarkAsync()
    {
        if (IsBenchmarking) return;
        IsBenchmarking = true;
        try
        {
            var progress = new Progress<string>(s => BenchmarkStatus = s);
            var target = LocalModels.Select(m => m.LocalPath).FirstOrDefault(p => p is not null && File.Exists(p));
            await probe.RunBenchmarkAsync(target, progress);
            await LoadHardwareAsync();
            BenchmarkStatus = "Benchmark complete.";
            Schedule();
        }
        catch (Exception ex)
        {
            BenchmarkStatus = $"Benchmark failed: {ex.Message}";
        }
        finally { IsBenchmarking = false; }
    }

    partial void OnVramBudgetGbChanged(double value) => Schedule();
    partial void OnRamBudgetGbChanged(double value) => Schedule();
    partial void OnQuantizationChanged(string value) => Schedule();
    partial void OnContextSizeChanged(int value) => Schedule();
    partial void OnKvTypeChanged(KvCacheType value) => Schedule();
    partial void OnFlashAttentionChanged(bool value) => Schedule();
    partial void OnParallelSequencesChanged(int value) => Schedule();
    partial void OnModelKindChanged(string value) { IsMoe = value == "MoE"; Schedule(); }
    partial void OnMoeActivePercentChanged(double value) => Schedule();
    partial void OnExpertCountChanged(int value) => Schedule();
    partial void OnExpertsUsedChanged(int value) => Schedule();
    partial void OnMinTokensPerSecondChanged(double value) => Schedule();
    partial void OnSelectedModelChanged(LocalModel? value) => Schedule();

    private RuntimeSettings Settings => new()
    {
        ContextSize = Math.Clamp(ContextSize, 256, 1 << 20),
        ParallelSequences = Math.Clamp(ParallelSequences, 1, 64),
        KvType = KvType,
        FlashAttention = FlashAttention || KvType is KvCacheType.Q8_0 or KvCacheType.Q4_0,
    };

    private TierBudget Budget => new()
    {
        VramBytes = (long)(Math.Max(0, VramBudgetGb) * (1L << 30)),
        RamBytes = (long)(Math.Max(0, RamBudgetGb) * (1L << 30)),
        AllowDisk = true,
    };

    /// <summary>Debounced recompute on a worker thread (each pass runs ~100 plans).</summary>
    private void Schedule()
    {
        if (!_initialized) return;
        _pending?.Cancel();
        var cts = _pending = new CancellationTokenSource();
        _ = Task.Run(async () =>
        {
            try
            {
                await Task.Delay(250, cts.Token);
                await RecomputeAsync(cts.Token);
            }
            catch (OperationCanceledException) { }
            catch (Exception ex)
            {
                Dispatcher.UIThread.Post(() => ModelFitStatus = $"Calculation failed: {ex.Message}");
            }
        });
    }

    private async Task RecomputeAsync(CancellationToken ct)
    {
        var query = new CapacityQuery
        {
            Hardware = _hw,
            Budget = Budget,
            Settings = Settings,
            Quantization = Quantization,
            MoeActiveRatio = ModelKind == "MoE" ? Math.Clamp(MoeActivePercent, 1, 90) / 100.0 : null,
            ExpertCount = Math.Max(2, ExpertCount),
            ExpertUsedCount = Math.Clamp(ExpertsUsed, 1, Math.Max(2, ExpertCount)),
            MinTokensPerSecond = Math.Max(0, MinTokensPerSecond),
        };

        var rows = CapacityCalculator.MaxModelSizes(query);
        ct.ThrowIfCancellationRequested();
        var curve = CapacityCalculator.SpeedCurve(query, CurveSizes);
        ct.ThrowIfCancellationRequested();

        TierBreakdownViewModel? fit = null;
        var fitStatus = "";
        var model = SelectedModel;
        if (model?.LocalPath is { } path && File.Exists(path))
        {
            try
            {
                var catalog = await tiering.GetCatalogAsync(path, ct);
                var usage = catalog.IsMoe ? await tiering.GetUsageAsync(model.Id) : null;
                if (usage is not null && (usage.Layers != catalog.Shape.LayerCount || usage.Experts != catalog.Shape.ExpertCount))
                    usage = null;
                var report = CapacityCalculator.Assess(catalog, Settings, Budget, _hw, usage, model.Quantization);
                fit = TierBreakdownViewModel.From(report);
            }
            catch (Exception ex) when (ex is IOException or InvalidDataException or NotSupportedException)
            {
                fitStatus = $"Can't read {Path.GetFileName(path)}: {ex.Message}";
            }
        }
        else if (model is not null)
        {
            fitStatus = "Model file not found on disk.";
        }
        ct.ThrowIfCancellationRequested();

        Dispatcher.UIThread.Post(() =>
        {
            Rows = new ObservableCollection<CapacityRowView>(rows.Select(r => new CapacityRowView(
                r.Strategy switch
                {
                    CapacityStrategy.GpuOnly => "GPU only",
                    CapacityStrategy.GpuAndRam => "GPU + RAM",
                    _ => "GPU + RAM + disk",
                },
                r.MaxParamsB > 0 ? (r.MaxParamsB >= 4000 ? "≥ 4000B" : $"{r.MaxParamsB:0.#}B") : "—",
                r.TokensPerSecond is { } t ? $"{t:F1} tok/s" : "—",
                r.WeightBytes > 0 ? TierBreakdownViewModel.Gb(r.WeightBytes) : "—",
                r.Note)));

            var labels = CurveSizes.Select(s => $"{s:0}B").ToArray();
            SpeedSeries =
            [
                Line("GPU only", curve.Select(c => c.GpuOnly).ToArray(), "#10B981"),
                Line("GPU + RAM", curve.Select(c => c.GpuAndRam).ToArray(), "#0EA5E9"),
                Line("+ disk", curve.Select(c => c.GpuRamDisk).ToArray(), "#F59E0B"),
            ];
            SizeAxis = [ChartTheme.Axis("model size", labels: labels)];
            var y = ChartTheme.Axis("tok/s (log scale)", v => Math.Pow(10, v) >= 1 ? $"{Math.Pow(10, v):0}" : $"{Math.Pow(10, v):0.#}");
            y.MinLimit = -1;
            y.MinStep = 1;
            SpeedAxis = [y];

            ModelFit = fit;
            ModelFitStatus = fitStatus;
        });
    }

    private static LineSeries<double?> Line(string name, double?[] tps, string color) => new()
    {
        Name = name,
        // Log scale so 0.5 tok/s and 200 tok/s are both readable.
        Values = tps.Select(v => v is > 0 ? Math.Log10(v.Value) : (double?)null).ToArray(),
        Stroke = new SolidColorPaint(SKColor.Parse(color), 2),
        Fill = null,
        GeometrySize = 6,
        GeometryStroke = new SolidColorPaint(SKColor.Parse(color), 2),
        YToolTipLabelFormatter = p => $"{Math.Pow(10, p.Coordinate.PrimaryValue):0.#} tok/s",
    };
}
