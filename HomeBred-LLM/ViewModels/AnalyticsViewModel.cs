using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using HomebredLLM.Data;
using HomebredLLM.Models;
using HomebredLLM.Services;
using LiveChartsCore;
using LiveChartsCore.SkiaSharpView;
using LiveChartsCore.SkiaSharpView.Painting;
using Microsoft.EntityFrameworkCore;
using SkiaSharp;
using System.Collections.ObjectModel;

namespace HomebredLLM.ViewModels;

public partial class AnalyticsViewModel(
    IDbContextFactory<AppDbContext> dbFactory,
    AnalyticsRepository analyticsRepo,
    GpuMetricsService gpuService) : ObservableObject
{
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
        _liveTimer.Elapsed += (_, _) => RefreshLiveMetrics();
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

        var points = await analyticsRepo.QueryAsync(SelectedModel.Id, FromDate, ToDate);
        if (points.Count == 0) return;

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
