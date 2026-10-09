using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using HomebredLLM.Data;
using HomebredLLM.Models;
using HomebredLLM.Services;
using HomebredLLM.Services.Tiering;
using Microsoft.EntityFrameworkCore;
using System.Collections.ObjectModel;
using Avalonia.Threading;

namespace HomebredLLM.ViewModels;

public partial class ModelConfigViewModel(
    IDbContextFactory<AppDbContext> dbFactory,
    LoraImportService loraImport,
    TieringCoordinator tiering,
    HardwareProbe hardware) : ObservableObject
{
    // ── Memory tiers (GGUF) — docs/tiered-memory-architecture.md ────────────
    // MemoryProfile is a plain EF entity, so its fields are mirrored here as
    // observable properties; every change re-runs the planner for the preview.
    public IReadOnlyList<KvCacheType> KvOptions { get; } = Enum.GetValues<KvCacheType>();
    public IReadOnlyList<RebalancePolicy> RebalanceOptions { get; } = Enum.GetValues<RebalancePolicy>();

    private MemoryProfile? _memory;
    private bool _loadingMemory;
    private CancellationTokenSource? _previewCts;

    [ObservableProperty] private bool _isGguf;
    [ObservableProperty] private bool _tieredMode = true;
    [ObservableProperty] private bool _autoVram = true;
    [ObservableProperty] private bool _autoRam = true;
    [ObservableProperty] private double _vramBudgetGb;
    [ObservableProperty] private double _ramBudgetGb;
    [ObservableProperty] private double _vramTotalGb = 24;
    [ObservableProperty] private double _ramTotalGb = 32;
    [ObservableProperty] private bool _allowDiskTier = true;
    [ObservableProperty] private bool _lockWarmTier;
    [ObservableProperty] private KvCacheType _kvCacheType = KvCacheType.F16;
    [ObservableProperty] private bool _kvOnGpu = true;
    [ObservableProperty] private bool _flashAttention = true;
    [ObservableProperty] private bool _routingProfilerEnabled = true;
    [ObservableProperty] private int _profilerWindowTokens = 2048;
    [ObservableProperty] private RebalancePolicy _rebalancePolicy = RebalancePolicy.Auto;
    [ObservableProperty] private double _rebalanceThresholdPct = 3;

    [ObservableProperty] private TierBreakdownViewModel? _tierPreview;
    [ObservableProperty] private string _tierStatus = "";
    [ObservableProperty] private string _runtimeStatus = "";
    [ObservableProperty] private string? _rebalanceSuggestion;
    [ObservableProperty] private bool _hasRebalanceSuggestion;
    [ObservableProperty] private string _hardwareNote = "";

    partial void OnTieredModeChanged(bool value) => MemoryChanged();
    partial void OnAutoVramChanged(bool value) => MemoryChanged();
    partial void OnAutoRamChanged(bool value) => MemoryChanged();
    partial void OnVramBudgetGbChanged(double value) => MemoryChanged();
    partial void OnRamBudgetGbChanged(double value) => MemoryChanged();
    partial void OnAllowDiskTierChanged(bool value) => MemoryChanged();
    partial void OnLockWarmTierChanged(bool value) => MemoryChanged();
    partial void OnKvCacheTypeChanged(KvCacheType value) => MemoryChanged();
    partial void OnKvOnGpuChanged(bool value) => MemoryChanged();
    partial void OnFlashAttentionChanged(bool value) => MemoryChanged();
    partial void OnRoutingProfilerEnabledChanged(bool value) => MemoryChanged();
    partial void OnProfilerWindowTokensChanged(int value) => MemoryChanged();
    partial void OnRebalancePolicyChanged(RebalancePolicy value) => MemoryChanged();
    partial void OnRebalanceThresholdPctChanged(double value) => MemoryChanged();

    private void MemoryChanged()
    {
        if (_loadingMemory || _memory is null) return;
        WriteMemoryFields(_memory);
        SchedulePreview();
    }

    private void WriteMemoryFields(MemoryProfile m)
    {
        m.Mode = TieredMode ? TieringMode.Tiered : TieringMode.Simple;
        m.VramBudgetMb = AutoVram ? -1 : (int)Math.Max(0, VramBudgetGb * 1024);
        m.RamBudgetMb = AutoRam ? -1 : (int)Math.Max(0, RamBudgetGb * 1024);
        m.AllowDiskTier = AllowDiskTier;
        m.LockWarmTier = LockWarmTier;
        m.KvCacheType = KvCacheType;
        m.KvOnGpu = KvOnGpu;
        m.FlashAttention = FlashAttention;
        m.RoutingProfilerEnabled = RoutingProfilerEnabled;
        m.ProfilerWindowTokens = Math.Clamp(ProfilerWindowTokens, 256, 32768);
        m.RebalancePolicy = RebalancePolicy;
        m.RebalanceThreshold = (float)Math.Clamp(RebalanceThresholdPct / 100, 0, 1);
    }

    private async Task LoadMemoryAsync(LocalModel model)
    {
        IsGguf = model.Format == ModelFormat.Gguf;
        _memory = await tiering.GetOrCreateMemoryProfileAsync(model.Id);
        var hw = await hardware.GetCurrentSpecAsync();
        VramTotalGb = Math.Max(1, Math.Round(hw.VramTotalBytes / (double)(1L << 30), 1));
        RamTotalGb = Math.Max(1, Math.Round(hw.RamTotalBytes / (double)(1L << 30), 0));
        HardwareNote = hw.HasGpuBackend
            ? $"{hw.GpuName ?? "GPU"}: {VramTotalGb:F1} GB VRAM ({hw.VramFreeBytes / (double)(1L << 30):F1} GB free) · RAM {RamTotalGb:F0} GB ({hw.RamAvailableBytes / (double)(1L << 30):F1} GB available)"
            : $"No GPU device in the llama.cpp backend — CPU only. RAM {RamTotalGb:F0} GB ({hw.RamAvailableBytes / (double)(1L << 30):F1} GB available)";

        _loadingMemory = true;
        try
        {
            TieredMode = _memory.Mode == TieringMode.Tiered;
            AutoVram = _memory.VramBudgetMb < 0;
            AutoRam = _memory.RamBudgetMb < 0;
            VramBudgetGb = AutoVram ? Math.Max(0, Math.Round(VramTotalGb - 1, 1)) : Math.Round(_memory.VramBudgetMb / 1024.0, 1);
            RamBudgetGb = AutoRam ? Math.Max(1, Math.Round(Math.Min(RamTotalGb - 4, RamTotalGb * 0.8))) : Math.Round(_memory.RamBudgetMb / 1024.0, 1);
            AllowDiskTier = _memory.AllowDiskTier;
            LockWarmTier = _memory.LockWarmTier;
            KvCacheType = _memory.KvCacheType;
            KvOnGpu = _memory.KvOnGpu;
            FlashAttention = _memory.FlashAttention;
            RoutingProfilerEnabled = _memory.RoutingProfilerEnabled;
            ProfilerWindowTokens = _memory.ProfilerWindowTokens;
            RebalancePolicy = _memory.RebalancePolicy;
            RebalanceThresholdPct = Math.Round(_memory.RebalanceThreshold * 100, 1);
        }
        finally { _loadingMemory = false; }

        tiering.StateChanged -= OnTieringStateChanged;
        tiering.StateChanged += OnTieringStateChanged;
        RefreshRuntimeStatus();
        SchedulePreview();
    }

    private void OnTieringStateChanged(Guid modelId)
    {
        if (Model?.Id == modelId) Dispatcher.UIThread.Post(RefreshRuntimeStatus);
    }

    private void RefreshRuntimeStatus()
    {
        var snap = Model is null ? null : tiering.GetSnapshot(Model.Id);
        if (snap is null)
        {
            RuntimeStatus = Model?.Status == ModelStatus.Running
                ? "Running with GPU-layer placement (Simple mode or tiering unavailable)."
                : "Not running — the preview shows the plan that will be used at the next start.";
            RebalanceSuggestion = null;
            HasRebalanceSuggestion = false;
            return;
        }
        var warm = snap.Warm is { } w
            ? $"warm tier {TierBreakdownViewModel.Gb(w.PrefetchedBytes)} / {TierBreakdownViewModel.Gb(w.TargetBytes)} prefetched" +
              (w.Locking ? $", {TierBreakdownViewModel.Gb(w.LockedBytes)} locked" : "") +
              (w.LockError is { } e ? $" ({e})" : "") + (w.Busy ? " …" : "")
            : "no warm tier";
        RuntimeStatus = $"Running tiered: est. {snap.CalibratedEstimate.TokensPerSecond:F1} tok/s · {warm} · " +
                        $"expected hit rate {snap.WarmHitRate:P0} · profiler: {snap.ProfilerStatus}" +
                        (snap.TokensProfiled > 0 ? $" ({snap.TokensProfiled:N0} tokens)" : "");
        RebalanceSuggestion = snap.Suggestion;
        HasRebalanceSuggestion = snap.Suggestion is not null && RebalancePolicy == RebalancePolicy.Suggest;
    }

    [RelayCommand]
    private void ApplyRebalance()
    {
        if (Model is null) return;
        tiering.ApplySuggestedRebalance(Model.Id);
        RefreshRuntimeStatus();
    }

    [RelayCommand]
    private void RefreshPreview() => SchedulePreview();

    private void SchedulePreview()
    {
        if (Model is null || Config is null || _memory is null || !IsGguf) return;
        _previewCts?.Cancel();
        var cts = _previewCts = new CancellationTokenSource();
        var model = Model;
        var cfg = Config;
        var mem = new MemoryProfile();
        WriteMemoryFields(mem);
        _ = Task.Run(async () =>
        {
            try
            {
                await Task.Delay(200, cts.Token);
                if (mem.Mode == TieringMode.Simple)
                {
                    Dispatcher.UIThread.Post(() =>
                    {
                        TierPreview = null;
                        TierStatus = "Simple mode: the GPU Layer Count above decides placement; no tier planning.";
                    });
                    return;
                }
                var report = await tiering.PreviewAsync(model, cfg, mem, cts.Token);
                if (cts.IsCancellationRequested) return;
                Dispatcher.UIThread.Post(() =>
                {
                    TierPreview = report is null ? null : TierBreakdownViewModel.From(report);
                    TierStatus = report is null ? "Model file not found — can't plan." : "";
                });
            }
            catch (OperationCanceledException) { }
            catch (Exception ex)
            {
                Dispatcher.UIThread.Post(() => TierStatus = $"Planning failed: {ex.Message}");
            }
        });
    }

    [ObservableProperty] private LocalModel? _model;
    [ObservableProperty] private ModelConfiguration? _config;
    [ObservableProperty] private string _saveStatus = "";

    // LoRA adapters attached to this model. Only supported for GGUF models
    // (applied at load time via LlamaCppInferenceService/LLamaSharp). ONNX
    // Runtime GenAI has no adapter support — they must be merged into the
    // base model before exporting to ONNX; the view blocks the import with
    // an informational message for ONNX models instead of opening a picker.
    [ObservableProperty] private ObservableCollection<LoraAdapterConfig> _adapters = [];
    [ObservableProperty] private string _adapterError = "";

    // ModelConfiguration is a plain EF entity (no INotifyPropertyChanged), so the
    // API-server checkbox is mirrored here to reactively toggle the port field's
    // visibility — writing straight to Config.ApiServerEnabled wouldn't notify the UI.
    [ObservableProperty] private bool _apiServerEnabled;

    partial void OnApiServerEnabledChanged(bool value)
    {
        if (Config is not null) Config.ApiServerEnabled = value;
    }

    public async Task LoadAsync(Guid modelId)
    {
        await using var db = await dbFactory.CreateDbContextAsync();
        Model = await db.Models
            .Include(m => m.Config)
            .Include(m => m.LoraAdapters)
            .FirstOrDefaultAsync(m => m.Id == modelId);
        if (Model is null) return;

        if (Model.Config is null)
        {
            var cfg = new ModelConfiguration { ModelId = modelId };
            db.ModelConfigurations.Add(cfg);
            await db.SaveChangesAsync();
            Model.Config = cfg;
        }
        Config = Model.Config;
        ApiServerEnabled = Config.ApiServerEnabled;
        Adapters = new ObservableCollection<LoraAdapterConfig>(
            Model.LoraAdapters.OrderBy(a => a.CreatedAt));
        await LoadMemoryAsync(Model);
    }

    /// <summary>
    /// Shows an informational message that LoRA adapters are not supported by
    /// the ONNX Runtime GenAI inference engine. Called from the view's adapter
    /// button click handler.
    /// </summary>
    public Task ShowAdapterNotSupportedAsync()
    {
        AdapterError = "LoRA adapters are not supported by ONNX Runtime GenAI. " +
                       "Merge the adapter into the base model before exporting to ONNX.";
        return Task.CompletedTask;
    }

    /// <summary>Imports a picked adapter, persists it, and adds a row. Called from the view's file picker.</summary>
    public async Task AddAdapterAsync(string sourcePath)
    {
        AdapterError = "";
        if (Model is null) return;
        try
        {
            var storedPath = await loraImport.ImportAsync(sourcePath);
            var adapter = new LoraAdapterConfig
            {
                ModelId = Model.Id,
                Name = Path.GetFileNameWithoutExtension(sourcePath),
                FilePath = storedPath,
                Scale = 1.0f,
                Enabled = true,
            };

            await using var db = await dbFactory.CreateDbContextAsync();
            db.LoraAdapters.Add(adapter);
            await db.SaveChangesAsync();

            Adapters.Add(adapter);
        }
        catch (Exception ex)
        {
            AdapterError = ex.Message;
        }
    }

    [RelayCommand]
    private async Task RemoveAdapterAsync(LoraAdapterConfig adapter)
    {
        await using var db = await dbFactory.CreateDbContextAsync();
        var existing = await db.LoraAdapters.FindAsync(adapter.Id);
        if (existing is not null)
        {
            db.LoraAdapters.Remove(existing);
            await db.SaveChangesAsync();
        }
        Adapters.Remove(adapter);
    }

    [RelayCommand]
    private async Task SaveAsync()
    {
        if (Config is null) return;
        await using var db = await dbFactory.CreateDbContextAsync();
        var existing = await db.ModelConfigurations.FindAsync(Config.Id);
        if (existing is null)
        {
            db.ModelConfigurations.Add(Config);
        }
        else
        {
            existing.ContextSize = Config.ContextSize;
            existing.Temperature = Config.Temperature;
            existing.TopP = Config.TopP;
            existing.TopK = Config.TopK;
            existing.RepeatPenalty = Config.RepeatPenalty;
            existing.GpuLayerCount = Config.GpuLayerCount;
            existing.ThreadCount = Config.ThreadCount;
            existing.BatchSize = Config.BatchSize;
            existing.MaxTokens = Config.MaxTokens;
            existing.SystemPrompt = Config.SystemPrompt;
            existing.ApiServerEnabled = Config.ApiServerEnabled;
            existing.ApiPort = Config.ApiPort;
            existing.UpdatedAt = DateTime.UtcNow;
        }

        // Persist inline edits to adapter scale / enabled toggles.
        foreach (var adapter in Adapters)
        {
            var existingAdapter = await db.LoraAdapters.FindAsync(adapter.Id);
            if (existingAdapter is not null)
            {
                existingAdapter.Name = adapter.Name;
                existingAdapter.Scale = adapter.Scale;
                existingAdapter.Enabled = adapter.Enabled;
            }
        }

        await db.SaveChangesAsync();

        if (_memory is not null)
        {
            WriteMemoryFields(_memory);
            await tiering.SaveMemoryProfileAsync(_memory);
        }
        SchedulePreview();

        SaveStatus = Model?.Status == ModelStatus.Running
            ? "Saved ✓ — stop and start the model to apply memory changes"
            : "Saved ✓";
        await Task.Delay(2000);
        SaveStatus = "";
    }

    [RelayCommand]
    private void ResetDefaults()
    {
        if (Config is null) return;
        Config.ContextSize = 4096;
        Config.Temperature = 0.7f;
        Config.TopP = 0.9f;
        Config.TopK = 40;
        Config.RepeatPenalty = 1.1f;
        Config.GpuLayerCount = -1;
        Config.ThreadCount = 4;
        Config.BatchSize = 512;
        Config.MaxTokens = 2048;
        Config.SystemPrompt = "";
        Config.ApiPort = 8080;
        ApiServerEnabled = false;
        OnPropertyChanged(nameof(Config));

        if (_memory is not null)
        {
            _loadingMemory = true;
            try
            {
                var d = new MemoryProfile();
                TieredMode = d.Mode == TieringMode.Tiered;
                AutoVram = AutoRam = true;
                AllowDiskTier = d.AllowDiskTier;
                LockWarmTier = d.LockWarmTier;
                KvCacheType = d.KvCacheType;
                KvOnGpu = d.KvOnGpu;
                FlashAttention = d.FlashAttention;
                RoutingProfilerEnabled = d.RoutingProfilerEnabled;
                ProfilerWindowTokens = d.ProfilerWindowTokens;
                RebalancePolicy = d.RebalancePolicy;
                RebalanceThresholdPct = d.RebalanceThreshold * 100;
            }
            finally { _loadingMemory = false; }
            MemoryChanged();
        }
    }
}
