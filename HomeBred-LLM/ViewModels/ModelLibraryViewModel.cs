using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CommunityToolkit.Mvvm.Messaging;
using HomebredLLM.Data;
using HomebredLLM.Models;
using HomebredLLM.Services;
using HomebredLLM.Services.Tiering;
using Microsoft.EntityFrameworkCore;
using System.Collections.ObjectModel;

namespace HomebredLLM.ViewModels;

/// <summary>A Hugging Face file row with its hardware-fit badge.</summary>
public sealed record HfFileRow(HfFileInfo File, string? FitLabel, string FitColor, string? FitDetail)
{
    public string Filename => File.Filename;
    public long? SizeBytes => File.SizeBytes;
    public string? Quantization => File.Quantization;
    public bool HasFit => FitLabel is not null;
}

public partial class ModelLibraryViewModel(
    IDbContextFactory<AppDbContext> dbFactory,
    IInferenceService inference,
    HuggingFaceService hf,
    MetricsCollectorService metricsCollector,
    ModelApiServerService apiServer,
    LocalModelImportService localImport,
    TieringCoordinator tiering,
    HardwareProbe hardware) : ObservableObject
{
    // ── Local models ────────────────────────────────────────────────────────
    [ObservableProperty] private ObservableCollection<LocalModel> _localModels = [];
    [ObservableProperty] private LocalModel? _selectedModel;
    [ObservableProperty] private string _loadingStatus = "";
    [ObservableProperty] private string _loadErrorDetails = "";
    [ObservableProperty] private bool _hasLoadError;
    [ObservableProperty] private bool _isImporting;
    [ObservableProperty] private string _importStatus = "";
    [ObservableProperty] private string _importErrorDetails = "";
    [ObservableProperty] private bool _hasImportError;

    // ── HuggingFace search ─────────────────────────────────────────────────
    [ObservableProperty] private string _hfSearchQuery = "";
    [ObservableProperty] private ObservableCollection<HfModelInfo> _hfResults = [];
    [ObservableProperty] private HfModelInfo? _selectedHfModel;
    [ObservableProperty] private ObservableCollection<HfFileRow> _hfFiles = [];
    [ObservableProperty] private bool _isSearching;
    [ObservableProperty] private bool _isDownloading;
    [ObservableProperty] private double _downloadProgress;
    [ObservableProperty] private string _downloadStatus = "";
    [ObservableProperty] private string _downloadErrorDetails = "";
    [ObservableProperty] private bool _hasDownloadError;

    public async Task InitializeAsync()
    {
        await using var db = await dbFactory.CreateDbContextAsync();
        var models = await db.Models.Include(m => m.Config).OrderByDescending(m => m.CreatedAt).ToListAsync();
        LocalModels = new ObservableCollection<LocalModel>(models);
        _ = RefreshFitBadgesAsync();
    }

    /// <summary>
    /// Computes the "fits on GPU / with RAM / needs disk / too large" badge for
    /// every local GGUF model under its own memory settings (same planner as
    /// loading). Runs in the background; rows update as results arrive.
    /// </summary>
    public async Task RefreshFitBadgesAsync()
    {
        var models = LocalModels.Where(m => m.Format == ModelFormat.Gguf && m.LocalPath is not null).ToList();
        foreach (var model in models)
        {
            try
            {
                var report = await Task.Run(async () =>
                {
                    var cfg = model.Config ?? new ModelConfiguration { ModelId = model.Id };
                    var mem = await tiering.GetOrCreateMemoryProfileAsync(model.Id);
                    return await tiering.PreviewAsync(model, cfg, mem);
                });
                if (report is null) continue;
                model.FitLabel = $"{report.VerdictLabel} · ~{report.Plan.Estimate.TokensPerSecond:0.#} tok/s";
                model.FitColor = TierBreakdownViewModel.VerdictColorOf(report.Verdict);
                model.FitDetail = string.Join("\n", new[]
                {
                    $"VRAM {TierBreakdownViewModel.Gb(report.Plan.VramTotalBytes)} · RAM {TierBreakdownViewModel.Gb(report.Plan.RamTotalBytes)} · disk {TierBreakdownViewModel.Gb(report.Plan.ColdBytes)}",
                }.Concat(report.Suggestions));
                UpdateModelInList(model);
            }
            catch
            {
                // A badge is optional — never break the library over it.
            }
        }
    }

    private async Task<IEnumerable<HfFileRow>> WithFitAsync(IEnumerable<HfFileInfo> files)
    {
        HardwareSpec hw;
        try { hw = await hardware.GetCurrentSpecAsync(); }
        catch { return files.Select(f => new HfFileRow(f, null, "#6B7280", null)); }

        var budget = new TierBudget
        {
            VramBytes = Math.Max(0, hw.VramFreeBytes - (768L << 20)),
            RamBytes = Math.Max(1L << 30, hw.RamAvailableBytes - (4L << 30)),
        };
        var settings = new RuntimeSettings { ContextSize = 4096 };
        return files.Select(f =>
        {
            // HF listings here are ONNX exports (loaded fully onto the device by ONNX Runtime GenAI),
            // so only "fits in VRAM" is meaningful for them.
            var r = f.SizeBytes is > 0
                ? CapacityCalculator.AssessFile(f.SizeBytes.Value, f.Quantization, f.Filename, settings, budget, hw)
                : null;
            if (r is null) return new HfFileRow(f, null, "#6B7280", null);
            var fitsGpu = r.Verdict == FitVerdict.FitsGpu;
            return new HfFileRow(f,
                fitsGpu ? "Fits GPU" : hw.HasGpuBackend ? "Exceeds VRAM" : "CPU only",
                fitsGpu ? "#10B981" : "#F59E0B",
                $"~{r.Plan.Catalog.Shape.TotalParameters / 1e9:0.#}B params · needs {TierBreakdownViewModel.Gb(r.Plan.VramTotalBytes + r.Plan.CpuWeightBytes)} at 4k context");
        });
    }

    /// <summary>Imports a locally-picked ONNX model directory and adds it to the library.</summary>
    public async Task ImportLocalModelAsync(string sourcePath)
    {
        if (IsImporting) return;
        IsImporting = true;
        HasImportError = false;
        ImportErrorDetails = "";
        ImportStatus = "Reading model metadata…";

        try
        {
            var model = await localImport.ImportAsync(sourcePath);
            _ = Task.Delay(500).ContinueWith(_ => RefreshFitBadgesAsync());

            if (Dispatcher.UIThread.CheckAccess())
            {
                LocalModels.Insert(0, model);
                if (SelectedModel is null) SelectedModel = model;
                ImportStatus = $"Imported {model.Name} ✓";
            }
            else
            {
                Dispatcher.UIThread.Post(() =>
                {
                    LocalModels.Insert(0, model);
                    if (SelectedModel is null) SelectedModel = model;
                    ImportStatus = $"Imported {model.Name} ✓";
                });
            }
        }
        catch (Exception ex)
        {
            ImportStatus = $"Import failed: {ex.Message}";
            ImportErrorDetails = ex.ToString();
            HasImportError = true;
        }
        finally
        {
            IsImporting = false;
        }
    }

    [RelayCommand]
    private void OpenConfig(LocalModel model) =>
        WeakReferenceMessenger.Default.Send(new OpenConfigMessage(model));

    [RelayCommand]
    private async Task SearchHuggingFaceAsync()
    {
        if (string.IsNullOrWhiteSpace(HfSearchQuery)) return;
        IsSearching = true;
        try
        {
            var results = await hf.SearchModelsAsync(HfSearchQuery);
            HfResults = new ObservableCollection<HfModelInfo>(results);
        }
        finally { IsSearching = false; }
    }

    // Called automatically when the ListBox SelectedItem binding changes —
    // replaces the WPF Interaction.Triggers approach
    partial void OnSelectedHfModelChanged(HfModelInfo? value)
    {
        if (value is not null) _ = SelectHfModelAsync(value);
    }

    [RelayCommand]
    private async Task SelectHfModelAsync(HfModelInfo model)
    {
        var files = await hf.ListOnnxFilesAsync(model.RepoId);
        HfFiles = new ObservableCollection<HfFileRow>(await WithFitAsync(files));
    }

    [RelayCommand]
    private async Task DownloadFileAsync(HfFileInfo file)
    {
        if (SelectedHfModel is null || IsDownloading) return;
        IsDownloading = true;
        DownloadProgress = 0;
        HasDownloadError = false;
        DownloadErrorDetails = "";

        var modelName = $"{SelectedHfModel.ModelName} {file.Quantization}".Trim();
        var destDir = Path.Combine(AppPaths.ModelsDirectory, SelectedHfModel.ModelName);

        // Create DB records
        await using var db = await dbFactory.CreateDbContextAsync();
        var model = new LocalModel
        {
            Name = modelName,
            HfRepoId = SelectedHfModel.RepoId,
            HfFilename = file.Filename,
            FileSizeBytes = file.SizeBytes,
            Quantization = file.Quantization,
            Status = ModelStatus.Downloading,
        };
        var cfg = new ModelConfiguration { ModelId = model.Id };
        var job = new DownloadJob
        {
            ModelId = model.Id,
            HfRepoId = SelectedHfModel.RepoId,
            HfFilename = file.Filename,
            TotalBytes = file.SizeBytes ?? 0,
            Status = Models.DownloadStatus.Downloading,
            StartedAt = DateTime.UtcNow,
        };
        db.Models.Add(model);
        db.ModelConfigurations.Add(cfg);
        db.DownloadJobs.Add(job);
        await db.SaveChangesAsync();

        Dispatcher.UIThread.Post(() => LocalModels.Insert(0, model));

        var progress = new Progress<(long d, long t, double pct)>(p =>
        {
            DownloadProgress = p.pct;
            DownloadStatus = $"{p.d / 1_000_000.0:F0} MB / {p.t / 1_000_000.0:F0} MB";
            job.DownloadedBytes = p.d;
            job.ProgressPct = p.pct;
        });

        try
        {
            // For ONNX models, file.Filename is a directory path (e.g. "onnx/")
            // We need to download all files in that directory.
            Directory.CreateDirectory(destDir);
            await hf.DownloadDirectoryAsync(SelectedHfModel.RepoId, file.Filename, destDir, progress);

            string? mmprojPath = null;
            var mmproj = await hf.FindMmprojFileAsync(SelectedHfModel.RepoId);
            if (mmproj is not null)
            {
                DownloadStatus = "Vision support detected — downloading projector…";
                mmprojPath = Path.Combine(destDir, Path.GetFileName(mmproj.Filename));
                await hf.DownloadFileAsync(SelectedHfModel.RepoId, mmproj.Filename, mmprojPath);
            }

            await using var db2 = await dbFactory.CreateDbContextAsync();
            var m = await db2.Models.FindAsync(model.Id);
            if (m != null)
            {
                m.LocalPath = destDir;
                m.FileSizeBytes = new DirectoryInfo(destDir).EnumerateFiles("*", SearchOption.AllDirectories).Sum(f => f.Length);
                m.MmprojPath = mmprojPath;
                m.Status = ModelStatus.Ready;
                m.UpdatedAt = DateTime.UtcNow;
            }
            var j = await db2.DownloadJobs.FirstOrDefaultAsync(x => x.ModelId == model.Id);
            if (j != null) { j.Status = Models.DownloadStatus.Completed; j.CompletedAt = DateTime.UtcNow; j.ProgressPct = 100; }
            await db2.SaveChangesAsync();

            model.Status = ModelStatus.Ready;
            model.LocalPath = destDir;
            model.MmprojPath = mmprojPath;
            DownloadStatus = mmprojPath is not null
                ? "Download complete! Vision support detected."
                : "Download complete!";
        }
        catch (Exception ex)
        {
            model.Status = ModelStatus.Error;
            try
            {
                // Persist the failure — otherwise the entry stays "Downloading" in the DB after a restart.
                await using var dbErr = await dbFactory.CreateDbContextAsync();
                var mErr = await dbErr.Models.FindAsync(model.Id);
                if (mErr != null) { mErr.Status = ModelStatus.Error; mErr.UpdatedAt = DateTime.UtcNow; }
                var jErr = await dbErr.DownloadJobs.FirstOrDefaultAsync(x => x.ModelId == model.Id);
                if (jErr != null) jErr.Status = Models.DownloadStatus.Failed;
                await dbErr.SaveChangesAsync();
            }
            catch { /* best effort */ }
            DownloadStatus = $"Error: {ex.Message}";
            DownloadErrorDetails = ex.ToString();
            HasDownloadError = true;
            try { if (Directory.Exists(destDir)) Directory.Delete(destDir, true); } catch { /* ignore locked dir */ }
        }
        finally
        {
            IsDownloading = false;
            Dispatcher.UIThread.Post(() =>
            {
                var idx = LocalModels.IndexOf(model);
                if (idx >= 0) { LocalModels.RemoveAt(idx); LocalModels.Insert(idx, model); }
            });
        }
    }

    [RelayCommand]
    private async Task StartModelAsync(LocalModel model)
    {
        // ONNX models live in a directory; GGUF models are a single file.
        if (model.LocalPath is null || !(Directory.Exists(model.LocalPath) || File.Exists(model.LocalPath)))
        {
            LoadingStatus = "Model file not found on disk.";
            return;
        }

        LoadingStatus = "Loading model into memory…";
        HasLoadError = false;
        LoadErrorDetails = "";
        model.Status = ModelStatus.Running;
        UpdateModelInList(model);

        await using var db = await dbFactory.CreateDbContextAsync();
        var cfg = await db.ModelConfigurations.FirstOrDefaultAsync(c => c.ModelId == model.Id)
                  ?? new ModelConfiguration { ModelId = model.Id };

        var adapters = await db.LoraAdapters
            .Where(a => a.ModelId == model.Id && a.Enabled)
            .Select(a => new LoraSpec(a.FilePath, a.Scale))
            .ToListAsync();

        var progress = new Progress<string>(s => LoadingStatus = s);
        try
        {
            await inference.LoadAsync(model.Id, model.LocalPath, cfg, model.MmprojPath, adapters, progress);

            var m = await db.Models.FindAsync(model.Id);
            if (m != null) { m.Status = ModelStatus.Running; m.UpdatedAt = DateTime.UtcNow; }
            await db.SaveChangesAsync();

            if (cfg.ApiServerEnabled)
            {
                try
                {
                    apiServer.Start(model.Id, cfg.ApiPort);
                    LoadingStatus = $"{model.Name} is running. API: {apiServer.EndpointUrl(model.Id)}";
                }
                catch (Exception apiEx)
                {
                    LoadingStatus = $"{model.Name} is running, but the API server failed to start (port {cfg.ApiPort} may be in use).";
                    LoadErrorDetails = apiEx.ToString();
                    HasLoadError = true;
                }
            }
            else
            {
                LoadingStatus = $"{model.Name} is running.";
            }
        }
        catch (Exception ex)
        {
            model.Status = ModelStatus.Error;
            LoadingStatus = $"Failed to load: {ex.Message}";
            LoadErrorDetails = ex.ToString();
            HasLoadError = true;
            UpdateModelInList(model);
        }
    }

    [RelayCommand]
    private async Task StopModelAsync(LocalModel model)
    {
        apiServer.Stop(model.Id);
        inference.Unload(model.Id);
        model.Status = ModelStatus.Ready;
        UpdateModelInList(model);

        await using var db = await dbFactory.CreateDbContextAsync();
        var m = await db.Models.FindAsync(model.Id);
        if (m != null) { m.Status = ModelStatus.Ready; m.UpdatedAt = DateTime.UtcNow; }
        await db.SaveChangesAsync();
        LoadingStatus = $"{model.Name} stopped.";
    }

    [RelayCommand]
    private async Task RemoveModelAsync(LocalModel model)
    {
        apiServer.Stop(model.Id);
        inference.Unload(model.Id);

        if (model.LocalPath is not null && Directory.Exists(model.LocalPath))
            try { Directory.Delete(model.LocalPath, true); } catch { /* ignore locked dir */ }

        await using var db = await dbFactory.CreateDbContextAsync();
        var m = await db.Models.FindAsync(model.Id);
        if (m != null) { db.Models.Remove(m); await db.SaveChangesAsync(); }

        Dispatcher.UIThread.Post(() => LocalModels.Remove(model));
    }

    private void UpdateModelInList(LocalModel model)
    {
        Dispatcher.UIThread.Post(() =>
        {
            var idx = LocalModels.IndexOf(model);
            if (idx >= 0) { LocalModels.RemoveAt(idx); LocalModels.Insert(idx, model); }
        });
    }
}