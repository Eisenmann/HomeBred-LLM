using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CommunityToolkit.Mvvm.Messaging;
using HomebredLLM.Data;
using HomebredLLM.Models;
using HomebredLLM.Services;
using Microsoft.EntityFrameworkCore;
using System.Collections.ObjectModel;

namespace HomebredLLM.ViewModels;

public partial class ModelLibraryViewModel(
    IDbContextFactory<AppDbContext> dbFactory,
    IInferenceService inference,
    HuggingFaceService hf,
    MetricsCollectorService metricsCollector,
    ModelApiServerService apiServer) : ObservableObject
{
    // ── Local models ────────────────────────────────────────────────────────
    [ObservableProperty] private ObservableCollection<LocalModel> _localModels = [];
    [ObservableProperty] private LocalModel? _selectedModel;
    [ObservableProperty] private string _loadingStatus = "";
    [ObservableProperty] private string _loadErrorDetails = "";
    [ObservableProperty] private bool _hasLoadError;

    // ── HuggingFace search ─────────────────────────────────────────────────
    [ObservableProperty] private string _hfSearchQuery = "";
    [ObservableProperty] private ObservableCollection<HfModelInfo> _hfResults = [];
    [ObservableProperty] private HfModelInfo? _selectedHfModel;
    [ObservableProperty] private ObservableCollection<HfFileInfo> _hfFiles = [];
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
        var files = await hf.ListGgufFilesAsync(model.RepoId);
        HfFiles = new ObservableCollection<HfFileInfo>(files);
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
        var destPath = Path.Combine(AppPaths.ModelsDirectory, Path.GetFileName(file.Filename));

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
            TotalBytes = file.SizeBytes,
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
            await hf.DownloadFileAsync(SelectedHfModel.RepoId, file.Filename, destPath, progress);

            string? mmprojPath = null;
            var mmproj = await hf.FindMmprojFileAsync(SelectedHfModel.RepoId);
            if (mmproj is not null)
            {
                DownloadStatus = "Vision support detected — downloading projector…";
                mmprojPath = Path.Combine(AppPaths.ModelsDirectory, Path.GetFileName(mmproj.Filename));
                await hf.DownloadFileAsync(SelectedHfModel.RepoId, mmproj.Filename, mmprojPath);
            }

            await using var db2 = await dbFactory.CreateDbContextAsync();
            var m = await db2.Models.FindAsync(model.Id);
            if (m != null)
            {
                m.LocalPath = destPath;
                m.FileSizeBytes = new FileInfo(destPath).Length;
                m.MmprojPath = mmprojPath;
                m.Status = ModelStatus.Ready;
                m.UpdatedAt = DateTime.UtcNow;
            }
            var j = await db2.DownloadJobs.FirstOrDefaultAsync(x => x.ModelId == model.Id);
            if (j != null) { j.Status = Models.DownloadStatus.Completed; j.CompletedAt = DateTime.UtcNow; j.ProgressPct = 100; }
            await db2.SaveChangesAsync();

            model.Status = ModelStatus.Ready;
            model.LocalPath = destPath;
            model.MmprojPath = mmprojPath;
            DownloadStatus = mmprojPath is not null
                ? "Download complete! Vision support detected."
                : "Download complete!";
        }
        catch (Exception ex)
        {
            model.Status = ModelStatus.Error;
            DownloadStatus = $"Error: {ex.Message}";
            DownloadErrorDetails = ex.ToString();
            HasDownloadError = true;
            if (File.Exists(destPath))
                try { File.Delete(destPath); } catch { /* ignore locked file */ }
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
        if (model.LocalPath is null || !File.Exists(model.LocalPath))
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

        var progress = new Progress<string>(s => LoadingStatus = s);
        try
        {
            await inference.LoadAsync(model.Id, model.LocalPath, cfg, model.MmprojPath, progress);

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

        if (model.LocalPath is not null && File.Exists(model.LocalPath))
            try { File.Delete(model.LocalPath); } catch { /* ignore locked file */ }

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
