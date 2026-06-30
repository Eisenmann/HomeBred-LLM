using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using HomebredLLM.Data;
using HomebredLLM.Models;
using Microsoft.EntityFrameworkCore;

namespace HomebredLLM.ViewModels;

public partial class ModelConfigViewModel(IDbContextFactory<AppDbContext> dbFactory) : ObservableObject
{
    [ObservableProperty] private LocalModel? _model;
    [ObservableProperty] private ModelConfiguration? _config;
    [ObservableProperty] private string _saveStatus = "";

    public async Task LoadAsync(Guid modelId)
    {
        await using var db = await dbFactory.CreateDbContextAsync();
        Model = await db.Models.Include(m => m.Config).FirstOrDefaultAsync(m => m.Id == modelId);
        if (Model is null) return;

        if (Model.Config is null)
        {
            var cfg = new ModelConfiguration { ModelId = modelId };
            db.ModelConfigurations.Add(cfg);
            await db.SaveChangesAsync();
            Model.Config = cfg;
        }
        Config = Model.Config;
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
            existing.UpdatedAt = DateTime.UtcNow;
        }
        await db.SaveChangesAsync();
        SaveStatus = "Saved ✓";
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
        OnPropertyChanged(nameof(Config));
    }
}
