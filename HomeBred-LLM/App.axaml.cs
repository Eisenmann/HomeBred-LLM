using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;
using HomebredLLM.Data;
using HomebredLLM.Services;
using HomebredLLM.ViewModels;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

namespace HomebredLLM;

public partial class App : Application
{
    public static IHost AppHost { get; private set; } = null!;

    public override void Initialize()
    {
        AvaloniaXamlLoader.Load(this);
    }

    public override void OnFrameworkInitializationCompleted()
    {
        AppHost = Host.CreateDefaultBuilder()
            .ConfigureServices(services =>
            {
                services.AddDbContextFactory<AppDbContext>(opt =>
                    opt.UseSqlite($"Data Source={AppPaths.DatabaseFile}"));

                services.AddSingleton<IInferenceService, LlamaSharpService>();
                services.AddSingleton<HuggingFaceService>();
                services.AddSingleton<GpuMetricsService>();
                services.AddSingleton<AnalyticsRepository>();
                services.AddSingleton<MetricsCollectorService>();

                // VMs are singletons so MainViewModel can hold references to them
                services.AddSingleton<ModelLibraryViewModel>();
                services.AddSingleton<ModelConfigViewModel>();
                services.AddSingleton<ChatViewModel>();
                services.AddSingleton<AnalyticsViewModel>();
                services.AddSingleton<MainViewModel>();
                services.AddSingleton<MainWindow>();
            })
            .Build();

        // Create DB schema (no migration files needed)
        Task.Run(async () =>
        {
            using var scope = AppHost.Services.CreateScope();
            var dbFactory = scope.ServiceProvider.GetRequiredService<IDbContextFactory<AppDbContext>>();
            await using var db = await dbFactory.CreateDbContextAsync();
            await db.Database.EnsureCreatedAsync();
        }).GetAwaiter().GetResult();

        // Pre-initialise ViewModels so the UI is ready immediately on first nav
        Task.WhenAll(
            AppHost.Services.GetRequiredService<ModelLibraryViewModel>().InitializeAsync(),
            AppHost.Services.GetRequiredService<ChatViewModel>().InitializeAsync(),
            AppHost.Services.GetRequiredService<AnalyticsViewModel>().InitializeAsync()
        ).GetAwaiter().GetResult();

        AppHost.Services.GetRequiredService<MetricsCollectorService>().Start();

        if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
        {
            desktop.MainWindow = AppHost.Services.GetRequiredService<MainWindow>();
            desktop.Exit += (_, _) =>
            {
                AppHost.Services.GetRequiredService<MetricsCollectorService>().Stop();
                AppHost.Services.GetRequiredService<IInferenceService>().UnloadAll();
                AppHost.StopAsync().GetAwaiter().GetResult();
                AppHost.Dispose();
            };
        }

        base.OnFrameworkInitializationCompleted();
    }

    public static T GetService<T>() where T : notnull =>
        AppHost.Services.GetRequiredService<T>();
}

public static class AppPaths
{
    private static readonly string _base = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "HomeBred-LLM");

    public static string Base => Directory.CreateDirectory(_base).FullName;
    public static string DatabaseFile => Path.Combine(Base, "homebred.db");
    public static string ModelsDirectory => Directory.CreateDirectory(Path.Combine(Base, "models")).FullName;
}
