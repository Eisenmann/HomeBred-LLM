using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;
using HomebredLLM.Data;
using HomebredLLM.Services;
using HomebredLLM.Services.Gguf;
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

                // Two concrete inference engines — one per model format — behind a
                // single router registered as IInferenceService. GGUF (v2/v3) support
                // runs through LlamaCppInferenceService via LLamaSharp; ONNX exports
                // keep using OnnxRuntimeService as before.
                services.AddSingleton<OnnxRuntimeService>();

                // Native llama.cpp log capture — registered once here, before any
                // model can be loaded, so the buffer can catch whatever the native
                // loader prints on a load failure (see HOMEBRED_LLM_NATIVE_LOG_CAPTURE_GUIDE.md).
                // The buffer lives for the app lifetime; llama.cpp's native callback
                // must stay rooted (the buffer holds a strong reference).
                services.AddSingleton<NativeLogBuffer>();

                services.AddSingleton<LlamaCppInferenceService>();
                services.AddSingleton<IInferenceService, InferenceServiceRouter>();

                services.AddSingleton<HuggingFaceService>();
                services.AddSingleton<GpuMetricsService>();
                services.AddSingleton<AnalyticsRepository>();
                services.AddSingleton<MetricsCollectorService>();
                services.AddSingleton<ModelApiServerService>();
                services.AddSingleton<LoraImportService>();
                services.AddSingleton<OnnxModelMetadataReader>();
                services.AddSingleton<GgufMetadataReader>();
                services.AddSingleton<LocalModelImportService>();

                // VMs are singletons so MainViewModel can hold references to them
                services.AddSingleton<ModelLibraryViewModel>();
                services.AddSingleton<ModelConfigViewModel>();
                services.AddSingleton<ChatViewModel>();
                services.AddSingleton<AnalyticsViewModel>();
                services.AddSingleton<MainViewModel>();
                services.AddSingleton<MainWindow>();
            })
            .Build();

        // Install llama.cpp's native log callback before anything can touch the
        // native library. This must happen before the first LLamaWeights.LoadFromFile
        // (or any other native call) so the buffer actually captures a load's
        // diagnostics instead of silently missing them.
        AppHost.Services.GetRequiredService<NativeLogBuffer>().RegisterNativeLogCallback();

        // Create DB schema (no migration files needed) and patch existing databases
        // that predate columns added later — see AppDbContextSchemaReconciler.
        Task.Run(async () =>
        {
            using var scope = AppHost.Services.CreateScope();
            var dbFactory = scope.ServiceProvider.GetRequiredService<IDbContextFactory<AppDbContext>>();
            await using var db = await dbFactory.CreateDbContextAsync();
            await db.Database.EnsureCreatedAsync();
            await db.ReconcileSchemaAsync();
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
                AppHost.Services.GetRequiredService<ModelApiServerService>().StopAll();
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
    public static string AttachmentsDirectory => Directory.CreateDirectory(Path.Combine(Base, "attachments")).FullName;
    public static string AdaptersDirectory => Directory.CreateDirectory(Path.Combine(Base, "adapters")).FullName;
}
