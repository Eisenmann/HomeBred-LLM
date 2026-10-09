using Avalonia;
using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Logging;
using Avalonia.Threading;
using HomebredLLM;
using HomebredLLM.Data;
using HomebredLLM.Models;
using HomebredLLM.ViewModels;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;

// Isolated app data: never touch the user's real database/models.
var dataDir = Path.Combine(Path.GetTempPath(), "homebred-ui-smoke-" + Guid.NewGuid().ToString("N")[..8]);
Directory.CreateDirectory(dataDir);
Environment.SetEnvironmentVariable("HOMEBRED_DATA_DIR", dataDir);

// Repo root = folder holding HomeBred-LLM.sln (walk up from the build output).
var repoRoot = AppContext.BaseDirectory;
while (repoRoot is not null && !File.Exists(Path.Combine(repoRoot, "HomeBred-LLM.sln"))) repoRoot = Path.GetDirectoryName(repoRoot);
if (repoRoot is null) { Console.Error.WriteLine("Run from inside the repository."); return 2; }
var testData = Path.Combine(repoRoot, "HomeBred-LLM", "TestData");
var moePath = Path.Combine(testData, "test_tiny_moe.gguf");
var densePath = Path.Combine(testData, "test_valid.gguf");
if (!File.Exists(moePath) || !File.Exists(densePath)) { Console.Error.WriteLine($"TestData missing in {testData} (test_tiny_moe.gguf is git-ignored: git add -f it)."); return 2; }

var outDir = args.Length > 0 ? args[0] : Path.Combine(repoRoot, "artifacts", "ui-shots");
Directory.CreateDirectory(outDir);

// Seed the app database with two GGUF models (tiny MoE + small dense test file).
var opts = new DbContextOptionsBuilder<AppDbContext>().UseSqlite($"Data Source={AppPaths.DatabaseFile}").Options;
Guid moeId;
await using (var db = new AppDbContext(opts))
{
    await db.Database.EnsureDeletedAsync();
    await db.Database.EnsureCreatedAsync();
    await db.ReconcileSchemaAsync();
    var moe = new LocalModel { Name = "tiny-moe (test)", Format = ModelFormat.Gguf, LocalPath = moePath,
        Status = ModelStatus.Ready, Quantization = "F32", FileSizeBytes = new FileInfo(moePath).Length };
    var dense = new LocalModel { Name = "test-valid (dense)", Format = ModelFormat.Gguf,
        LocalPath = densePath, Status = ModelStatus.Ready, Quantization = "F16" };
    db.Models.AddRange(moe, dense);
    db.ModelConfigurations.AddRange(new ModelConfiguration { ModelId = moe.Id, ContextSize = 1024, BatchSize = 256 },
                                    new ModelConfiguration { ModelId = dense.Id, ContextSize = 1024 });
    await db.SaveChangesAsync();
    moeId = moe.Id;
}

HomebredLLM.NativeLibraries.ConfigureCustomLlama(); Console.WriteLine("custom natives: " + HomebredLLM.NativeLibraries.CustomDirectory); Console.WriteLine("seeded"); var sink = new Sink();
Logger.Sink = sink;
AppBuilder.Configure<App>()
    .UseSkia()
    .UseHeadless(new AvaloniaHeadlessPlatformOptions { UseHeadlessDrawing = false })
    .WithInterFont()
    .SetupWithoutStarting();

Console.WriteLine("setup done"); var exitCode = 0;
var loop = new CancellationTokenSource();
Dispatcher.UIThread.Post(async () =>
{
  try
  {
    var services = App.AppHost.Services;
    var main = services.GetRequiredService<MainViewModel>();
    var window = new MainWindow(main) { Width = 1400, Height = 2300 };
    window.Show();

    async Task Pump(int ms) => await Task.Delay(ms);
    void Shot(string name)
    {
        var frame = window.CaptureRenderedFrame();
        frame?.Save(Path.Combine(outDir, name + ".png"));
        Console.WriteLine($"shot {name}: {(frame is null ? "none" : "ok")}");
    }

    await Pump(2500); // badges compute in background
    Shot("1-models");

    var lib = services.GetRequiredService<ModelLibraryViewModel>();
    var moeModel = lib.LocalModels.First(m => m.Id == moeId);
    await lib.StartModelCommand.ExecuteAsync(moeModel);
    Console.WriteLine("start: " + lib.LoadingStatus);
    await Pump(500);

    await main.NavigateToCommand.ExecuteAsync("Calculator");
    await Pump(3000);
    Shot("2-calculator");
    var calc = services.GetRequiredService<CalculatorViewModel>();
    calc.ModelKind = "MoE";
    await Pump(2000);
    Shot("3-calculator-moe");

    calc.ComputeModeIndex = 1;
    await Pump(3000);
    Shot("6-gpu-check");
    Console.WriteLine("gpu summary: " + calc.GpuSummary);
    foreach (var r in calc.GpuChecklist) Console.WriteLine($"  {r.Icon} {r.Title}: {r.Detail} | {r.Advice}");
    calc.ComputeModeIndex = 0;
    await Pump(1500);
    Console.WriteLine("cpu summary: " + calc.GpuSummary);

    lib.OpenConfigCommand.Execute(moeModel);
    await Pump(3000);
    Shot("4-config"); Console.WriteLine("expert cache note: " + services.GetRequiredService<ModelConfigViewModel>().ExpertCacheNote); Console.WriteLine("runtime: " + services.GetRequiredService<ModelConfigViewModel>().RuntimeStatus);

    await main.NavigateToCommand.ExecuteAsync("Analytics");
    var an = services.GetRequiredService<AnalyticsViewModel>();
    an.SelectedModel = an.Models.First(m => m.Id == moeId);
    await Pump(3000);
    Shot("5-analytics");

    Console.WriteLine($"binding/other warnings: {sink.Lines.Count}");
    foreach (var l in sink.Lines.Distinct().Take(40)) Console.WriteLine("  " + l);
    services.GetRequiredService<HomebredLLM.Services.IInferenceService>().UnloadAll();
  }
  catch (Exception ex) { Console.WriteLine("TEST ERROR: " + ex); exitCode = 1; }
  finally { loop.Cancel(); }
});
Dispatcher.UIThread.MainLoop(loop.Token);
try { Directory.Delete(dataDir, true); } catch { /* sqlite may still hold the file */ }
return exitCode;


sealed class Sink : ILogSink
{
    public List<string> Lines { get; } = [];
    public bool IsEnabled(LogEventLevel level, string area) => level >= LogEventLevel.Warning && area is LogArea.Binding or LogArea.Property or LogArea.Control or LogArea.Visual or LogArea.Layout;
    public void Log(LogEventLevel level, string area, object? source, string messageTemplate) => Add(area, messageTemplate, []);
    public void Log(LogEventLevel level, string area, object? source, string messageTemplate, params object?[] propertyValues) => Add(area, messageTemplate, propertyValues);
    private void Add(string area, string t, object?[] v) { lock (Lines) Lines.Add($"[{area}] {t} :: {string.Join(" | ", v.Select(x => x?.ToString()))}"); }
}
