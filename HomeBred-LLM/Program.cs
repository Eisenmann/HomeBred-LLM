using Avalonia;

internal class Program
{
    [STAThread]
    public static void Main(string[] args) =>
        BuildAvaloniaApp().StartWithClassicDesktopLifetime(args);

    public static AppBuilder BuildAvaloniaApp() =>
        AppBuilder.Configure<HomebredLLM.App>()
            .UsePlatformDetect()
            .WithInterFont()
            .LogToTrace();
}
