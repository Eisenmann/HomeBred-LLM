using Avalonia;

internal class Program
{
    [STAThread]
    public static void Main(string[] args)
    {
        // A locally built llama.cpp (native/llama.cpp-hbec → HomeBred-LLM/native/) takes
        // precedence over the LLamaSharp backend package; must happen before any native call.
        HomebredLLM.NativeLibraries.ConfigureCustomLlama();
        BuildAvaloniaApp().StartWithClassicDesktopLifetime(args);
    }

    public static AppBuilder BuildAvaloniaApp() =>
        AppBuilder.Configure<HomebredLLM.App>()
            .UsePlatformDetect()
            .WithInterFont()
            .LogToTrace();
}
