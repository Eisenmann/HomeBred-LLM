using System.Runtime.InteropServices;
using LLama.Native;

namespace HomebredLLM;

/// <summary>
/// Points LLamaSharp at a locally built llama.cpp (e.g. the expert-cache
/// patched build from native/llama.cpp-hbec) when one is present, instead of
/// the natives shipped in the LLamaSharp backend NuGet package.
/// Looked up in: $HOMEBRED_LLAMA_NATIVE_DIR, then &lt;app&gt;/native/.
/// Must run before anything touches llama.cpp.
/// </summary>
public static class NativeLibraries
{
    public static string? CustomDirectory { get; private set; }

    /// <summary>Per-user folder the in-app GPU setup installs into (writable, survives app updates).</summary>
    public static string UserDirectory => Path.Combine(AppPaths.Base, "native");

    private static string LlamaFile =>
        OperatingSystem.IsWindows() ? "llama.dll" : OperatingSystem.IsMacOS() ? "libllama.dylib" : "libllama.so";

    /// <summary>Folder to install into / look at: env override, a folder that already holds llama, else the per-user folder.</summary>
    public static string InstallDirectory
    {
        get
        {
            var env = Environment.GetEnvironmentVariable("HOMEBRED_LLAMA_NATIVE_DIR");
            if (!string.IsNullOrWhiteSpace(env)) return env;
            var app = Path.Combine(AppContext.BaseDirectory, "native");
            return File.Exists(Path.Combine(app, LlamaFile)) ? app : UserDirectory;
        }
    }

    public static void ConfigureCustomLlama()
    {
        var candidates = new[]
        {
            Environment.GetEnvironmentVariable("HOMEBRED_LLAMA_NATIVE_DIR"),
            Path.Combine(AppContext.BaseDirectory, "native"),
            UserDirectory,
        };

        var (llama, mtmd) = OperatingSystem.IsWindows() ? ("llama.dll", "mtmd.dll")
            : OperatingSystem.IsMacOS() ? ("libllama.dylib", "libmtmd.dylib")
            : ("libllama.so", "libmtmd.so");

        foreach (var dir in candidates)
        {
            if (string.IsNullOrWhiteSpace(dir)) continue;
            var llamaPath = Path.Combine(dir, llama);
            if (!File.Exists(llamaPath)) continue;
            var mtmdPath = Path.Combine(dir, mtmd);

            // Dependent ggml libraries live next to llama; make sure the loader finds them.
            if (OperatingSystem.IsWindows()) SetDllDirectory(dir);

            NativeLibraryConfig.All.WithLibrary(llamaPath, File.Exists(mtmdPath) ? mtmdPath : null);
            CustomDirectory = dir;
            return;
        }
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool SetDllDirectory(string path);
}
