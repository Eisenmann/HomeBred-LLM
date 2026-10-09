using System.Runtime.InteropServices;
using HomebredLLM.Models;

namespace HomebredLLM.Services.Tiering;

public enum CheckStatus { Ok, Warning, Missing, Info }

public enum GpuBackendKind { None, Cuda, Vulkan, Metal }

public sealed record RequirementItem(string Title, CheckStatus Status, string Detail, string? Advice = null, string? Url = null);

public sealed record GpuRequirementsReport(
    GpuBackendKind Kind,
    IReadOnlyList<RequirementItem> Items,
    bool Ready,
    bool RestartRequired,
    string Summary,
    string NativeDirectory);

/// <summary>
/// Checks what GPU acceleration needs on this machine (driver, llama.cpp GPU build next to the app,
/// CUDA runtime / Vulkan loader) and tells the user, with versions and links, what is missing.
/// GPU support is opt-in: releases bundle only the CPU backend (docs/gpu-setup.md).
/// </summary>
public sealed class GpuRequirementsChecker(GpuMetricsService gpu)
{
    /// <summary>llama.cpp build that matches the bundled LLamaSharp (0.27.0).</summary>
    public const string LlamaCppTag = "b8816";
    public const string ReleaseUrl = "https://github.com/ggml-org/llama.cpp/releases/tag/" + LlamaCppTag;

    // Minimum NVIDIA driver for CUDA 12.x apps via minor-version compatibility.
    private static readonly Version MinDriverWindows = new(527, 41);
    private static readonly Version MinDriverLinux = new(525, 60, 13);

    /// <summary>Where the llama.cpp GPU build must be placed (the folder NativeLibraries looks in).</summary>
    public static string TargetDirectory
    {
        get
        {
            return NativeLibraries.CustomDirectory is { Length: > 0 } loaded ? loaded : NativeLibraries.InstallDirectory;
        }
    }

    public GpuRequirementsReport Check()
    {
        var items = new List<RequirementItem>();
        var dir = TargetDirectory;
        var win = OperatingSystem.IsWindows();
        var linux = OperatingSystem.IsLinux();
        var backendHasGpu = HardwareProbe.GetBackendDevices().Any(d => !d.StartsWith("CPU", StringComparison.OrdinalIgnoreCase));

        if (OperatingSystem.IsMacOS())
        {
            items.Add(new("GPU backend", backendHasGpu ? CheckStatus.Ok : CheckStatus.Info,
                backendHasGpu ? "llama.cpp exposes a Metal device." : "llama.cpp exposes only the CPU device.",
                backendHasGpu ? null : $"Use a macOS llama.cpp {LlamaCppTag} build with Metal and put its libraries into {dir}.", ReleaseUrl));
            return Finish(GpuBackendKind.Metal, items, backendHasGpu, false, dir);
        }

        // 1. Which GPU?
        var info = gpu.GetStaticInfo();
        var nvidia = info is not null;
        var vulkanLoader = HasVulkanLoader();
        var kind = nvidia ? GpuBackendKind.Cuda : vulkanLoader ? GpuBackendKind.Vulkan : GpuBackendKind.None;

        if (nvidia)
            items.Add(new("NVIDIA GPU", CheckStatus.Ok, $"{info!.Name ?? "NVIDIA GPU"} · {info.VramTotalBytes / (double)(1L << 30):F1} GB VRAM · uses the CUDA backend"));
        else if (vulkanLoader)
            items.Add(new("GPU", CheckStatus.Info, "No NVIDIA driver found; a Vulkan loader is present, so AMD/Intel GPUs can use the Vulkan backend."));
        else
            items.Add(new("GPU driver", CheckStatus.Missing,
                "No NVIDIA driver (NVML) and no Vulkan loader found.",
                "Install the current driver for your GPU: NVIDIA (CUDA 12 capable) or AMD/Intel (with Vulkan support).",
                "https://www.nvidia.com/Download/index.aspx"));

        // 2. Driver version (NVIDIA).
        if (nvidia)
        {
            var ver = gpu.GetDriverVersion();
            var min = win ? MinDriverWindows : MinDriverLinux;
            if (ver is null || !Version.TryParse(Normalize(ver), out var have))
                items.Add(new("NVIDIA driver", CheckStatus.Warning, $"Version not readable ({ver ?? "unknown"}).",
                    $"CUDA 12 needs driver {min} or newer; the latest Game Ready / Studio driver is recommended.",
                    "https://www.nvidia.com/Download/index.aspx"));
            else if (have < min)
                items.Add(new("NVIDIA driver", CheckStatus.Missing, $"{ver} is too old for CUDA 12.",
                    $"Update to driver {min} or newer (latest recommended).", "https://www.nvidia.com/Download/index.aspx"));
            else
                items.Add(new("NVIDIA driver", CheckStatus.Ok, $"{ver} (CUDA 12 needs ≥ {min})"));
        }

        // 3. Vulkan loader (only matters for the Vulkan backend).
        if (kind == GpuBackendKind.Vulkan)
            items.Add(new("Vulkan loader", CheckStatus.Ok, win ? "vulkan-1.dll found" : "libvulkan.so.1 found"));
        else if (kind == GpuBackendKind.None)
            items.Add(new("Vulkan loader", CheckStatus.Missing, win ? "vulkan-1.dll not found." : "libvulkan.so.1 not found.",
                "Installed with current GPU drivers (Linux: package libvulkan1 / vulkan-icd-loader).", "https://vulkan.lunarg.com/sdk/home"));

        // 4. llama.cpp GPU build next to the app.
        var llamaLib = win ? "llama.dll" : "libllama.so";
        var gpuLib = kind switch
        {
            GpuBackendKind.Vulkan => win ? "ggml-vulkan.dll" : "libggml-vulkan.so",
            _ => win ? "ggml-cuda.dll" : "libggml-cuda.so",
        };
        var archive = kind == GpuBackendKind.Vulkan
            ? $"llama-{LlamaCppTag}-bin-win-vulkan-x64.zip"
            : $"llama-{LlamaCppTag}-bin-win-cuda-12.x-x64.zip (+ cudart-llama-bin-win-cuda-12.x-x64.zip)";
        if (!win) archive = $"the Linux {(kind == GpuBackendKind.Vulkan ? "Vulkan" : "CUDA")} archive of llama.cpp {LlamaCppTag}";

        var haveLlama = File.Exists(Path.Combine(dir, llamaLib));
        var haveGpuLib = File.Exists(Path.Combine(dir, gpuLib));
        if (haveLlama && haveGpuLib)
            items.Add(new($"llama.cpp {LlamaCppTag} GPU build", CheckStatus.Ok, $"{llamaLib} and {gpuLib} found in {dir}"));
        else
            items.Add(new($"llama.cpp {LlamaCppTag} GPU build", CheckStatus.Missing,
                $"{(haveLlama ? gpuLib : llamaLib)} not found in {dir}.",
                $"Download {archive} (tag {LlamaCppTag} exactly — other versions may not load) and copy its DLLs/.so files into {dir}.", ReleaseUrl));

        // 5. CUDA runtime libraries.
        if (kind == GpuBackendKind.Cuda)
        {
            string[] libs = win
                ? ["cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"]
                : ["libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12"];
            var missing = libs.Where(l => !FindLibrary(l, dir)).ToList();
            if (missing.Count == 0)
                items.Add(new("CUDA 12 runtime", CheckStatus.Ok, string.Join(", ", libs) + " found"));
            else
                items.Add(new("CUDA 12 runtime", CheckStatus.Missing, "Not found: " + string.Join(", ", missing),
                    win ? $"Copy these DLLs from the cudart-llama-bin-win-cuda-12.x archive of the same release into {dir}, or install the CUDA Toolkit 12.x."
                        : "Install the CUDA Toolkit 12.x runtime libraries (cuda-cudart-12-x, libcublas-12-x).",
                    win ? ReleaseUrl : "https://developer.nvidia.com/cuda-toolkit-archive"));
        }

        // 6. Is the GPU backend actually loaded in this process?
        var filesReady = items.All(i => i.Status != CheckStatus.Missing);
        var restart = filesReady && !backendHasGpu;
        if (backendHasGpu)
            items.Add(new("Loaded backend", CheckStatus.Ok, "llama.cpp in this process exposes: " + string.Join(", ", HardwareProbe.GetBackendDevices())));
        else if (restart)
            items.Add(new("Loaded backend", CheckStatus.Warning, "The CPU backend is still loaded; llama.cpp is loaded once per process.",
                $"Restart HomeBred-LLM so it picks up the files in {dir}. If it still shows CPU only afterwards, the llama.cpp build version or the CUDA runtime does not match ({LlamaCppTag})."));

        return Finish(kind, items, backendHasGpu, restart, dir);
    }

    private static GpuRequirementsReport Finish(GpuBackendKind kind, List<RequirementItem> items, bool backendHasGpu, bool restart, string dir)
    {
        var missing = items.Count(i => i.Status == CheckStatus.Missing);
        var ready = missing == 0 && backendHasGpu;
        var summary = ready ? "GPU mode is ready."
            : restart ? "Almost there: restart the app to load the GPU backend."
            : missing > 0 ? $"GPU mode needs {missing} more thing{(missing == 1 ? "" : "s")} — see below. Until then the app runs on CPU."
            : "GPU backend not active.";
        return new GpuRequirementsReport(kind, items, ready, restart, summary, dir);
    }

    private static string Normalize(string v)
    {
        var parts = v.Split('.').Take(3).ToArray();
        return parts.Length == 1 ? parts[0] + ".0" : string.Join('.', parts);
    }

    private static bool HasVulkanLoader()
    {
        var name = OperatingSystem.IsWindows() ? "vulkan-1.dll" : "libvulkan.so.1";
        if (!NativeLibrary.TryLoad(name, out var h)) return false;
        NativeLibrary.Free(h);
        return true;
    }

    private static bool FindLibrary(string file, string nativeDir)
    {
        if (File.Exists(Path.Combine(nativeDir, file))) return true;
        if (OperatingSystem.IsWindows())
        {
            var dirs = (Environment.GetEnvironmentVariable("PATH") ?? "").Split(';', StringSplitOptions.RemoveEmptyEntries).ToList();
            if (Environment.GetEnvironmentVariable("CUDA_PATH") is { Length: > 0 } cp) dirs.Add(Path.Combine(cp, "bin"));
            return dirs.Any(d => { try { return File.Exists(Path.Combine(d.Trim('"'), file)); } catch { return false; } });
        }
        if (!NativeLibrary.TryLoad(file, out var h)) return false;
        NativeLibrary.Free(h);
        return true;
    }
}
