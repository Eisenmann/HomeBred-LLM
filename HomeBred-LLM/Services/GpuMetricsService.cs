using System.Diagnostics;
using System.Runtime.InteropServices;

namespace HomebredLLM.Services;

public record GpuSnapshot(
    float? GpuUtilPct,
    float? GpuMemUsedMb,
    float? GpuMemTotalMb,
    float? GpuTempC,
    float  CpuUtilPct,
    float  RamUsedMb,
    float  RamTotalMb);

/// <summary>
/// Cross-platform GPU and system metrics.
/// NVML loaded dynamically at runtime (works on Windows + Linux with NVIDIA drivers).
/// CPU usage reads /proc/stat on Linux, PerformanceCounter on Windows.
/// Falls back gracefully when NVIDIA or the relevant OS API is absent.
/// </summary>
public sealed class GpuMetricsService : IDisposable
{
    // ── NVML dynamic loading ───────────────────────────────────────────────

    private bool   _nvmlInitialized;
    private IntPtr _nvmlLib;

    private delegate int NvmlInitFunc();
    private delegate int NvmlShutdownFunc();
    private delegate int NvmlGetHandleFunc(uint index, out IntPtr device);
    private delegate int NvmlGetUtilFunc(IntPtr device, out NvmlUtilization util);
    private delegate int NvmlGetMemFunc(IntPtr device, out NvmlMemory mem);
    private delegate int NvmlGetTempFunc(IntPtr device, int sensor, out uint temp);

    private NvmlInitFunc?      _nvmlInit;
    private NvmlShutdownFunc?  _nvmlShutdown;
    private NvmlGetHandleFunc? _nvmlGetHandle;
    private NvmlGetUtilFunc?   _nvmlGetUtil;
    private NvmlGetMemFunc?    _nvmlGetMem;
    private NvmlGetTempFunc?   _nvmlGetTemp;

    // ── CPU sampling ───────────────────────────────────────────────────────

    private IDisposable? _cpuDisposable;
    private Func<float>? _getCpu;
    private long _prevCpuTotal, _prevCpuIdle;

    public GpuMetricsService()
    {
        LoadNvml();
        InitCpu();
    }

    private void LoadNvml()
    {
        string[] candidates = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
            ? ["nvml.dll"]
            : ["libnvidia-ml.so.1", "libnvidia-ml.so"];

        foreach (var lib in candidates)
        {
            if (!NativeLibrary.TryLoad(lib, out _nvmlLib)) continue;
            try
            {
                _nvmlInit      = Bind<NvmlInitFunc>     (_nvmlLib, "nvmlInit_v2");
                _nvmlShutdown  = Bind<NvmlShutdownFunc> (_nvmlLib, "nvmlShutdown");
                _nvmlGetHandle = Bind<NvmlGetHandleFunc>(_nvmlLib, "nvmlDeviceGetHandleByIndex_v2");
                _nvmlGetUtil   = Bind<NvmlGetUtilFunc>  (_nvmlLib, "nvmlDeviceGetUtilizationRates");
                _nvmlGetMem    = Bind<NvmlGetMemFunc>   (_nvmlLib, "nvmlDeviceGetMemoryInfo");
                _nvmlGetTemp   = Bind<NvmlGetTempFunc>  (_nvmlLib, "nvmlDeviceGetTemperature");

                if (_nvmlInit!() == 0) { _nvmlInitialized = true; break; }
            }
            catch { NativeLibrary.Free(_nvmlLib); _nvmlLib = IntPtr.Zero; }
        }
    }

    private static T Bind<T>(IntPtr lib, string name) where T : Delegate =>
        Marshal.GetDelegateForFunctionPointer<T>(NativeLibrary.GetExport(lib, name));

    private void InitCpu()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            try
            {
                var pc = new PerformanceCounter("Processor", "% Processor Time", "_Total");
                pc.NextValue(); // first sample always 0
                _cpuDisposable = pc;
                _getCpu = pc.NextValue;
                return;
            }
            catch { /* fall through */ }
        }

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            ReadLinuxStat(out _prevCpuTotal, out _prevCpuIdle);
            _getCpu = SampleLinuxCpu;
            return;
        }

        _getCpu = () => 0f; // macOS — Metal APIs not implemented
    }

    private float SampleLinuxCpu()
    {
        ReadLinuxStat(out var total, out var idle);
        var dt = total - _prevCpuTotal;
        var di = idle  - _prevCpuIdle;
        _prevCpuTotal = total;
        _prevCpuIdle  = idle;
        return dt > 0 ? (1f - (float)di / dt) * 100f : 0f;
    }

    private static void ReadLinuxStat(out long total, out long idle)
    {
        total = 0; idle = 0;
        try
        {
            var line = File.ReadLines("/proc/stat").First(l => l.StartsWith("cpu "));
            var p = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            // cpu user nice system idle iowait irq softirq
            long user = long.Parse(p[1]), nice = long.Parse(p[2]),
                 sys  = long.Parse(p[3]);
            idle   = long.Parse(p[4]);
            long io = long.Parse(p[5]), irq = long.Parse(p[6]), soft = long.Parse(p[7]);
            total  = user + nice + sys + idle + io + irq + soft;
        }
        catch { }
    }

    // ── Public API ──────────────────────────────────────────────────────────

    public GpuSnapshot Sample()
    {
        float? gpuUtil = null, gpuMem = null, gpuTotal = null, gpuTemp = null;

        if (_nvmlInitialized)
        {
            try
            {
                _nvmlGetHandle!(0, out var dev);

                _nvmlGetUtil!(dev, out var util);
                gpuUtil = util.Gpu;

                _nvmlGetMem!(dev, out var mem);
                gpuMem   = mem.Used  / 1024f / 1024f;
                gpuTotal = mem.Total / 1024f / 1024f;

                _nvmlGetTemp!(dev, 0, out uint t);
                gpuTemp = t;
            }
            catch { /* GPU may have been reset */ }
        }

        var gc         = GC.GetGCMemoryInfo();
        var ramTotalMb = (float)(gc.TotalAvailableMemoryBytes / 1024 / 1024);
        var ramUsedMb  = ramTotalMb - (float)(gc.MemoryLoadBytes / 1024 / 1024);

        return new GpuSnapshot(gpuUtil, gpuMem, gpuTotal, gpuTemp,
            _getCpu?.Invoke() ?? 0f,
            Math.Max(0f, ramUsedMb),
            ramTotalMb);
    }

    public void Dispose()
    {
        _cpuDisposable?.Dispose();
        if (_nvmlInitialized) try { _nvmlShutdown?.Invoke(); } catch { }
        if (_nvmlLib != IntPtr.Zero) NativeLibrary.Free(_nvmlLib);
    }

    // ── NVML structs ────────────────────────────────────────────────────────

    [StructLayout(LayoutKind.Sequential)]
    private struct NvmlUtilization { public uint Gpu; public uint Memory; }

    [StructLayout(LayoutKind.Sequential)]
    private struct NvmlMemory { public ulong Total; public ulong Free; public ulong Used; }
}
