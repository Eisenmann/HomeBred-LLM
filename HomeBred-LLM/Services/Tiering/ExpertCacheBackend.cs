using System.Diagnostics;
using System.Runtime.InteropServices;

namespace HomebredLLM.Services.Tiering;

/// <summary>
/// Phase 2: per-expert VRAM slots. Needs the patched llama.cpp from
/// <c>native/llama.cpp-hbec</c> (adds the <c>hbec_*</c> C API to llama.dll);
/// with the stock LLamaSharp natives <see cref="IsAvailable"/> is false and
/// the planner falls back to per-layer expert placement.
/// </summary>
public interface IExpertCacheBackend
{
    bool IsAvailable { get; }
    string Description { get; }

    /// <summary>Arms the cache for the next model load in this process.</summary>
    void ConfigureNextLoad(int slotsPerLayer, int device = NativeExpertCacheBackend.DeviceFirstGpu);

    /// <summary>Disarms a pending configuration (e.g. when a load is abandoned).</summary>
    void ClearPendingLoad();

    /// <summary>Handle to the cache of a loaded model; null when the model has no active cache.</summary>
    IExpertCacheHandle? Attach(IntPtr llamaModel);
}

/// <summary>Operations on one loaded model's expert cache. Call between decodes only.</summary>
public interface IExpertCacheHandle
{
    int Layers { get; }
    int SlotsPerLayer { get; }
    long Bytes { get; }
    bool LayerEnabled(int layer);

    /// <summary>Makes <paramref name="experts"/> resident in the layer's slots; returns experts uploaded (&lt;0 = error).</summary>
    int SetResidency(int layer, ReadOnlySpan<int> experts);
    int[] GetResidency(int layer);
    ExpertCacheCounters Drain();
}

public readonly record struct ExpertCacheCounters(long Promotions, long Evictions, long BytesUploaded);

/// <summary>Native implementation over the patched llama.cpp's hbec_* API.</summary>
public sealed unsafe class ExpertCacheHandle : IExpertCacheHandle
{
    private readonly IntPtr _model;

    internal ExpertCacheHandle(IntPtr model, int layers, int slots, long bytes)
    {
        _model = model;
        Layers = layers;
        SlotsPerLayer = slots;
        Bytes = bytes;
    }

    public int Layers { get; }
    public int SlotsPerLayer { get; }
    public long Bytes { get; }

    public bool LayerEnabled(int layer) => HbecNative.LayerEnabled(_model, layer) != 0;

    public int SetResidency(int layer, ReadOnlySpan<int> experts)
    {
        fixed (int* p = experts)
            return HbecNative.SetResidency(_model, layer, p, experts.Length);
    }

    public int[] GetResidency(int layer)
    {
        var buf = new int[SlotsPerLayer];
        fixed (int* p = buf)
        {
            var n = HbecNative.GetResidency(_model, layer, p, buf.Length);
            return n == buf.Length ? buf : buf[..Math.Min(n, buf.Length)];
        }
    }

    public ExpertCacheCounters Drain()
    {
        long pr, ev, by;
        HbecNative.DrainCounters(_model, &pr, &ev, &by);
        return new ExpertCacheCounters(pr, ev, by);
    }
}

/// <summary>Binds the <c>hbec_*</c> exports from the llama library LLamaSharp loaded.</summary>
public sealed class NativeExpertCacheBackend : IExpertCacheBackend
{
    public const int DeviceFirstGpu = -1;
    public const int DeviceCpu = -2; // test mode: slots in host memory

    public bool IsAvailable => HbecNative.IsAvailable;

    public string Description => IsAvailable
        ? $"Per-expert VRAM cache available (hbec v{HbecNative.Version})."
        : "Per-expert VRAM cache not installed (stock llama.cpp natives) — experts are placed per layer block. " +
          "Build native/llama.cpp-hbec to enable it.";

    public void ConfigureNextLoad(int slotsPerLayer, int device = DeviceFirstGpu)
    {
        if (IsAvailable) HbecNative.SetLoadConfig(slotsPerLayer, device);
    }

    public void ClearPendingLoad()
    {
        if (IsAvailable) HbecNative.SetLoadConfig(0, DeviceFirstGpu);
    }

    public IExpertCacheHandle? Attach(IntPtr llamaModel)
    {
        if (!IsAvailable || llamaModel == IntPtr.Zero) return null;
        var layers = HbecNative.ModelLayers(llamaModel);
        if (layers <= 0) return null;
        return new ExpertCacheHandle(llamaModel, layers, HbecNative.ModelSlots(llamaModel), HbecNative.ModelBytes(llamaModel));
    }
}

/// <summary>
/// Function pointers into the patched llama library. Resolved lazily from the
/// module LLamaSharp already loaded (its path depends on CPU features/CUDA),
/// and re-tried until the native library is present.
/// </summary>
internal static unsafe class HbecNative
{
    private static readonly object Gate = new();
    private static bool _resolved;

    private static delegate* unmanaged[Cdecl]<int> _version;
    private static delegate* unmanaged[Cdecl]<int, int, void> _setLoadConfig;
    private static delegate* unmanaged[Cdecl]<IntPtr, int> _modelLayers;
    private static delegate* unmanaged[Cdecl]<IntPtr, int> _modelSlots;
    private static delegate* unmanaged[Cdecl]<IntPtr, long> _modelBytes;
    private static delegate* unmanaged[Cdecl]<IntPtr, int, byte> _layerEnabled;
    private static delegate* unmanaged[Cdecl]<IntPtr, int, int*, int, int> _setResidency;
    private static delegate* unmanaged[Cdecl]<IntPtr, int, int*, int, int> _getResidency;
    private static delegate* unmanaged[Cdecl]<IntPtr, long*, long*, long*, void> _drain;

    public static bool IsAvailable
    {
        get
        {
            Resolve();
            return _resolved;
        }
    }

    public static int Version => IsAvailable ? _version() : 0;

    private static void Resolve()
    {
        if (_resolved) return;
        lock (Gate)
        {
            if (_resolved) return;
            try
            {
                LLama.Native.NativeApi.llama_max_devices(); // make LLamaSharp load its natives first
                var lib = FindLlamaModule();
                if (lib == IntPtr.Zero || !NativeLibrary.TryGetExport(lib, "hbec_version", out var v)) return;
                _version = (delegate* unmanaged[Cdecl]<int>)v;
                _setLoadConfig = (delegate* unmanaged[Cdecl]<int, int, void>)NativeLibrary.GetExport(lib, "hbec_set_load_config");
                _modelLayers = (delegate* unmanaged[Cdecl]<IntPtr, int>)NativeLibrary.GetExport(lib, "hbec_model_layers");
                _modelSlots = (delegate* unmanaged[Cdecl]<IntPtr, int>)NativeLibrary.GetExport(lib, "hbec_model_slots");
                _modelBytes = (delegate* unmanaged[Cdecl]<IntPtr, long>)NativeLibrary.GetExport(lib, "hbec_model_bytes");
                _layerEnabled = (delegate* unmanaged[Cdecl]<IntPtr, int, byte>)NativeLibrary.GetExport(lib, "hbec_layer_enabled");
                _setResidency = (delegate* unmanaged[Cdecl]<IntPtr, int, int*, int, int>)NativeLibrary.GetExport(lib, "hbec_set_residency");
                _getResidency = (delegate* unmanaged[Cdecl]<IntPtr, int, int*, int, int>)NativeLibrary.GetExport(lib, "hbec_get_residency");
                _drain = (delegate* unmanaged[Cdecl]<IntPtr, long*, long*, long*, void>)NativeLibrary.GetExport(lib, "hbec_drain_counters");
                _resolved = true;
            }
            catch
            {
                // Missing native library or exports: the cache is simply unavailable.
            }
        }
    }

    private static IntPtr FindLlamaModule()
    {
        foreach (ProcessModule m in Process.GetCurrentProcess().Modules)
        {
            var file = Path.GetFileName(m.FileName);
            if (file is null) continue;
            if (file.Equals("llama.dll", StringComparison.OrdinalIgnoreCase) ||
                file.StartsWith("libllama.so", StringComparison.Ordinal) ||
                file.Equals("libllama.dylib", StringComparison.Ordinal))
                return NativeLibrary.Load(m.FileName!);
        }
        return IntPtr.Zero;
    }

    public static void SetLoadConfig(int slots, int device) => _setLoadConfig(slots, device);
    public static int ModelLayers(IntPtr model) => _modelLayers(model);
    public static int ModelSlots(IntPtr model) => _modelSlots(model);
    public static long ModelBytes(IntPtr model) => _modelBytes(model);
    public static byte LayerEnabled(IntPtr model, int layer) => _layerEnabled(model, layer);
    public static int SetResidency(IntPtr model, int layer, int* experts, int n) => _setResidency(model, layer, experts, n);
    public static int GetResidency(IntPtr model, int layer, int* outExperts, int capacity) => _getResidency(model, layer, outExperts, capacity);
    public static void DrainCounters(IntPtr model, long* pr, long* ev, long* by) => _drain(model, pr, ev, by);
}
