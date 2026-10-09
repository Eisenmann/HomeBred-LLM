using System.Runtime.InteropServices;

namespace HomebredLLM.Services.Tiering;

/// <summary>
/// Phase 2 seam: per-expert VRAM caching needs a native component that
/// replaces llama.cpp's MUL_MAT_ID with a slot-indirected kernel (see
/// docs/tiered-memory-architecture.md and docs/hb_expertcache.h). The managed
/// side talks to it only through this interface, so the planner, rebalancer
/// and analytics do not change when it lands.
/// </summary>
public interface IExpertCacheBackend
{
    bool IsAvailable { get; }
    string Description { get; }

    /// <summary>Number of expert slots per layer that fit <paramref name="vramBytes"/>.</summary>
    int SlotsFor(long vramBytes, long bytesPerExpert, int layers);

    /// <summary>Seeds residency with the hottest experts (from <see cref="ExpertUsageProfile"/>).</summary>
    void Prefill(IReadOnlyList<(int Layer, int Expert)> hottest);

    /// <summary>Counters since the last call: hits, misses, promotions, evictions, bytes uploaded.</summary>
    ExpertCacheCounters DrainCounters();
}

public readonly record struct ExpertCacheCounters(long Hits, long Misses, long Promotions, long Evictions, long BytesUploaded);

/// <summary>Probes for the optional native library; reports "not available" until it exists.</summary>
public sealed class NativeExpertCacheBackend : IExpertCacheBackend
{
    public const string LibraryName = "hb_expertcache";
    private readonly IntPtr _lib;

    public NativeExpertCacheBackend()
    {
        NativeLibrary.TryLoad(LibraryName, typeof(NativeExpertCacheBackend).Assembly, null, out _lib);
    }

    public bool IsAvailable => _lib != IntPtr.Zero && NativeLibrary.TryGetExport(_lib, "hbec_version", out _);

    public string Description => IsAvailable
        ? "Native per-expert VRAM cache loaded."
        : "Per-expert VRAM caching (Phase 2) is not installed — experts are placed per layer block.";

    public int SlotsFor(long vramBytes, long bytesPerExpert, int layers) =>
        bytesPerExpert <= 0 || layers <= 0 ? 0 : (int)Math.Max(0, vramBytes / bytesPerExpert / layers);

    public void Prefill(IReadOnlyList<(int Layer, int Expert)> hottest) { }

    public ExpertCacheCounters DrainCounters() => default;
}
