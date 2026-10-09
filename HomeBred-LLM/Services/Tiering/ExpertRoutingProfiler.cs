using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using LLama;
using LLama.Abstractions;
using LLama.Extensions;
using LLama.Native;

namespace HomebredLLM.Services.Tiering;

/// <summary>
/// Learns which MoE experts are actually used. Owns a small side context on
/// the loaded weights with llama.cpp's graph-evaluation callback
/// (<c>cb_eval</c>) installed; replaying recent chat text through it lets the
/// callback read the router's selected expert ids (<c>ffn_moe_topk-N</c>, or
/// the first k columns of <c>ffn_moe_argsort-N</c> on older builds) for every
/// layer and feed them to an <see cref="ExpertUsageProfile"/>.
///
/// Why a side context: LLamaSharp builds the chat context itself and doesn't
/// expose <c>cb_eval</c>, and an eval callback forces scheduler splits that
/// would slow normal generation. Routing of a token depends only on its
/// prefix, so replaying the same tokens reproduces the same expert choices
/// (exactly within the window; older context beyond it is approximated).
/// </summary>
public sealed unsafe class ExpertRoutingProfiler : IDisposable
{
    // ── ggml-base exports (resolved from the module llama.cpp already loaded) ──
    private static delegate* unmanaged[Cdecl]<IntPtr, byte*> GgmlGetName;
    private static delegate* unmanaged[Cdecl]<IntPtr, nuint> GgmlNbytes;
    private static delegate* unmanaged[Cdecl]<IntPtr, void*, nuint, nuint, void> GgmlTensorGet;
    private static readonly object ResolveGate = new();
    private static bool _resolved;
    private static string? _unavailableReason = "not resolved yet";

    /// <summary>Why profiling is unavailable (null when available).</summary>
    public static string? UnavailableReason
    {
        get { EnsureResolved(); return _unavailableReason; }
    }

    public static bool IsAvailable => UnavailableReason is null;

    /// <summary>
    /// Resolves the ggml exports lazily (and retries after a failure), because
    /// they only exist once LLamaSharp has loaded its native backend.
    /// </summary>
    private static void EnsureResolved()
    {
        if (_resolved) return;
        lock (ResolveGate)
        {
            if (_resolved) return;
            try
            {
                NativeApi.llama_max_devices(); // forces LLamaSharp to load llama/ggml natives
                var lib = ResolveGgmlBase();
                if (lib == IntPtr.Zero)
                {
                    _unavailableReason = "ggml-base native library not found in the process.";
                    return;
                }
                GgmlGetName = (delegate* unmanaged[Cdecl]<IntPtr, byte*>)NativeLibrary.GetExport(lib, "ggml_get_name");
                GgmlNbytes = (delegate* unmanaged[Cdecl]<IntPtr, nuint>)NativeLibrary.GetExport(lib, "ggml_nbytes");
                GgmlTensorGet = (delegate* unmanaged[Cdecl]<IntPtr, void*, nuint, nuint, void>)NativeLibrary.GetExport(lib, "ggml_backend_tensor_get");
                _unavailableReason = null;
                _resolved = true;
            }
            catch (Exception ex)
            {
                _unavailableReason = $"ggml exports unavailable: {ex.Message}";
            }
        }
    }

    private static IntPtr ResolveGgmlBase()
    {
        // LLamaSharp picks a CPU-feature-specific folder (avx2/avx512/...) at runtime,
        // so the only reliable path is the module the process actually loaded.
        foreach (ProcessModule m in Process.GetCurrentProcess().Modules)
        {
            var file = Path.GetFileName(m.FileName);
            if (file is null) continue;
            if (file.StartsWith("ggml-base", StringComparison.OrdinalIgnoreCase) ||
                file.StartsWith("libggml-base", StringComparison.OrdinalIgnoreCase))
                return NativeLibrary.Load(m.FileName!);
        }
        return NativeLibrary.TryLoad("ggml-base", out var h) ? h : IntPtr.Zero;
    }

    // ── Instance ──────────────────────────────────────────────────────────

    private const int ChunkTokens = 64;
    private readonly SafeLLamaContextHandle _ctx;
    private readonly ExpertUsageProfile _profile;
    private readonly int _expertCount;
    private readonly int _expertUsed;
    private readonly int _window;
    private readonly GCHandle _self;
    private int _tokensInChunk;
    private bool _sawTopK;
    private int[] _scratch = new int[4096];

    // Rows read from ffn_moe_argsort, held until the end of a decode: if the
    // same graph also exposes ffn_moe_topk (newer llama.cpp), they are dropped.
    private readonly List<(int Layer, int[] Ids)> _argsortRows = [];

    public ExpertUsageProfile Profile => _profile;
    public long TokensProfiled { get; private set; }
    public DateTime? LastRunAt { get; private set; }

    public ExpertRoutingProfiler(LLamaWeights weights, IContextParams baseParams, ModelShape shape,
        ExpertUsageProfile profile, int windowTokens)
    {
        if (!IsAvailable) throw new NotSupportedException(UnavailableReason);
        if (!_resolved) throw new NotSupportedException("ggml exports not resolved.");
        if (!shape.IsMoe) throw new ArgumentException("Routing profiling only applies to MoE models.");

        _profile = profile;
        _expertCount = shape.ExpertCount;
        _expertUsed = shape.ExpertUsedCount;
        _window = Math.Clamp(windowTokens, 256, 32768);
        _self = GCHandle.Alloc(this);

        baseParams.ToLlamaContextParams(out var lp);
        lp.n_ctx = (uint)_window;
        lp.n_batch = ChunkTokens;
        lp.n_ubatch = ChunkTokens; // the callback must know how many tokens a graph holds
        lp.n_seq_max = 1;
        lp.offload_kqv = false;    // keep the side context's KV cache out of VRAM
        lp.cb_eval = (IntPtr)(delegate* unmanaged[Cdecl]<IntPtr, byte, IntPtr, byte>)&EvalCallback;
        lp.cb_eval_user_data = GCHandle.ToIntPtr(_self);
        _ctx = SafeLLamaContextHandle.Create(weights.NativeHandle, lp);
    }

    /// <summary>
    /// Replays the last <c>window</c> tokens of <paramref name="text"/> and
    /// records their routing. Call only while the main context is idle.
    /// </summary>
    public int ProfileText(string text, CancellationToken ct = default)
    {
        var tokens = _ctx.Tokenize(text, add_bos: true, special: true, Encoding.UTF8);
        if (tokens.Length == 0) return 0;
        if (tokens.Length > _window) tokens = tokens[^_window..];

        _ctx.MemoryClear(true);
        _sawTopK = false;
        var batch = new LLamaBatch();
        var done = 0;
        for (var start = 0; start < tokens.Length; start += ChunkTokens)
        {
            ct.ThrowIfCancellationRequested();
            batch.Clear();
            var n = Math.Min(ChunkTokens, tokens.Length - start);
            for (var i = 0; i < n; i++)
                batch.Add(tokens[start + i], start + i, LLamaSeqId.Zero, logits: true); // all outputs → last layer sees every token
            _tokensInChunk = n;
            _argsortRows.Clear();
            var ok = _ctx.Decode(batch) == DecodeResult.Ok;
            if (!_sawTopK)
                foreach (var (layer, ids) in _argsortRows) _profile.Record(layer, ids);
            _argsortRows.Clear();
            if (!ok) break;
            done += n;
        }
        TokensProfiled += done;
        LastRunAt = DateTime.UtcNow;
        return done;
    }

    [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
    private static byte EvalCallback(IntPtr tensor, byte ask, IntPtr userData)
    {
        try
        {
            if (userData == IntPtr.Zero) return 0;
            var self = (ExpertRoutingProfiler)GCHandle.FromIntPtr(userData).Target!;
            var name = GgmlGetName(tensor);
            var kind = Match(name, out var layer);
            if (kind == 0) return (byte)(ask != 0 ? 0 : 1);
            if (ask != 0)
                return (byte)(kind == 1 || !self._sawTopK ? 1 : 0); // argsort only until topk is known to exist
            self.Read(tensor, kind, layer);
        }
        catch
        {
            // Never let an exception cross into native code.
        }
        return 1; // ask=false: true means "keep computing"
    }

    /// <summary>1 = ffn_moe_topk-N, 2 = ffn_moe_argsort-N, 0 = other.</summary>
    internal static int Match(byte* name, out int layer)
    {
        layer = -1;
        if (name == null) return 0;
        ReadOnlySpan<byte> topk = "ffn_moe_topk-"u8;
        ReadOnlySpan<byte> argsort = "ffn_moe_argsort-"u8;
        var s = new ReadOnlySpan<byte>(name, 64);
        var end = s.IndexOf((byte)0);
        if (end >= 0) s = s[..end];
        int kind;
        if (s.StartsWith(topk)) { kind = 1; s = s[topk.Length..]; }
        else if (s.StartsWith(argsort)) { kind = 2; s = s[argsort.Length..]; }
        else return 0;
        var v = 0;
        foreach (var c in s)
        {
            if (c < '0' || c > '9') return 0;
            v = v * 10 + (c - '0');
        }
        layer = v;
        return s.Length > 0 ? kind : 0;
    }

    private void Read(IntPtr tensor, int kind, int layer)
    {
        if (kind == 1) _sawTopK = true;
        else if (_sawTopK) return; // prefer topk when the build emits both

        var bytes = (long)GgmlNbytes(tensor);
        var ints = bytes / sizeof(int);
        if (ints <= 0 || ints > 1 << 24) return;

        // Work out the layout from the known token count (all tokens, or only
        // the output token(s) in the pruned last layer).
        int k = _expertUsed, e = _expertCount, rows = 0, stride = 0;
        foreach (var t in new[] { _tokensInChunk, 1 })
        {
            if (t <= 0) continue;
            if (kind == 1 && ints == (long)k * t) { rows = t; stride = k; break; }              // contiguous [k, T]
            if (ints == (long)e * (t - 1) + k) { rows = t; stride = e; break; }                // topk view of argsort
            if (ints == (long)e * t) { rows = t; stride = e; break; }                          // full argsort
        }
        if (rows == 0) return;

        if (_scratch.Length < ints) _scratch = new int[ints];
        fixed (int* dst = _scratch)
            GgmlTensorGet(tensor, dst, 0, (nuint)bytes);

        for (var r = 0; r < rows; r++)
        {
            var ids = _scratch.AsSpan(r * stride, k);
            if (kind == 1) _profile.Record(layer, ids);
            else _argsortRows.Add((layer, ids.ToArray()));
        }
    }

    public void Dispose()
    {
        _ctx.Dispose();
        if (_self.IsAllocated) _self.Free();
    }
}
