using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;

namespace HomebredLLM.Services.Tiering;

public sealed record WarmTierStatus(
    long TargetBytes,
    long PrefetchedBytes,
    long LockedBytes,
    bool Locking,
    string? LockError,
    bool Busy);

/// <summary>
/// The RAM tier. llama.cpp memory-maps CPU-side weights straight from the
/// GGUF, so whatever is in the OS page cache is "in RAM" and the rest is "on
/// disk". This class maps the same file read-only and steers the page cache:
/// it prefetches the warm-tier byte ranges (hot experts, CPU-side always-on
/// weights) and — when enabled — locks them so memory pressure can't evict
/// them. Re-targeting is live: no model reload needed.
/// </summary>
public sealed unsafe class WarmTierManager : IDisposable
{
    private readonly MemoryMappedFile _file;
    private readonly MemoryMappedViewAccessor _view;
    private readonly byte* _base;
    private readonly long _length;
    private readonly int _page = Environment.SystemPageSize;
    private readonly object _gate = new();
    private readonly SemaphoreSlim _runLock = new(1, 1);

    private List<WarmRange> _locked = [];
    private CancellationTokenSource? _work;
    private long _target, _prefetched, _lockedBytes;
    private bool _locking, _busy;
    private string? _lockError;

    public string FilePath { get; }

    public WarmTierManager(string filePath)
    {
        FilePath = filePath;
        _length = new FileInfo(filePath).Length;
        _file = MemoryMappedFile.CreateFromFile(filePath, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        _view = _file.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
        byte* p = null;
        _view.SafeMemoryMappedViewHandle.AcquirePointer(ref p);
        _base = p + _view.PointerOffset;
    }

    public WarmTierStatus Status
    {
        get
        {
            lock (_gate)
                return new WarmTierStatus(_target, Interlocked.Read(ref _prefetched), _lockedBytes, _locking, _lockError, _busy);
        }
    }

    /// <summary>
    /// Makes <paramref name="ranges"/> the warm tier: unlocks the previous set,
    /// prefetches the new one in the background and optionally locks it.
    /// </summary>
    public Task ApplyAsync(IReadOnlyList<WarmRange> ranges, bool lockPages)
    {
        CancellationTokenSource cts;
        lock (_gate)
        {
            _work?.Cancel();
            _work = cts = new CancellationTokenSource();
            _target = ranges.Sum(r => r.Length);
            _locking = lockPages;
            _busy = true;
            Interlocked.Exchange(ref _prefetched, 0);
        }

        var clipped = ranges.Where(r => r.Offset >= 0 && r.Offset < _length)
            .Select(r => new WarmRange(r.Offset, Math.Min(r.Length, _length - r.Offset))).ToList();

        return Task.Factory.StartNew(() => Run(clipped, lockPages, cts), CancellationToken.None,
            TaskCreationOptions.LongRunning, TaskScheduler.Default);
    }

    private void Run(List<WarmRange> ranges, bool lockPages, CancellationTokenSource cts)
    {
        var ct = cts.Token;
        _runLock.Wait();
        try
        {
            if (ct.IsCancellationRequested) return;
            Thread.CurrentThread.Priority = ThreadPriority.BelowNormal;
            UnlockAll();

            // Ask the OS for read-ahead first (cheap, async), then touch every page so
            // the range is actually resident when the first token needs it.
            foreach (var r in ranges) Advise(r);
            foreach (var r in ranges)
            {
                ct.ThrowIfCancellationRequested();
                Touch(r, ct);
            }

            if (lockPages)
            {
                var locked = new List<WarmRange>();
                string? error = null;
                if (OperatingSystem.IsWindows()) GrowWorkingSet(ranges.Sum(r => r.Length));
                foreach (var r in ranges)
                {
                    if (ct.IsCancellationRequested)
                    {
                        // A newer target superseded this one: release what we pinned.
                        foreach (var l in locked) Unlock(l);
                        return;
                    }
                    if (!Lock(r, out error)) break;
                    locked.Add(r);
                }
                lock (_gate)
                {
                    _locked = locked;
                    _lockedBytes = locked.Sum(r => r.Length);
                    _lockError = error;
                }
            }
            else
            {
                lock (_gate) _lockError = null;
            }
        }
        catch (OperationCanceledException) { }
        finally
        {
            lock (_gate)
                if (ReferenceEquals(_work, cts)) _busy = false;
            _runLock.Release();
        }
    }

    private void Touch(WarmRange r, CancellationToken ct)
    {
        var start = r.Offset;
        var end = r.Offset + r.Length;
        long sink = 0;
        long done = 0;
        for (var off = start; off < end; off += _page)
        {
            sink += _base[off];
            if (((off - start) & ((64L << 20) - 1)) == 0)
            {
                ct.ThrowIfCancellationRequested();
                Interlocked.Add(ref _prefetched, off - start - done);
                done = off - start;
            }
        }
        Interlocked.Add(ref _prefetched, r.Length - done);
        GC.KeepAlive(sink);
    }

    private (IntPtr Addr, nuint Len) Aligned(WarmRange r)
    {
        var startOff = r.Offset / _page * _page;
        var endOff = Math.Min(_length, (r.Offset + r.Length + _page - 1) / _page * _page);
        return ((IntPtr)(_base + startOff), (nuint)(endOff - startOff));
    }

    private void Advise(WarmRange r)
    {
        try
        {
            var (addr, len) = Aligned(r);
            if (OperatingSystem.IsWindows())
            {
                var entry = new Win32MemoryRangeEntry { VirtualAddress = addr, NumberOfBytes = len };
                PrefetchVirtualMemory(GetCurrentProcess(), 1, &entry, 0);
            }
            else
            {
                const int MADV_WILLNEED = 3;
                madvise(addr, len, MADV_WILLNEED);
            }
        }
        catch { /* advisory only */ }
    }

    private bool Lock(WarmRange r, out string? error)
    {
        error = null;
        try
        {
            var (addr, len) = Aligned(r);
            if (OperatingSystem.IsWindows())
            {
                if (VirtualLock(addr, len)) return true;
                error = $"VirtualLock failed (Win32 error {Marshal.GetLastWin32Error()}); warm tier is prefetched but not pinned.";
                return false;
            }
            if (OperatingSystem.IsLinux() || OperatingSystem.IsMacOS())
            {
                if (mlock(addr, len) == 0) return true;
                error = $"mlock failed (errno {Marshal.GetLastPInvokeError()}) — raise RLIMIT_MEMLOCK (ulimit -l) to pin the warm tier; it stays prefetched.";
                return false;
            }
            error = "Page locking is not supported on this OS.";
            return false;
        }
        catch (Exception ex)
        {
            error = ex.Message;
            return false;
        }
    }

    private void UnlockAll()
    {
        List<WarmRange> old;
        lock (_gate)
        {
            old = _locked;
            _locked = [];
            _lockedBytes = 0;
        }
        foreach (var r in old) Unlock(r);
    }

    private void Unlock(WarmRange r)
    {
        try
        {
            var (addr, len) = Aligned(r);
            if (OperatingSystem.IsWindows()) VirtualUnlock(addr, len);
            else munlock(addr, len);
        }
        catch { /* best effort */ }
    }

    /// <summary>
    /// Bytes of the given ranges currently resident in RAM. Linux/macOS via
    /// mincore; null on Windows (no cheap equivalent for file-backed pages).
    /// </summary>
    public long? MeasureResidentBytes(IReadOnlyList<WarmRange> ranges)
    {
        if (OperatingSystem.IsWindows()) return null;
        try
        {
            long resident = 0;
            const long chunk = 1L << 30;
            var vec = new byte[chunk / _page + 1];
            foreach (var r in ranges)
            {
                var (addr, len) = Aligned(r);
                for (nuint off = 0; off < len; off += (nuint)chunk)
                {
                    var n = (nuint)Math.Min((long)chunk, (long)(len - off));
                    fixed (byte* v = vec)
                    {
                        if (mincore(addr + (nint)off, n, v) != 0) return null;
                    }
                    var pages = (int)((n + (nuint)_page - 1) / (nuint)_page);
                    for (var i = 0; i < pages; i++)
                        if ((vec[i] & 1) != 0) resident += _page;
                }
            }
            return resident;
        }
        catch { return null; }
    }

    private static void GrowWorkingSet(long extraBytes)
    {
        try
        {
            var proc = GetCurrentProcess();
            if (!GetProcessWorkingSetSize(proc, out var min, out var max)) return;
            var newMin = (nuint)((long)min + extraBytes + (64L << 20));
            var newMax = (nuint)Math.Max((long)max, (long)newMin + (256L << 20));
            SetProcessWorkingSetSize(proc, newMin, newMax);
        }
        catch { /* VirtualLock will report the failure */ }
    }

    public void Dispose()
    {
        lock (_gate) _work?.Cancel();
        // Let an in-flight prefetch/lock pass stop before unmapping; if it is stuck
        // on a slow disk, leak the mapping rather than unmap pages it may still touch.
        if (!_runLock.Wait(TimeSpan.FromSeconds(30))) return;
        UnlockAll();
        _view.SafeMemoryMappedViewHandle.ReleasePointer();
        _view.Dispose();
        _file.Dispose();
    }

    // ── Native ────────────────────────────────────────────────────────────

    [StructLayout(LayoutKind.Sequential)]
    private struct Win32MemoryRangeEntry
    {
        public IntPtr VirtualAddress;
        public nuint NumberOfBytes;
    }

    [DllImport("kernel32.dll")]
    private static extern IntPtr GetCurrentProcess();

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool PrefetchVirtualMemory(IntPtr process, nuint count, Win32MemoryRangeEntry* entries, uint flags);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool VirtualLock(IntPtr address, nuint size);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool VirtualUnlock(IntPtr address, nuint size);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool GetProcessWorkingSetSize(IntPtr process, out nuint min, out nuint max);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool SetProcessWorkingSetSize(IntPtr process, nuint min, nuint max);

    [DllImport("libc", SetLastError = true)]
    private static extern int madvise(IntPtr addr, nuint length, int advice);

    [DllImport("libc", SetLastError = true)]
    private static extern int mlock(IntPtr addr, nuint length);

    [DllImport("libc", SetLastError = true)]
    private static extern int munlock(IntPtr addr, nuint length);

    [DllImport("libc", SetLastError = true)]
    private static extern int mincore(IntPtr addr, nuint length, byte* vec);
}
