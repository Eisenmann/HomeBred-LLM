using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;
using HomebredLLM.Data;
using HomebredLLM.Models;
using LLama.Native;
using Microsoft.EntityFrameworkCore;

namespace HomebredLLM.Services.Tiering;

/// <summary>Result of the disk micro-benchmark.</summary>
public sealed record DiskBenchmark(double SequentialMBs, double RandomMBs, double LatencyMs, bool CacheBypassed);

/// <summary>
/// Discovers and measures what the tier planner needs to know about this
/// machine: GPU (NVML), llama.cpp backend devices, RAM size and bandwidth,
/// and the read speed of the drive models live on. Results are stored in the
/// single <see cref="HardwareProfile"/> row; live values (free VRAM,
/// available RAM) are read on every <see cref="GetCurrentSpecAsync"/>.
/// </summary>
public sealed class HardwareProbe(GpuMetricsService gpu, IDbContextFactory<AppDbContext> dbFactory)
{
    private HardwareProfile? _cached;
    private readonly SemaphoreSlim _gate = new(1, 1);

    /// <summary>Stored profile, or a quick probe (no disk benchmark) on first use.</summary>
    public async Task<HardwareProfile> GetProfileAsync(CancellationToken ct = default)
    {
        if (_cached is not null) return _cached;
        await _gate.WaitAsync(ct);
        try
        {
            if (_cached is not null) return _cached;
            await using var db = await dbFactory.CreateDbContextAsync(ct);
            var stored = await db.HardwareProfiles.AsNoTracking().FirstOrDefaultAsync(ct);
            if (stored is not null)
            {
                RefreshStaticFacts(stored); // GPU/backend can change between runs (driver, csproj backend)
                return _cached = stored;
            }

            var quick = await Task.Run(() => Probe(diskPath: null, ramBenchBytes: 256L << 20, progress: null, ct), ct);
            db.HardwareProfiles.Add(quick);
            await db.SaveChangesAsync(ct);
            return _cached = quick;
        }
        finally { _gate.Release(); }
    }

    /// <summary>Full benchmark (RAM bandwidth + disk read speed of <paramref name="diskPath"/>), persisted.</summary>
    public async Task<HardwareProfile> RunBenchmarkAsync(string? diskPath, IProgress<string>? progress = null,
        CancellationToken ct = default)
    {
        var fresh = await Task.Run(() => Probe(diskPath, 1L << 30, progress, ct), ct);
        await _gate.WaitAsync(ct);
        try
        {
            await using var db = await dbFactory.CreateDbContextAsync(ct);
            var existing = await db.HardwareProfiles.FirstOrDefaultAsync(ct);
            if (existing is not null)
            {
                fresh.SpeedCalibration = existing.SpeedCalibration;
                fresh.ComputeBufferCalibration = existing.ComputeBufferCalibration;
                db.HardwareProfiles.Remove(existing);
                await db.SaveChangesAsync(ct);
            }
            db.HardwareProfiles.Add(fresh);
            await db.SaveChangesAsync(ct);
            return _cached = fresh;
        }
        finally { _gate.Release(); }
    }

    /// <summary>Profile + live free VRAM / available RAM, as estimator input.</summary>
    public async Task<HardwareSpec> GetCurrentSpecAsync(CancellationToken ct = default)
    {
        var p = await GetProfileAsync(ct);
        var info = gpu.GetStaticInfo();
        var (_, ramAvail) = ReadSystemMemory();
        return p.ToSpec(info?.VramFreeBytes ?? p.VramTotalBytes, ramAvail);
    }

    /// <summary>Blends a measured/estimated decode-speed ratio into the stored calibration (EMA).</summary>
    public async Task RecordSpeedSampleAsync(double measuredOverEstimated)
    {
        if (measuredOverEstimated is <= 0.05 or >= 20 || double.IsNaN(measuredOverEstimated)) return;
        var p = await GetProfileAsync();
        await _gate.WaitAsync();
        try
        {
            p.SpeedCalibration = 0.8 * p.SpeedCalibration + 0.2 * (1 / measuredOverEstimated);
            await using var db = await dbFactory.CreateDbContextAsync();
            var row = await db.HardwareProfiles.FirstOrDefaultAsync();
            if (row is null) return;
            row.SpeedCalibration = p.SpeedCalibration;
            await db.SaveChangesAsync();
        }
        finally { _gate.Release(); }
    }

    /// <summary>Blends a measured/estimated compute-buffer ratio (parsed from llama.cpp's load log) into the calibration.</summary>
    public async Task RecordComputeBufferSampleAsync(double measuredOverEstimated)
    {
        if (measuredOverEstimated is <= 0.1 or >= 10 || double.IsNaN(measuredOverEstimated)) return;
        var p = await GetProfileAsync();
        await _gate.WaitAsync();
        try
        {
            // The estimate already includes the previous calibration, so compound it.
            p.ComputeBufferCalibration = Math.Clamp(
                0.7 * p.ComputeBufferCalibration + 0.3 * p.ComputeBufferCalibration * measuredOverEstimated, 0.25, 4);
            await using var db = await dbFactory.CreateDbContextAsync();
            var row = await db.HardwareProfiles.FirstOrDefaultAsync();
            if (row is null) return;
            row.ComputeBufferCalibration = p.ComputeBufferCalibration;
            await db.SaveChangesAsync();
        }
        finally { _gate.Release(); }
    }

    private HardwareProfile Probe(string? diskPath, long ramBenchBytes, IProgress<string>? progress, CancellationToken ct)
    {
        var profile = new HardwareProfile();
        progress?.Report("Detecting GPU and llama.cpp backend…");
        RefreshStaticFacts(profile);

        var (ramTotal, _) = ReadSystemMemory();
        profile.RamTotalBytes = ramTotal;
        profile.CpuCores = Environment.ProcessorCount;

        progress?.Report("Measuring RAM bandwidth…");
        ct.ThrowIfCancellationRequested();
        var benchBytes = Math.Min(ramBenchBytes, Math.Max(64L << 20, ramTotal / 16));
        profile.RamBandwidthGBs = MeasureRamBandwidthGBs(benchBytes);

        var target = ResolveDiskTarget(diskPath);
        if (target is not null)
        {
            progress?.Report($"Measuring disk read speed ({Path.GetFileName(target)})…");
            ct.ThrowIfCancellationRequested();
            try
            {
                var d = MeasureDisk(target, ct);
                profile.DiskPath = target;
                profile.DiskSequentialMBs = d.SequentialMBs;
                profile.DiskRandomMBs = d.RandomMBs;
                profile.DiskLatencyMs = d.LatencyMs;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                progress?.Report($"Disk benchmark skipped: {ex.Message}");
            }
        }

        profile.MeasuredAt = DateTime.UtcNow;
        progress?.Report("Benchmark complete.");
        return profile;
    }

    private void RefreshStaticFacts(HardwareProfile profile)
    {
        var devices = GetBackendDevices();
        profile.BackendDevices = string.Join(", ", devices);
        profile.HasGpuBackend = devices.Any(d => !d.StartsWith("CPU", StringComparison.OrdinalIgnoreCase));

        var info = gpu.GetStaticInfo();
        if (info is not null)
        {
            profile.GpuName = info.Name;
            profile.VramTotalBytes = info.VramTotalBytes;
            if (info.MemoryBandwidthGBs is > 0) profile.VramBandwidthGBs = info.MemoryBandwidthGBs.Value;
            if (info.PcieBandwidthGBs is > 0) profile.PcieBandwidthGBs = info.PcieBandwidthGBs.Value;
        }
        else if (profile.HasGpuBackend)
        {
            profile.GpuName ??= devices.FirstOrDefault(d => !d.StartsWith("CPU", StringComparison.OrdinalIgnoreCase));
        }
    }

    /// <summary>
    /// Buffer-type names of the devices llama.cpp's loaded backend exposes
    /// ("CPU", "CUDA0", "Vulkan0", "Metal"…). These are also the valid
    /// <c>TensorBufferOverride.BufferType</c> values.
    /// </summary>
    public static IReadOnlyList<string> GetBackendDevices()
    {
        var names = new List<string>();
        try
        {
            var count = (long)NativeApi.ggml_backend_dev_count();
            for (long i = 0; i < count; i++)
            {
                var dev = NativeApi.ggml_backend_dev_get((UIntPtr)i);
                if (dev == IntPtr.Zero) continue;
                var buft = NativeApi.ggml_backend_dev_buffer_type(dev);
                if (buft == IntPtr.Zero) continue;
                var name = Marshal.PtrToStringUTF8(NativeApi.ggml_backend_buft_name(buft));
                if (!string.IsNullOrEmpty(name)) names.Add(name);
            }
        }
        catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException or TypeInitializationException)
        {
            // Native library missing: the app can still plan, it just can't use a GPU.
        }
        if (!names.Contains("CPU")) names.Insert(0, "CPU");
        return names;
    }

    /// <summary>Total and currently available physical memory in bytes.</summary>
    public static (long Total, long Available) ReadSystemMemory()
    {
        try
        {
            if (OperatingSystem.IsWindows())
            {
                var st = new MemoryStatusEx { dwLength = (uint)Marshal.SizeOf<MemoryStatusEx>() };
                if (GlobalMemoryStatusEx(ref st)) return ((long)st.ullTotalPhys, (long)st.ullAvailPhys);
            }
            else if (OperatingSystem.IsLinux() && File.Exists("/proc/meminfo"))
            {
                long total = 0, avail = 0;
                foreach (var line in File.ReadLines("/proc/meminfo"))
                {
                    if (line.StartsWith("MemTotal:")) total = ParseKb(line);
                    else if (line.StartsWith("MemAvailable:")) avail = ParseKb(line);
                }
                if (total > 0) return (total, avail > 0 ? avail : total / 2);
            }
        }
        catch { /* fall through to GC numbers */ }

        var gc = GC.GetGCMemoryInfo();
        return (gc.TotalAvailableMemoryBytes, Math.Max(0, gc.TotalAvailableMemoryBytes - gc.MemoryLoadBytes));

        static long ParseKb(string line)
        {
            var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            return parts.Length >= 2 && long.TryParse(parts[1], out var kb) ? kb * 1024 : 0;
        }
    }

    /// <summary>Multi-threaded streaming-read bandwidth over a buffer much larger than CPU caches.</summary>
    public static double MeasureRamBandwidthGBs(long bytes)
    {
        var count = (int)Math.Min(bytes / sizeof(long), Array.MaxLength - 64);
        var data = GC.AllocateUninitializedArray<long>(count);
        Parallel.For(0, Environment.ProcessorCount, i =>
        {
            var chunk = count / Environment.ProcessorCount;
            data.AsSpan(i * chunk, i == Environment.ProcessorCount - 1 ? count - i * chunk : chunk).Fill(i + 1);
        });

        var best = double.MaxValue;
        for (var pass = 0; pass < 4; pass++)
        {
            var sw = Stopwatch.StartNew();
            long sink = 0;
            var threads = Environment.ProcessorCount;
            var chunk = count / threads;
            Parallel.For(0, threads, () => 0L, (i, _, acc) =>
            {
                var span = data.AsSpan(i * chunk, i == threads - 1 ? count - i * chunk : chunk);
                var vecs = MemoryMarshal.Cast<long, Vector<long>>(span);
                var v = Vector<long>.Zero;
                foreach (var x in vecs) v += x;
                return acc + Vector.Sum(v);
            }, acc => Interlocked.Add(ref sink, acc));
            sw.Stop();
            GC.KeepAlive(sink);
            if (pass > 0) best = Math.Min(best, sw.Elapsed.TotalSeconds); // pass 0 warms page tables
        }
        return (long)count * sizeof(long) / best / 1e9;
    }

    private static string? ResolveDiskTarget(string? path)
    {
        if (path is not null && File.Exists(path)) return path;
        // Largest GGUF in the models folder: real, big, and on the drive that matters.
        try
        {
            return Directory.EnumerateFiles(path is not null && Directory.Exists(path) ? path : AppPaths.ModelsDirectory,
                    "*.gguf", SearchOption.AllDirectories)
                .Select(f => new FileInfo(f))
                .Where(f => f.Length > 256L << 20)
                .OrderByDescending(f => f.Length)
                .FirstOrDefault()?.FullName;
        }
        catch { return null; }
    }

    /// <summary>
    /// Reads 256 MB sequentially and 48 × 1 MB at random offsets of a large
    /// file, bypassing the OS cache where the platform allows (Windows
    /// FILE_FLAG_NO_BUFFERING, Linux POSIX_FADV_DONTNEED before reading).
    /// </summary>
    public static unsafe DiskBenchmark MeasureDisk(string file, CancellationToken ct = default)
    {
        const int alignment = 4096;
        const int seqChunk = 4 << 20;
        const int randChunk = 1 << 20;
        var length = new FileInfo(file).Length;
        var seqBytes = Math.Min(256L << 20, length / 2 / alignment * alignment);
        if (seqBytes < seqChunk) throw new IOException("File too small for a disk benchmark.");

        var options = FileOptions.None;
        var bypass = false;
        if (OperatingSystem.IsWindows())
        {
            options = (FileOptions)0x20000000; // FILE_FLAG_NO_BUFFERING
            bypass = true;
        }

        using var handle = File.OpenHandle(file, FileMode.Open, FileAccess.Read, FileShare.ReadWrite, options);
        if (OperatingSystem.IsLinux())
            bypass = TryDropLinuxCache(handle);

        var buffer = (byte*)NativeMemory.AlignedAlloc(seqChunk, alignment);
        try
        {
            var rng = new Random(12345);
            var start = rng.NextInt64(0, (length - seqBytes) / alignment) * alignment;

            var sw = Stopwatch.StartNew();
            long read = 0;
            while (read < seqBytes)
            {
                ct.ThrowIfCancellationRequested();
                var n = RandomAccess.Read(handle, new Span<byte>(buffer, seqChunk), start + read);
                if (n <= 0) break;
                read += n;
            }
            var seq = read / sw.Elapsed.TotalSeconds / 1e6;

            const int reads = 48;
            sw.Restart();
            long randRead = 0;
            for (var i = 0; i < reads; i++)
            {
                ct.ThrowIfCancellationRequested();
                var off = rng.NextInt64(0, (length - randChunk) / alignment) * alignment;
                randRead += RandomAccess.Read(handle, new Span<byte>(buffer, randChunk), off);
            }
            var randSecs = sw.Elapsed.TotalSeconds;
            var rand = randRead / randSecs / 1e6;
            var latency = Math.Max(0.02, (randSecs / reads - randChunk / (seq * 1e6)) * 1000);
            return new DiskBenchmark(seq, rand, latency, bypass);
        }
        finally
        {
            NativeMemory.AlignedFree(buffer);
        }
    }

    private static bool TryDropLinuxCache(Microsoft.Win32.SafeHandles.SafeFileHandle handle)
    {
        try
        {
            const int POSIX_FADV_DONTNEED = 4;
            return posix_fadvise((int)handle.DangerousGetHandle(), 0, 0, POSIX_FADV_DONTNEED) == 0;
        }
        catch { return false; }
    }

    [DllImport("libc", SetLastError = true)]
    private static extern int posix_fadvise(int fd, long offset, long len, int advice);

    [StructLayout(LayoutKind.Sequential)]
    private struct MemoryStatusEx
    {
        public uint dwLength;
        public uint dwMemoryLoad;
        public ulong ullTotalPhys;
        public ulong ullAvailPhys;
        public ulong ullTotalPageFile;
        public ulong ullAvailPageFile;
        public ulong ullTotalVirtual;
        public ulong ullAvailVirtual;
        public ulong ullAvailExtendedVirtual;
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool GlobalMemoryStatusEx(ref MemoryStatusEx buffer);
}
