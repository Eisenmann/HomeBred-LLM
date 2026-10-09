using System.Diagnostics;

namespace HomebredLLM.Services.Tiering;

/// <summary>
/// Disk-read and page-fault rates for the analytics page. Linux: this
/// process's storage reads (/proc/self/io read_bytes, which includes mmap
/// page-ins) and major faults (/proc/self/stat). Windows: system-wide
/// PhysicalDisk read bytes/s (page-ins of mapped files don't show up in
/// per-process I/O counters there).
/// </summary>
public sealed class ProcessIoSampler : IDisposable
{
    private long _lastRead = -1, _lastFaults = -1;
    private DateTime _lastAt;
    private readonly PerformanceCounter? _winDisk;

    public ProcessIoSampler()
    {
        if (OperatingSystem.IsWindows())
        {
            try
            {
                _winDisk = new PerformanceCounter("PhysicalDisk", "Disk Read Bytes/sec", "_Total");
                _winDisk.NextValue();
            }
            catch { _winDisk = null; }
        }
    }

    public (float? DiskReadMbps, float? MajorFaultsPerSec) Sample()
    {
        try
        {
            if (_winDisk is not null) return (_winDisk.NextValue() / 1e6f, null);
            if (!OperatingSystem.IsLinux()) return (null, null);

            long read = -1, faults = -1;
            foreach (var line in File.ReadLines("/proc/self/io"))
                if (line.StartsWith("read_bytes:") && long.TryParse(line[11..].Trim(), out var v)) read = v;
            var stat = File.ReadAllText("/proc/self/stat");
            var fields = stat[(stat.LastIndexOf(')') + 2)..].Split(' ');
            if (fields.Length > 9 && long.TryParse(fields[9], out var mf)) faults = mf; // field 12: majflt

            var now = DateTime.UtcNow;
            (float?, float?) result = (null, null);
            if (_lastRead >= 0)
            {
                var secs = Math.Max(0.001, (now - _lastAt).TotalSeconds);
                result = ((float)((read - _lastRead) / secs / 1e6), faults >= 0 ? (float)((faults - _lastFaults) / secs) : null);
            }
            _lastRead = read;
            _lastFaults = faults;
            _lastAt = now;
            return result;
        }
        catch { return (null, null); }
    }

    public void Dispose() => _winDisk?.Dispose();
}
