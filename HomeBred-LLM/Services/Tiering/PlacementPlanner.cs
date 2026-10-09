namespace HomebredLLM.Services.Tiering;

public enum Tier { Vram, Ram, Disk }

/// <summary>A contiguous byte range of the GGUF file that belongs to the warm (RAM) tier.</summary>
public readonly record struct WarmRange(long Offset, long Length);

/// <summary>Modeled time spent per generated token, split by where it is spent.</summary>
public sealed record PerTokenEstimate(double GpuMs, double CpuMs, double DiskMs, double SyncMs)
{
    public double TotalMs => GpuMs + CpuMs + DiskMs + SyncMs;
    public double TokensPerSecond => TotalMs > 0 ? 1000.0 / TotalMs : 0;

    /// <summary>Which part dominates — tells the user which budget to raise.</summary>
    public string Bottleneck
    {
        get
        {
            var max = new[] { (GpuMs, "GPU"), (CpuMs, "CPU/RAM"), (DiskMs, "disk"), (SyncMs, "GPU↔CPU sync") }
                .MaxBy(x => x.Item1);
            return max.Item2;
        }
    }
}

/// <summary>Where every weight lives, what that costs, and whether it fits the budgets.</summary>
public sealed class PlacementPlan
{
    public required TensorCatalog Catalog { get; init; }
    public required RuntimeSettings Settings { get; init; }
    public required TierBudget Budget { get; init; }

    public required IReadOnlyList<CatalogTensor> GpuTensors { get; init; }
    public required IReadOnlyList<CatalogTensor> CpuTensors { get; init; }

    /// <summary>CPU-side tensors that are entirely in the warm tier.</summary>
    public required IReadOnlyList<CatalogTensor> WarmTensors { get; init; }

    /// <summary>(layer, expert) pairs whose slices of every CPU-side expert tensor are warm.</summary>
    public required IReadOnlySet<(int Layer, int Expert)> WarmExperts { get; init; }

    /// <summary>Merged file ranges for the warm tier (empty for synthetic catalogs).</summary>
    public required IReadOnlyList<WarmRange> WarmRanges { get; init; }

    public bool UsesGpu { get; init; }
    public bool KvOnGpu { get; init; }

    public long VramWeightBytes { get; init; }
    public long VramKvBytes { get; init; }
    public long VramComputeBytes { get; init; }
    public long VramOverheadBytes { get; init; }
    public long VramTotalBytes => VramWeightBytes + VramKvBytes + VramComputeBytes + VramOverheadBytes;

    public long CpuWeightBytes { get; init; }
    public long WarmBytes { get; init; }
    public long ColdBytes { get; init; }
    public long RamKvBytes { get; init; }
    public long RamComputeBytes { get; init; }
    public long RamTotalBytes => WarmBytes + RamKvBytes + RamComputeBytes;

    /// <summary>Expected share of CPU-side expert bytes read per token that the warm tier serves.</summary>
    public double WarmHitRate { get; init; } = 1;

    public long GpuBytesPerToken { get; init; }
    public long CpuBytesPerToken { get; init; }
    public long ColdBytesPerToken { get; init; }
    public double ColdReadsPerToken { get; init; }
    public int GraphSplits { get; init; }

    public bool Fits { get; init; }
    public required IReadOnlyList<string> Warnings { get; init; }
    public required PerTokenEstimate Estimate { get; init; }

    public Tier TierOf(CatalogTensor t) =>
        GpuTensors.Contains(t) ? Tier.Vram : WarmTensors.Contains(t) ? Tier.Ram : Tier.Disk;
}

/// <summary>
/// Decides the tier of every weight for given budgets (see
/// docs/tiered-memory-architecture.md, "Placement algorithm"). Pure and
/// deterministic: the same inputs always give the same plan.
/// </summary>
public static class PlacementPlanner
{
    /// <summary>Scratch RAM llama.cpp needs for CPU-side compute when a GPU is also used.</summary>
    private const long CpuScratchCap = 256L << 20;

    public static PlacementPlan Plan(
        TensorCatalog catalog,
        RuntimeSettings settings,
        TierBudget budget,
        HardwareSpec hw,
        ExpertUsageProfile? profile = null)
    {
        var shape = catalog.Shape;
        var warnings = new List<string>();
        var kv = MemoryEstimator.KvCacheBytes(shape, settings);
        var compute = MemoryEstimator.ComputeBufferBytes(shape, settings);

        // ── VRAM fixed costs ──────────────────────────────────────────────
        var useGpu = hw.HasGpuBackend && budget.VramBytes > budget.VramOverheadBytes;
        if (!hw.HasGpuBackend && budget.VramBytes > 0)
            warnings.Add("No GPU device in the llama.cpp backend — everything runs on the CPU. " +
                         "Reference a GPU backend package (e.g. LLamaSharp.Backend.Cuda12) to use VRAM.");

        long vramLeft = useGpu ? budget.VramBytes - budget.VramOverheadBytes : 0;
        if (useGpu && vramLeft < compute)
        {
            warnings.Add("VRAM budget is smaller than the compute buffer — GPU not used.");
            useGpu = false;
            vramLeft = 0;
        }
        long vramCompute = 0, vramKv = 0;
        if (useGpu)
        {
            vramCompute = compute;
            vramLeft -= compute;
        }

        var kvOnGpu = useGpu && settings.KvOnGpu;
        if (kvOnGpu && vramLeft < kv)
        {
            warnings.Add($"KV cache ({Gb(kv)}) does not fit next to the compute buffer in VRAM — kept in RAM. " +
                         "Lower the context size or use a quantized KV cache to keep it on the GPU.");
            kvOnGpu = false;
        }
        if (kvOnGpu)
        {
            vramKv = kv;
            vramLeft -= kv;
        }

        var ramCompute = useGpu ? Math.Min(compute / 4, CpuScratchCap) : compute;
        var ramKv = kvOnGpu ? 0 : kv;

        // ── GPU fill ──────────────────────────────────────────────────────
        var gpu = new HashSet<CatalogTensor>(ReferenceEqualityComparer.Instance);
        if (useGpu)
        {
            bool TryPlace(CatalogTensor t)
            {
                if (t.Bytes > vramLeft) return false;
                gpu.Add(t);
                vramLeft -= t.Bytes;
                return true;
            }

            // Output head and small globals first: read every token, and on the GPU
            // they let the last layers' activations stay there.
            foreach (var t in catalog.Tensors.Where(t => t.Layer < 0 && t.Kind is TensorKind.Output or TensorKind.OutputNorm or TensorKind.Other)
                         .OrderBy(t => t.Kind == TensorKind.Output ? 1 : 0))
                TryPlace(t);

            // Always-on part of each layer, last layer first (same order llama.cpp's
            // n_gpu_layers uses, so the GPU part is one contiguous run next to the head).
            var byLayer = catalog.Tensors.Where(t => t.Layer >= 0)
                .GroupBy(t => t.Layer).OrderByDescending(g => g.Key).ToList();
            var gpuLayers = new HashSet<int>();
            foreach (var g in byLayer)
            {
                var alwaysOn = g.Where(t => t.IsAlwaysOn).ToList();
                var bytes = alwaysOn.Sum(t => t.Bytes);
                if (bytes > vramLeft) break;
                foreach (var t in alwaysOn) TryPlace(t);
                gpuLayers.Add(g.Key);
            }

            // Routed experts of GPU layers: every layer reads exactly n_used experts per
            // token, so each fused block has the same value per byte — fill greedily.
            foreach (var g in byLayer.Where(g => gpuLayers.Contains(g.Key)))
                foreach (var t in g.Where(t => t.Kind == TensorKind.ExpertFfn).OrderByDescending(t => t.Bytes))
                    TryPlace(t);
        }

        var cpu = catalog.Tensors.Where(t => !gpu.Contains(t)).ToList();

        // ── Warm (RAM) fill ───────────────────────────────────────────────
        long ramLeft = budget.RamBytes - ramKv - ramCompute;
        if (ramLeft < 0)
        {
            warnings.Add($"RAM budget does not cover the CPU-side KV cache/compute buffers ({Gb(ramKv + ramCompute)}).");
            ramLeft = 0;
        }

        var warmTensors = new HashSet<CatalogTensor>(ReferenceEqualityComparer.Instance);
        var warmExperts = new HashSet<(int, int)>();
        long warmBytes = 0;

        // 1) CPU-side always-on weights: read every token.
        foreach (var t in cpu.Where(t => t.IsAlwaysOn).OrderBy(t => t.Layer))
        {
            if (t.Bytes > ramLeft) continue;
            warmTensors.Add(t);
            ramLeft -= t.Bytes;
            warmBytes += t.Bytes;
        }

        // 2) Individual experts, hottest per byte first.
        var cpuExpertTensors = cpu.Where(t => t.IsExpert).ToList();
        var expertLayers = cpuExpertTensors.GroupBy(t => t.Layer)
            .ToDictionary(g => g.Key, g => g.ToList());
        var k = Math.Max(1, shape.ExpertUsedCount);
        var candidates = new List<(int Layer, int Expert, long Bytes, double Reads)>();
        foreach (var (layer, tensors) in expertLayers)
        {
            var n = tensors.Max(t => t.ExpertCount);
            var bytesPerExpert = tensors.Sum(t => t.BytesPerExpert);
            for (var e = 0; e < n; e++)
            {
                var share = Share(profile, layer, e, n);
                candidates.Add((layer, e, bytesPerExpert, k * share));
            }
        }
        foreach (var c in candidates.OrderByDescending(c => c.Reads / Math.Max(1, c.Bytes)).ThenBy(c => c.Layer).ThenBy(c => c.Expert))
        {
            if (c.Bytes > ramLeft) continue;
            warmExperts.Add((c.Layer, c.Expert));
            ramLeft -= c.Bytes;
            warmBytes += c.Bytes;
        }

        // 3) Token embedding last: only a few rows are read per token.
        foreach (var t in cpu.Where(t => t.Kind == TensorKind.Embedding))
        {
            if (t.Bytes > ramLeft) continue;
            warmTensors.Add(t);
            ramLeft -= t.Bytes;
            warmBytes += t.Bytes;
        }

        // Expert tensors whose every slice is warm count as fully warm tensors
        // (including any bytes not covered by whole slices, e.g. padding).
        foreach (var t in cpuExpertTensors)
            if (Enumerable.Range(0, t.ExpertCount).All(e => warmExperts.Contains((t.Layer, e))))
            {
                warmTensors.Add(t);
                warmBytes += t.Bytes - t.BytesPerExpert * t.ExpertCount;
            }

        var cpuWeightBytes = cpu.Sum(t => t.Bytes);
        var coldBytes = Math.Max(0, cpuWeightBytes - warmBytes);

        // ── Per-token traffic ─────────────────────────────────────────────
        var kvRead = MemoryEstimator.KvReadPerToken(shape, settings);
        long gpuPerToken = gpu.Where(t => t.IsAlwaysOn).Sum(t => t.Bytes)
                           + (long)gpu.Where(t => t.IsExpert).Sum(t => t.Bytes * shape.ExpertReadFraction)
                           + (kvOnGpu ? kvRead : 0);

        var embeddingRow = shape.EmbeddingLength * 2L;
        long cpuPerToken = cpu.Where(t => t.IsAlwaysOn).Sum(t => t.Bytes)
                           + (cpu.Any(t => t.Kind == TensorKind.Embedding) ? embeddingRow : 0)
                           + (kvOnGpu ? 0 : kvRead);
        double warmExpertBytes = 0, cpuExpertBytes = 0, coldExpertBytes = 0, coldReads = 0;
        foreach (var (layer, tensors) in expertLayers)
        {
            var n = tensors.Max(t => t.ExpertCount);
            for (var e = 0; e < n; e++)
            {
                var reads = k * Share(profile, layer, e, n);
                foreach (var t in tensors)
                {
                    var bytes = reads * t.BytesPerExpert;
                    cpuExpertBytes += bytes;
                    if (warmExperts.Contains((layer, e))) warmExpertBytes += bytes;
                    else
                    {
                        coldExpertBytes += bytes;
                        coldReads += reads;
                    }
                }
            }
        }
        cpuPerToken += (long)cpuExpertBytes;

        long coldAlwaysOn = cpu.Where(t => t.IsAlwaysOn && !warmTensors.Contains(t)).Sum(t => t.Bytes);
        var coldPerToken = (long)coldExpertBytes + coldAlwaysOn;
        coldReads += cpu.Count(t => t.IsAlwaysOn && !warmTensors.Contains(t));
        var embeddingCold = cpu.Any(t => t.Kind == TensorKind.Embedding && !warmTensors.Contains(t));
        if (embeddingCold)
        {
            coldPerToken += embeddingRow;
            coldReads += 1;
        }

        var hitRate = cpuExpertBytes > 0 ? warmExpertBytes / cpuExpertBytes : 1.0;
        var splits = CountSplits(catalog, gpu, useGpu);

        // ── Verdict & warnings ────────────────────────────────────────────
        var fits = budget.AllowDisk || coldBytes == 0;
        if (coldBytes > 0 && !budget.AllowDisk)
            warnings.Add($"{Gb(coldBytes)} of weights fit neither VRAM nor the RAM budget and the disk tier is off.");
        if (coldAlwaysOn > 0)
            warnings.Add($"{Gb(coldAlwaysOn)} of weights that are read on every token live on disk — " +
                         "every token streams them from storage. Raise the RAM budget.");
        if (catalog.IsMoe && coldExpertBytes > 0 && hw.DiskLooksRotational)
            warnings.Add("Cold experts are on a drive that benchmarks like an HDD — expect multi-second stalls on misses.");
        if (!catalog.IsMoe && cpu.Any(t => t.IsAlwaysOn) && useGpu)
            warnings.Add("Dense model split between GPU and CPU: speed is limited by RAM bandwidth for the CPU layers.");
        if (settings.KvType != KvCacheType.F16 && settings.KvType != KvCacheType.F32 && !settings.FlashAttention)
            warnings.Add("A quantized V cache needs flash attention; it will be enabled at load.");

        var plan = new PlacementPlan
        {
            Catalog = catalog,
            Settings = settings,
            Budget = budget,
            GpuTensors = gpu.ToList(),
            CpuTensors = cpu,
            WarmTensors = warmTensors.ToList(),
            WarmExperts = warmExperts,
            WarmRanges = catalog.FilePath is null ? [] : BuildRanges(cpu, warmTensors, warmExperts),
            UsesGpu = useGpu && gpu.Count > 0,
            KvOnGpu = kvOnGpu,
            VramWeightBytes = gpu.Sum(t => t.Bytes),
            VramKvBytes = vramKv,
            VramComputeBytes = vramCompute,
            VramOverheadBytes = useGpu ? budget.VramOverheadBytes : 0,
            CpuWeightBytes = cpuWeightBytes,
            WarmBytes = warmBytes,
            ColdBytes = coldBytes,
            RamKvBytes = ramKv,
            RamComputeBytes = ramCompute,
            WarmHitRate = hitRate,
            GpuBytesPerToken = gpuPerToken,
            CpuBytesPerToken = cpuPerToken,
            ColdBytesPerToken = coldPerToken,
            ColdReadsPerToken = coldReads,
            GraphSplits = splits,
            Fits = fits,
            Warnings = warnings,
            Estimate = new PerTokenEstimate(0, 0, 0, 0),
        };
        return plan.WithEstimate(PerformanceEstimator.Estimate(plan, hw));
    }

    /// <summary>
    /// Expected warm-tier hit rate of <paramref name="plan"/>'s warm expert set
    /// under (possibly newer) routing statistics — what the rebalancer compares
    /// against a freshly planned warm set.
    /// </summary>
    public static double ExpectedWarmHitRate(PlacementPlan plan, ExpertUsageProfile? usage)
    {
        var k = Math.Max(1, plan.Catalog.Shape.ExpertUsedCount);
        double warm = 0, all = 0;
        foreach (var t in plan.CpuTensors.Where(t => t.IsExpert))
        {
            for (var e = 0; e < t.ExpertCount; e++)
            {
                var bytes = k * Share(usage, t.Layer, e, t.ExpertCount) * t.BytesPerExpert;
                all += bytes;
                if (plan.WarmExperts.Contains((t.Layer, e))) warm += bytes;
            }
        }
        return all > 0 ? warm / all : 1.0;
    }

    private static double Share(ExpertUsageProfile? p, int layer, int expert, int n) =>
        p is not null && p.Experts == n && p.HasData(layer) ? p.Share(layer, expert) : 1.0 / n;

    /// <summary>Counts device switches along the layer sequence (each costs a sync + activation copy).</summary>
    private static int CountSplits(TensorCatalog catalog, HashSet<CatalogTensor> gpu, bool useGpu)
    {
        if (!useGpu) return 0;
        var splits = 0;
        var onGpu = false; // input embedding is CPU
        foreach (var g in catalog.Tensors.Where(t => t.Layer >= 0).GroupBy(t => t.Layer).OrderBy(g => g.Key))
        {
            var always = g.Where(t => t.IsAlwaysOn).ToList();
            if (always.Count > 0)
            {
                var dev = always.All(gpu.Contains);
                if (dev != onGpu) { splits++; onGpu = dev; }
            }
            var experts = g.Where(t => t.IsExpert).ToList();
            if (experts.Count > 0)
            {
                var anyCpu = experts.Any(t => !gpu.Contains(t));
                var anyGpu = experts.Any(gpu.Contains);
                if (anyCpu && onGpu) splits += 2; // out to CPU and back
                else if (anyGpu && !onGpu) splits += 2;
            }
        }
        var output = catalog.Tensors.FirstOrDefault(t => t.Kind == TensorKind.Output);
        if (output is not null && gpu.Contains(output) != onGpu) splits++;
        return splits;
    }

    private static List<WarmRange> BuildRanges(
        IEnumerable<CatalogTensor> cpu, HashSet<CatalogTensor> warmTensors, HashSet<(int, int)> warmExperts)
    {
        var ranges = new List<WarmRange>();
        foreach (var t in cpu)
        {
            if (warmTensors.Contains(t))
            {
                ranges.Add(new WarmRange(t.FileOffset, t.Bytes));
                continue;
            }
            if (!t.IsExpert) continue;
            for (var e = 0; e < t.ExpertCount; e++)
                if (warmExperts.Contains((t.Layer, e)))
                    ranges.Add(new WarmRange(t.ExpertOffset(e), t.BytesPerExpert));
        }
        return Merge(ranges);
    }

    internal static List<WarmRange> Merge(List<WarmRange> ranges)
    {
        ranges.Sort((a, b) => a.Offset.CompareTo(b.Offset));
        var merged = new List<WarmRange>(ranges.Count);
        foreach (var r in ranges)
        {
            if (merged.Count > 0)
            {
                var last = merged[^1];
                if (r.Offset <= last.Offset + last.Length)
                {
                    var end = Math.Max(last.Offset + last.Length, r.Offset + r.Length);
                    merged[^1] = new WarmRange(last.Offset, end - last.Offset);
                    continue;
                }
            }
            merged.Add(r);
        }
        return merged;
    }

    internal static string Gb(long bytes) => $"{bytes / (double)(1L << 30):F1} GB";
}

internal static class PlacementPlanExtensions
{
    public static PlacementPlan WithEstimate(this PlacementPlan p, PerTokenEstimate e) => new()
    {
        Catalog = p.Catalog,
        Settings = p.Settings,
        Budget = p.Budget,
        GpuTensors = p.GpuTensors,
        CpuTensors = p.CpuTensors,
        WarmTensors = p.WarmTensors,
        WarmExperts = p.WarmExperts,
        WarmRanges = p.WarmRanges,
        UsesGpu = p.UsesGpu,
        KvOnGpu = p.KvOnGpu,
        VramWeightBytes = p.VramWeightBytes,
        VramKvBytes = p.VramKvBytes,
        VramComputeBytes = p.VramComputeBytes,
        VramOverheadBytes = p.VramOverheadBytes,
        CpuWeightBytes = p.CpuWeightBytes,
        WarmBytes = p.WarmBytes,
        ColdBytes = p.ColdBytes,
        RamKvBytes = p.RamKvBytes,
        RamComputeBytes = p.RamComputeBytes,
        WarmHitRate = p.WarmHitRate,
        GpuBytesPerToken = p.GpuBytesPerToken,
        CpuBytesPerToken = p.CpuBytesPerToken,
        ColdBytesPerToken = p.ColdBytesPerToken,
        ColdReadsPerToken = p.ColdReadsPerToken,
        GraphSplits = p.GraphSplits,
        Fits = p.Fits,
        Warnings = p.Warnings,
        Estimate = e,
    };
}
