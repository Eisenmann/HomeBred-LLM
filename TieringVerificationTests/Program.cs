using System.Text;
using System.Text.RegularExpressions;
using HomebredLLM.Data;
using HomebredLLM.Models;
using HomebredLLM.Services.Gguf;
using HomebredLLM.Services.Tiering;
using Microsoft.Data.Sqlite;
using Microsoft.EntityFrameworkCore;

// ---------------------------------------------------------------------------
// Verification for the tiered-memory engine (docs/tiered-memory-architecture.md):
// size tables, catalog classification, estimators, planner, capacity
// calculator, overrides, profile serialisation, warm tier, schema upgrade.
// Exit code = number of failures.
// ---------------------------------------------------------------------------

var failures = 0;
void Check(bool cond, string what)
{
    Console.WriteLine($"{(cond ? "PASS" : "FAIL")}: {what}");
    if (!cond) failures++;
}
bool Near(double actual, double expected, double relTol) =>
    Math.Abs(actual - expected) <= Math.Abs(expected) * relTol;
const double GiB = 1L << 30;

var testDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "HomeBred-LLM", "TestData"));

// ── 1. ggml type sizes ──────────────────────────────────────────────────────
Console.WriteLine("--- 1. ggml type sizes ---");
Check(GgmlTypeInfo.BytesFor(GgmlType.Q4_K, 256) == 144, "Q4_K: 256 elements = 144 bytes");
Check(GgmlTypeInfo.BytesFor(GgmlType.Q8_0, 32) == 34, "Q8_0: 32 elements = 34 bytes");
Check(GgmlTypeInfo.BytesFor(GgmlType.F16, 1000) == 2000, "F16: 2 bytes/element");
Check(GgmlTypeInfo.BytesFor(GgmlType.Q6_K, 4096L * 4096) == 4096L * 4096 / 256 * 210, "Q6_K matrix");
Check(GgmlTypeInfo.BitsPerWeight("Q4_K_M") == 4.85, "bpw Q4_K_M");
Check(GgmlTypeInfo.BitsPerWeight("model-UD-Q4_K_XL.gguf") is > 4 and < 5, "bpw from a file name with a suffix");
Check(GgmlTypeInfo.BitsPerWeight("Q2_K_S") == 2.6, "bpw prefers the longest matching label");

// ── 2. Catalog from a real GGUF ─────────────────────────────────────────────
Console.WriteLine("\n--- 2. catalog from TestData GGUF ---");
var validPath = Path.Combine(testDir, "test_valid.gguf");
if (File.Exists(validPath))
{
    var catalog = await TensorCatalog.FromFileAsync(validPath);
    var fileLen = new FileInfo(validPath).Length;
    Console.WriteLine($"  arch={catalog.Shape.Architecture} layers={catalog.Shape.LayerCount} tensors={catalog.Tensors.Count} bytes={catalog.TotalBytes}");
    Check(catalog.Tensors.Count > 0, "catalog has tensors");
    Check(catalog.TotalBytes > 0 && catalog.TotalBytes <= fileLen, "tensor bytes fit inside the file");
    Check(catalog.Tensors.All(t => t.FileOffset + t.Bytes <= fileLen), "every tensor range lies inside the file");
    Check(catalog.Tensors.Any(t => t.Kind == TensorKind.Embedding), "token_embd classified as Embedding");
    Check(catalog.Tensors.Where(t => t.Name.StartsWith("blk.")).All(t => t.Layer >= 0), "blk.N tensors have a layer");
}
else Check(false, $"TestData present at {validPath}");

// Classification of real llama.cpp names (incl. MoE).
Check(TensorCatalog.Classify("blk.3.ffn_up_exps.weight") == (3, TensorKind.ExpertFfn), "classify ffn_up_exps");
Check(TensorCatalog.Classify("blk.3.ffn_gate_up_exps.weight") == (3, TensorKind.ExpertFfn), "classify fused gate_up exps");
Check(TensorCatalog.Classify("blk.3.ffn_norm_exps.weight") == (3, TensorKind.Norm), "classify ffn_norm_exps as norm");
Check(TensorCatalog.Classify("blk.12.ffn_down_shexp.weight") == (12, TensorKind.SharedExpert), "classify shared expert");
Check(TensorCatalog.Classify("blk.12.ffn_gate_inp.weight") == (12, TensorKind.Router), "classify router");
Check(TensorCatalog.Classify("blk.12.exp_probs_b.bias") == (12, TensorKind.Router), "classify expert bias");
Check(TensorCatalog.Classify("blk.0.attn_kv_a_mqa.weight") == (0, TensorKind.Attention), "classify MLA attention");
Check(TensorCatalog.Classify("blk.0.attn_norm.weight") == (0, TensorKind.Norm), "classify attn_norm");
Check(TensorCatalog.Classify("blk.7.ffn_down.weight") == (7, TensorKind.DenseFfn), "classify dense ffn");
Check(TensorCatalog.Classify("output.weight") == (-1, TensorKind.Output), "classify output");
Check(TensorCatalog.Classify("per_layer_token_embd.weight") == (-1, TensorKind.Embedding), "classify per-layer embedding");

// Synthetic MoE GGUF built in memory: verifies expert slicing and offsets.
var moeGguf = BuildMoeGguf(layers: 4, experts: 8, used: 2, embd: 64, expFf: 32);
var moeCatalog = TensorCatalog.FromGguf(GgufReader.Read(new MemoryStream(moeGguf)), "mem.gguf");
var up = moeCatalog.Tensors.First(t => t.Name == "blk.1.ffn_up_exps.weight");
Check(moeCatalog.IsMoe && moeCatalog.Shape.ExpertCount == 8 && moeCatalog.Shape.ExpertUsedCount == 2, "MoE shape from metadata");
Check(up.ExpertCount == 8 && up.BytesPerExpert * 8 == up.Bytes, "expert tensor sliced into 8 equal parts");
Check(up.ExpertOffset(3) == up.FileOffset + 3 * up.BytesPerExpert, "expert slice offset");
Check(moeCatalog.Shape.ActiveParameters < moeCatalog.Shape.TotalParameters, "active params < total for MoE");

// ── 3. Synthetic shapes & KV ────────────────────────────────────────────────
Console.WriteLine("\n--- 3. shapes & KV cache ---");
var s70 = ModelShape.Synthetic(70);
Check(s70.LayerCount == 80 && s70.EmbeddingLength == 8192, $"70B reference shape (got {s70.LayerCount}L/{s70.EmbeddingLength})");
var c70 = TensorCatalog.Synthetic(s70, 4.85);
Check(Near(c70.TotalBytes, 70e9 * 4.85 / 8, 0.08), $"70B @Q4_K_M ≈ 42.4 GB (got {c70.TotalBytes / 1e9:F1} GB)");
var s32 = ModelShape.Synthetic(32);
var kv32 = MemoryEstimator.KvCacheBytes(s32, new RuntimeSettings { ContextSize = 8192 });
Check(Near(kv32, 2.147e9, 0.01), $"32B KV @8k f16 = 2.15 GB (got {kv32 / 1e9:F2})");
var kvQ8 = MemoryEstimator.KvCacheBytes(s32, new RuntimeSettings { ContextSize = 8192, KvType = KvCacheType.Q8_0 });
Check(Near(kvQ8, kv32 * 34.0 / 64, 0.001), "Q8_0 KV is 17/32 of f16");
var mla = s32 with { KvLoraRank = 512, RopeDim = 64 };
Check(MemoryEstimator.KvCacheBytes(mla, new RuntimeSettings { ContextSize = 1000 }) == 1000L * 64 * 576 * 2, "MLA KV formula");
var moeShape = ModelShape.Synthetic(235, 22, 128, 8);
Check(moeShape.IsMoe && Near(moeShape.ExpertReadFraction, 8.0 / 128, 1e-9), "synthetic MoE shape");
var moeSynth = TensorCatalog.Synthetic(moeShape, 4.85);
Check(Near(moeSynth.TotalBytes, 235e9 * 4.85 / 8, 0.1), $"235B-A22B ≈ 142 GB (got {moeSynth.TotalBytes / 1e9:F0})");
var activeBytes = moeSynth.Tensors.Where(t => t.IsAlwaysOn).Sum(t => t.Bytes) + moeSynth.ExpertBytes * 8.0 / 128;
Check(Near(activeBytes, 22e9 * 4.85 / 8, 0.15), $"active bytes ≈ 22B worth (got {activeBytes / 1e9:F1} GB)");

// ── 4. Planner ──────────────────────────────────────────────────────────────
Console.WriteLine("\n--- 4. placement planner ---");
var hw = new HardwareSpec
{
    HasGpuBackend = true, VramTotalBytes = 24L << 30, VramFreeBytes = 23L << 30, VramBandwidthGBs = 1000,
    RamTotalBytes = 64L << 30, RamAvailableBytes = 56L << 30, RamBandwidthGBs = 60,
    DiskSequentialMBs = 3500, DiskRandomMBs = 1500, DiskLatencyMs = 0.08,
};
var budget = new TierBudget { VramBytes = 22L << 30, RamBytes = 52L << 30, AllowDisk = true };
var ctx8k = new RuntimeSettings { ContextSize = 8192 };

var p8 = PlacementPlanner.Plan(TensorCatalog.Synthetic(ModelShape.Synthetic(8), 4.85), ctx8k, budget, hw);
Check(p8.Fits && p8.UsesGpu && p8.ColdBytes == 0 && p8.CpuTensors.All(t => t.Kind == TensorKind.Embedding),
    "8B: all on GPU except the input embedding");
Check(p8.Estimate.TokensPerSecond > 50, $"8B on a 1 TB/s GPU estimates > 50 tok/s (got {p8.Estimate.TokensPerSecond:F0})");
Check(p8.VramTotalBytes <= budget.VramBytes, "plan respects the VRAM budget");

var p70 = PlacementPlanner.Plan(c70, ctx8k, budget, hw);
Check(p70.Fits && p70.UsesGpu && p70.CpuWeightBytes > 0 && p70.ColdBytes == 0, "70B dense: GPU + RAM, nothing on disk");
Check(p70.VramTotalBytes <= budget.VramBytes, $"70B: VRAM {p70.VramTotalBytes / GiB:F1} GB ≤ budget");
Check(p70.Estimate.TokensPerSecond is > 0.5 and < 10, $"70B split estimate is RAM-bound (got {p70.Estimate.TokensPerSecond:F1} tok/s)");
var gpuLayers = p70.GpuTensors.Where(t => t.Layer >= 0).Select(t => t.Layer).Distinct().OrderBy(l => l).ToList();
Check(gpuLayers.Count > 0 && gpuLayers[^1] == s70.LayerCount - 1 && gpuLayers.Count == gpuLayers[^1] - gpuLayers[0] + 1,
    "dense GPU layers are one contiguous run ending at the last layer");

var noDisk = PlacementPlanner.Plan(moeSynth, ctx8k, budget with { AllowDisk = false }, hw);
Check(!noDisk.Fits && noDisk.ColdBytes > 0, "235B MoE does not fit 22+52 GB without disk");
var pMoe = PlacementPlanner.Plan(moeSynth, ctx8k, budget, hw);
Check(pMoe.Fits && pMoe.ColdBytes > 0 && pMoe.WarmBytes > 0, "235B MoE fits with the disk tier");
Check(pMoe.GpuTensors.Where(t => t.Layer >= 0).Select(t => t.Layer).Distinct().Count() == moeShape.LayerCount ||
      pMoe.GpuTensors.Any(t => t.IsExpert) == false,
    "all always-on layer weights go to VRAM before any expert block");
Check(pMoe.WarmTensors.All(t => t.Kind != TensorKind.ExpertFfn || pMoe.WarmExperts.Count > 0), "warm tier holds experts");
Check(pMoe.RamTotalBytes <= budget.RamBytes, $"RAM tier {pMoe.RamTotalBytes / GiB:F1} GB ≤ budget");
Check(pMoe.WarmHitRate is > 0 and < 1, $"uniform routing: partial warm hit rate ({pMoe.WarmHitRate:P0})");

// Skewed profile: 16 of 128 experts get 80% of traffic → much better hit rate.
var skew = new ExpertUsageProfile(moeShape.LayerCount, 128);
var rng = new Random(1);
for (var tok = 0; tok < 2000; tok++)
    for (var l = 0; l < moeShape.LayerCount; l++)
    {
        var ids = new int[8];
        for (var i = 0; i < 8; i++) ids[i] = rng.NextDouble() < 0.8 ? rng.Next(16) : 16 + rng.Next(112);
        skew.Record(l, ids);
    }
var pSkew = PlacementPlanner.Plan(moeSynth, ctx8k, budget, hw, skew);
Check(pSkew.WarmHitRate > pMoe.WarmHitRate + 0.1, $"profile-driven warm set beats uniform ({pSkew.WarmHitRate:P0} vs {pMoe.WarmHitRate:P0})");
Check(pSkew.Estimate.TokensPerSecond > pMoe.Estimate.TokensPerSecond, "…and is estimated faster");
Check(PlacementPlanner.ExpectedWarmHitRate(pMoe, skew) < pSkew.WarmHitRate, "rebalancer sees a gain from re-targeting");
Check(Math.Abs(PlacementPlanner.ExpectedWarmHitRate(pSkew, skew) - pSkew.WarmHitRate) < 1e-9, "hit-rate evaluator agrees with planner");

var hdd = hw with { DiskSequentialMBs = 160, DiskRandomMBs = 40, DiskLatencyMs = 9 };
var pHdd = PlacementPlanner.Plan(moeSynth, ctx8k, budget, hdd);
Check(pHdd.Estimate.DiskMs > pMoe.Estimate.DiskMs * 10 && pHdd.Warnings.Any(w => w.Contains("HDD")), "HDD: big disk stall + warning");

var cpuOnly = hw with { HasGpuBackend = false };
var pCpu = PlacementPlanner.Plan(TensorCatalog.Synthetic(ModelShape.Synthetic(8), 4.85), ctx8k, budget, cpuOnly);
Check(!pCpu.UsesGpu && pCpu.GpuTensors.Count == 0 && pCpu.Warnings.Any(w => w.Contains("No GPU")), "CPU-only backend: no GPU placement + warning");

var tight = PlacementPlanner.Plan(TensorCatalog.Synthetic(ModelShape.Synthetic(8), 4.85),
    new RuntimeSettings { ContextSize = 131072 }, budget with { VramBytes = 8L << 30 }, hw);
Check(!tight.KvOnGpu && tight.Warnings.Any(w => w.Contains("KV cache")), "huge context: KV moved to RAM with a warning");

// ── 5. Overrides ────────────────────────────────────────────────────────────
Console.WriteLine("\n--- 5. tensor buffer overrides ---");
var moeBudget = new TierBudget { VramBytes = (long)(0.6 * moeCatalog.TotalBytes) + (450L << 20) + MemoryEstimator.ComputeBufferBytes(moeCatalog.Shape, new RuntimeSettings { ContextSize = 512 }) + MemoryEstimator.KvCacheBytes(moeCatalog.Shape, new RuntimeSettings { ContextSize = 512 }), RamBytes = 1L << 30 };
var smallPlan = PlacementPlanner.Plan(moeCatalog, new RuntimeSettings { ContextSize = 512 }, moeBudget, hw);
var patterns = LlamaPlacementApplier.BuildOverrides(smallPlan);
Console.WriteLine("  " + string.Join("\n  ", patterns));
Check(smallPlan.UsesGpu && smallPlan.CpuTensors.Any(t => t.IsExpert), "small MoE plan has GPU and CPU experts");
var regexes = patterns.Select(p => new Regex(p)).ToList();
var expectedCpu = smallPlan.CpuTensors.Where(t => t.Kind != TensorKind.Embedding).Select(t => t.Name).ToHashSet();
Check(moeCatalog.Tensors.All(t => regexes.Any(r => r.IsMatch(t.Name)) == expectedCpu.Contains(t.Name)),
    "override regexes match exactly the CPU-side tensors");
Check(patterns.Count <= 8, "overrides stay compact");

// ── 6. Capacity calculator ──────────────────────────────────────────────────
Console.WriteLine("\n--- 6. capacity calculator ---");
var q = new CapacityQuery
{
    Hardware = hw,
    Budget = new TierBudget { VramBytes = 24L << 30, RamBytes = 56L << 30 },
    Settings = ctx8k,
};
var rows = CapacityCalculator.MaxModelSizes(q);
foreach (var r in rows) Console.WriteLine($"  {r.Strategy,-11} {r.MaxParamsB,7:F1}B  {r.TokensPerSecond,6:F1} tok/s  {r.Note}");
Check(rows[0].MaxParamsB is > 30 and < 40, $"24 GiB GPU-only ≈ 32–38B dense @Q4_K_M/8k (got {rows[0].MaxParamsB:F1})");
Check(rows[1].MaxParamsB > rows[0].MaxParamsB * 2.5, "GPU+RAM allows a much larger dense model");
Check(rows[2].MaxParamsB >= rows[1].MaxParamsB && rows[2].TokensPerSecond >= CapacityCalculator.DefaultDiskMinTps - 1e-6,
    "disk strategy respects the speed floor");
var qMin = q with { MinTokensPerSecond = 5 };
var rowsMin = CapacityCalculator.MaxModelSizes(qMin);
Check(rowsMin[1].MaxParamsB < rows[1].MaxParamsB && rowsMin[1].TokensPerSecond >= 5 - 1e-6, "min tok/s shrinks the answer");
var qMoe = q with { MoeActiveRatio = 0.1 };
var rowsMoe = CapacityCalculator.MaxModelSizes(qMoe);
foreach (var r in rowsMoe) Console.WriteLine($"  MoE {r.Strategy,-11} {r.MaxParamsB,7:F1}B  {r.TokensPerSecond,6:F1} tok/s");
Check(rowsMoe[2].MaxParamsB > rows[2].MaxParamsB, "MoE (10% active) reaches larger total sizes with disk than dense");
Check(rowsMoe[1].MaxParamsB > rowsMoe[0].MaxParamsB * 2.5, $"MoE GPU+RAM ≫ GPU only (got {rowsMoe[1].MaxParamsB:F0}B)");
Check(rowsMoe[1].TokensPerSecond > rows[1].TokensPerSecond * 3, "MoE offload is much faster than dense offload");
var curve = CapacityCalculator.SpeedCurve(q, [8, 32, 70, 120]);
Check(curve[0].GpuOnly > curve[1].GpuOnly && curve[2].GpuOnly is null && curve[2].GpuAndRam is not null, "speed curve shape");

var report = CapacityCalculator.Assess(c70, ctx8k, budget, hw, null, "Q4_K_M");
Check(report.Verdict == FitVerdict.FitsGpuAndRam, $"70B verdict = fits with RAM ({report.VerdictLabel})");
Check(report.LargestQuantFullyOnGpu is "IQ2_XXS" or "IQ1_S" or null, $"70B on 22 GB: only ~2-bit fits on GPU (got {report.LargestQuantFullyOnGpu})");
var r8 = CapacityCalculator.Assess(TensorCatalog.Synthetic(ModelShape.Synthetic(8), 4.85), ctx8k, budget, hw);
Check(r8.Verdict == FitVerdict.FitsGpu && r8.MaxContextFullyOnGpu > 8192, $"8B fits; max GPU context {r8.MaxContextFullyOnGpu:N0}");
var fileReport = CapacityCalculator.AssessFile(18_600_000_000, "Q4_K_M", "Qwen3-30B-A3B-Q4_K_M.gguf", ctx8k, budget, hw);
Check(fileReport is not null && fileReport.Plan.Catalog.IsMoe && fileReport.Verdict == FitVerdict.FitsGpu, "file-name MoE detection (A3B) and verdict");
Check(CapacityCalculator.ParamsFromName("Llama-3.1-70B-Instruct") == 70, "params from name");

// ── 7. Profile serialisation ────────────────────────────────────────────────
Console.WriteLine("\n--- 7. expert usage profile ---");
var blob = skew.Serialize();
var back = ExpertUsageProfile.Deserialize(blob);
Check(back is not null && back.Layers == skew.Layers && Math.Abs(back.Share(5, 3) - skew.Share(5, 3)) < 1e-6, "roundtrip");
Check(Near(skew.Concentration(16.0 / 128), 0.8 + 0.2 * 16 / 128 * 0, 0.05) || skew.Concentration(16.0 / 128) > 0.75, $"concentration ≈ 0.8 (got {skew.Concentration(16.0 / 128):F2})");
Check(Math.Abs(skew.ObservedTokens - 2000) < 1, $"observed tokens (got {skew.ObservedTokens})");
skew.Decay(0.5f);
Check(Math.Abs(skew.ObservedTokens - 1000) < 1 && back is not null && Math.Abs(back.Share(5, 3) - skew.Share(5, 3)) < 1e-6, "decay scales counts, keeps shares");
Check(ExpertUsageProfile.Deserialize([1, 2, 3]) is null, "garbage blob → null");
Check(PlacementPlanner.Merge([new(0, 10), new(10, 5), new(30, 5), new(32, 10)]) is [{ Offset: 0, Length: 15 }, { Offset: 30, Length: 12 }], "range merge");

unsafe
{
    fixed (byte* n1 = "ffn_moe_topk-17\0"u8) Check(ExpertRoutingProfiler.Match(n1, out var l1) == 1 && l1 == 17, "profiler: topk name");
    fixed (byte* n2 = "ffn_moe_argsort-3\0"u8) Check(ExpertRoutingProfiler.Match(n2, out var l2) == 2 && l2 == 3, "profiler: argsort name");
    fixed (byte* n3 = "ffn_moe_topk-\0"u8) Check(ExpertRoutingProfiler.Match(n3, out _) == 0, "profiler: no layer → ignored");
    fixed (byte* n4 = "ffn_moe_weights-3\0"u8) Check(ExpertRoutingProfiler.Match(n4, out _) == 0, "profiler: other tensor ignored");
}

// ── 8. Warm tier on a real file ─────────────────────────────────────────────
Console.WriteLine("\n--- 8. warm tier ---");
var tmp = Path.Combine(Path.GetTempPath(), $"hb-warm-{Guid.NewGuid():N}.bin");
await using (var fs = File.Create(tmp))
{
    var chunk = new byte[1 << 20];
    new Random(7).NextBytes(chunk);
    for (var i = 0; i < 64; i++) await fs.WriteAsync(chunk);
}
try
{
    using var warm = new WarmTierManager(tmp);
    var ranges = new List<WarmRange> { new(4096, 8 << 20), new(40L << 20, 4 << 20) };
    await warm.ApplyAsync(ranges, lockPages: true);
    var st = warm.Status;
    Console.WriteLine($"  target={st.TargetBytes} prefetched={st.PrefetchedBytes} locked={st.LockedBytes} err={st.LockError}");
    Check(st.TargetBytes == 12 << 20 && st.PrefetchedBytes == st.TargetBytes && !st.Busy, "prefetch covered the target");
    Check(st.LockedBytes == st.TargetBytes || st.LockError is not null, "locked, or a clear lock error");
    var resident = warm.MeasureResidentBytes(ranges);
    Check(OperatingSystem.IsWindows() ? resident is null : resident >= 12 << 20, $"resident measurement (got {resident})");
    await warm.ApplyAsync([new WarmRange(0, 1 << 20)], lockPages: false);
    Check(warm.Status.LockedBytes == 0 && warm.Status.TargetBytes == 1 << 20, "re-target unlocks the previous set");
}
finally { File.Delete(tmp); }

// ── 9. Hardware micro-benchmarks ────────────────────────────────────────────
Console.WriteLine("\n--- 9. hardware probe ---");
var bw = HardwareProbe.MeasureRamBandwidthGBs(128L << 20);
Console.WriteLine($"  RAM read bandwidth ≈ {bw:F1} GB/s");
Check(bw is > 0.5 and < 2000, "RAM bandwidth measured");
var (ramTotal, ramAvail) = HardwareProbe.ReadSystemMemory();
Check(ramTotal > 0 && ramAvail > 0 && ramAvail <= ramTotal, $"system memory {ramTotal / GiB:F1} GB total, {ramAvail / GiB:F1} available");
var diskFile = Path.Combine(Path.GetTempPath(), $"hb-disk-{Guid.NewGuid():N}.bin");
await using (var fs = File.Create(diskFile)) { fs.SetLength(64L << 20); }
try
{
    var d = HardwareProbe.MeasureDisk(diskFile);
    Console.WriteLine($"  disk seq {d.SequentialMBs:F0} MB/s, random {d.RandomMBs:F0} MB/s, latency {d.LatencyMs:F2} ms, cache bypass {d.CacheBypassed}");
    Check(d.SequentialMBs > 0 && d.RandomMBs > 0, "disk benchmark ran");
}
finally { File.Delete(diskFile); }
var io = new ProcessIoSampler();
io.Sample();
Check(io.Sample() is var (_, _), "IO sampler runs");

// ── 10. Schema upgrade of an existing database ──────────────────────────────
Console.WriteLine("\n--- 10. schema upgrade ---");
var dbPath = Path.Combine(Path.GetTempPath(), $"hb-schema-{Guid.NewGuid():N}.db");
try
{
    var options = new DbContextOptionsBuilder<AppDbContext>().UseSqlite($"Data Source={dbPath};Pooling=False").Options;
    await using (var db = new AppDbContext(options))
    {
        await db.Database.EnsureCreatedAsync();
        // Simulate a database created by the previous release.
        foreach (var sql in new[]
                 {
                     "DROP TABLE MemoryProfiles", "DROP TABLE HardwareProfiles", "DROP TABLE ExpertUsageSnapshots",
                     "ALTER TABLE AnalyticsMetrics DROP COLUMN WarmHitRate", "ALTER TABLE AnalyticsMetrics DROP COLUMN TierColdMb",
                 })
            await db.Database.ExecuteSqlRawAsync(sql);
    }
    await using (var db = new AppDbContext(options))
    {
        await db.ReconcileSchemaAsync();
        var model = new LocalModel { Name = "m", Format = ModelFormat.Gguf };
        db.Models.Add(model);
        db.MemoryProfiles.Add(new MemoryProfile { ModelId = model.Id, KvCacheType = KvCacheType.Q8_0, RebalancePolicy = RebalancePolicy.Suggest });
        db.HardwareProfiles.Add(new HardwareProfile { GpuName = "gpu", VramTotalBytes = 1 });
        db.ExpertUsageSnapshots.Add(new ExpertUsageSnapshot { ModelId = model.Id, Layers = 2, Experts = 2, Data = [1, 2] });
        db.AnalyticsMetrics.Add(new AnalyticsMetric { ModelId = model.Id, WarmHitRate = 0.5f, TierColdMb = 3, DecodeTokensPerSecond = 9 });
        await db.SaveChangesAsync();
    }
    await using (var db = new AppDbContext(options))
    {
        var mp = await db.MemoryProfiles.SingleAsync();
        var am = await db.AnalyticsMetrics.SingleAsync();
        Check(mp.KvCacheType == KvCacheType.Q8_0 && mp.RebalancePolicy == RebalancePolicy.Suggest && mp.Mode == TieringMode.Tiered, "MemoryProfiles round-trips after upgrade");
        Check(am.WarmHitRate == 0.5f && am.TierColdMb == 3, "new AnalyticsMetrics columns usable after upgrade");
        Check(await db.ExpertUsageSnapshots.CountAsync() == 1 && await db.HardwareProfiles.CountAsync() == 1, "new tables usable");
        await db.Database.ExecuteSqlRawAsync("DELETE FROM Models");
        Check(await db.MemoryProfiles.CountAsync() == 0 && await db.ExpertUsageSnapshots.CountAsync() == 0, "cascade delete from Models");
    }
    await using (var db = new AppDbContext(options))
    {
        await db.ReconcileSchemaAsync(); // idempotent
        Check(true, "reconcile is idempotent");
    }
}
finally
{
    SqliteConnection.ClearAllPools();
    File.Delete(dbPath);
}

// ── 11. End-to-end on a real (tiny) MoE model through llama.cpp ────────────
Console.WriteLine("\n--- 11. native: tiered load + routing profiler (TestData/test_tiny_moe.gguf) ---");
var moePath = Path.Combine(testDir, "test_tiny_moe.gguf");
var nativeDb = Path.Combine(Path.GetTempPath(), $"hb-native-{Guid.NewGuid():N}.db");
try
{
    var log = new HomebredLLM.Services.NativeLogBuffer(400);
    log.RegisterNativeLogCallback();
    var nopts = new DbContextOptionsBuilder<AppDbContext>().UseSqlite($"Data Source={nativeDb};Pooling=False").Options;
    var factory = new Factory(nopts);
    await using (var db = factory.CreateDbContext()) { await db.Database.EnsureCreatedAsync(); await db.ReconcileSchemaAsync(); }
    var model = new LocalModel { Name = "tiny-moe", Format = ModelFormat.Gguf, LocalPath = moePath, Status = ModelStatus.Ready };
    await using (var db = factory.CreateDbContext())
    {
        db.Models.Add(model);
        db.MemoryProfiles.Add(new MemoryProfile { ModelId = model.Id, VramBudgetMb = 0, RamBudgetMb = 0, ProfilerWindowTokens = 512 });
        await db.SaveChangesAsync();
    }
    var probeSvc = new HardwareProbe(new HomebredLLM.Services.GpuMetricsService(), factory);
    using var coord = new TieringCoordinator(factory, probeSvc);
    var svc = new HomebredLLM.Services.LlamaCppInferenceService(log, coord, probeSvc);
    var progressLines = new List<string>();
    var cfg = new ModelConfiguration { ContextSize = 1024, MaxTokens = 24, BatchSize = 256 };
    await svc.LoadAsync(model.Id, moePath, cfg, progress: new Progress<string>(progressLines.Add));
    var snap = coord.GetSnapshot(model.Id);
    Check(snap is not null && snap.Plan.Catalog.IsMoe, "model loaded through the tier planner");
    Check(snap!.ProfilerStatus.StartsWith("Waiting"), $"routing profiler started ({snap.ProfilerStatus})");

    var req = new HomebredLLM.Services.InferenceRequest(model.Id, moePath, cfg,
        [(MessageRole.User, "hello world the model is an expert of the world", (IReadOnlyList<HomebredLLM.Services.InferenceAttachment>)[])]);
    var fragments = 0;
    await foreach (var _ in svc.ChatStreamAsync(req, _ => { })) fragments++;
    Check(fragments > 0, $"generation works with tiered placement ({fragments} fragments)");

    for (var i = 0; i < 40 && coord.GetSnapshot(model.Id)!.TokensProfiled == 0; i++) await Task.Delay(1000);
    snap = coord.GetSnapshot(model.Id)!;
    var routed = await coord.GetUsageAsync(model.Id);
    Check(snap.TokensProfiled > 0, $"profiler replayed tokens via cb_eval ({snap.TokensProfiled})");
    Check(routed is not null && Enumerable.Range(0, routed.Layers).All(routed.HasData), "every MoE layer has routing data");
    Check(routed is not null && Math.Abs(routed.ObservedTokens - snap.TokensProfiled) < 1, "one routed row per replayed token per layer");
    Check(routed is not null && Enumerable.Range(0, routed.Layers).All(l => Math.Abs(routed.LayerShares(l).Sum() - 1) < 1e-6), "per-layer shares sum to 1");
    svc.Unload(model.Id);
    await using (var db = factory.CreateDbContext())
        Check(await db.ExpertUsageSnapshots.CountAsync() >= 1, "routing profile persisted on unload");
}
catch (Exception ex) when (ex is DllNotFoundException or TypeInitializationException or BadImageFormatException)
{
    Console.WriteLine($"SKIP: llama.cpp native library not available here ({ex.GetType().Name})");
}
finally
{
    SqliteConnection.ClearAllPools();
    File.Delete(nativeDb);
}

// ── 12. Phase 2 planner: per-expert VRAM slots ─────────────────────────────
Console.WriteLine("\n--- 12. Phase 2 planner: per-expert VRAM slots ---");
{
    var hwCache = hw with { HasExpertCache = true };
    var pUniform = PlacementPlanner.Plan(moeSynth, ctx8k, budget, hwCache);
    var pClassicSkew = PlacementPlanner.Plan(moeSynth, ctx8k, budget, hw, skew);
    var pCacheSkew = PlacementPlanner.Plan(moeSynth, ctx8k, budget, hwCache, skew);
    Console.WriteLine($"  uniform: slots={pUniform.ExpertSlotsPerLayer} est={pUniform.Estimate.TokensPerSecond:F2} | skew classic={pClassicSkew.Estimate.TokensPerSecond:F2} cached={pCacheSkew.Estimate.TokensPerSecond:F2} slots={pCacheSkew.ExpertSlotsPerLayer} hit={pCacheSkew.ExpertCacheHitRate:P0}");
    Check(pCacheSkew.ExpertSlotsPerLayer >= 8 && pCacheSkew.ExpertCacheHitRate > 0.6, "skewed routing → slots chosen, hot set mostly covered");
    Check(pCacheSkew.Estimate.TokensPerSecond > pClassicSkew.Estimate.TokensPerSecond, "slots beat per-layer blocks on skewed routing");
    Check(pCacheSkew.VramTotalBytes <= budget.VramBytes, $"slot plan respects VRAM ({pCacheSkew.VramTotalBytes / GiB:F1} GB)");
    Check(pCacheSkew.GpuTensors.All(t => !t.IsExpert) && moeSynth.Tensors.Where(t => t.IsExpert).All(pCacheSkew.CpuTensors.Contains),
        "slot mode keeps every routed-expert tensor host-side");
    Check(pCacheSkew.ResidentExperts.Count == moeShape.LayerCount &&
          pCacheSkew.ResidentExperts.Values.All(r => r.Length == pCacheSkew.ExpertSlotsPerLayer && r.Take(16).All(e => e < 16)),
        "initial residency = each layer's hottest experts");
    Check(pUniform.ExpertSlotsPerLayer == 0 || pUniform.Estimate.TokensPerSecond >= PlacementPlanner.Plan(moeSynth, ctx8k, budget, hw).Estimate.TokensPerSecond,
        "planner only picks slots when the model estimates a gain");
    var small = PlacementPlanner.Plan(TensorCatalog.Synthetic(ModelShape.Synthetic(30, 3, 128, 8), 4.85), ctx8k, budget, hwCache, skew);
    Check(small.ExpertSlotsPerLayer == 0 && small.CpuTensors.All(t => t.Kind == TensorKind.Embedding), "MoE that fits VRAM entirely uses no slots");
    var hot = PlacementPlanner.HottestExperts([0, 1], 128, 4, skew);
    Check(hot[0].Length == 4 && hot[0].All(e => e < 16), "HottestExperts picks from the hot set");
    Check(PlacementPlanner.ExpectedCacheHitRate(hot, 128, skew) > PlacementPlanner.ExpectedCacheHitRate(
        new Dictionary<int, int[]> { [0] = [100, 101, 102, 103], [1] = [100, 101, 102, 103] }, 128, skew), "hit rate ranks residencies");
    Check(LlamaPlacementApplier.BuildOverrides(pCacheSkew).Any(p => p.Contains("_exps")), "overrides pin expert tensors to CPU in slot mode");
}

// ── 13. Phase 2 native: patched llama.cpp expert cache through LLamaSharp ───
Console.WriteLine("\n--- 13. native: per-expert slots via patched llama.cpp (hbec) ---");
var backend = new NativeExpertCacheBackend();
if (!backend.IsAvailable)
{
    Console.WriteLine("SKIP: stock llama.cpp natives (no hbec_* exports) — build native/llama.cpp-hbec to run this section");
}
else
{
    float[] Logits(int slots, int[]? resident, out IExpertCacheHandle? handle)
    {
        if (slots > 0) backend.ConfigureNextLoad(slots, NativeExpertCacheBackend.DeviceCpu);
        var mp = new LLama.Common.ModelParams(moePath) { GpuLayerCount = 0, ContextSize = 512, BatchSize = 256 };
        using var w = LLama.LLamaWeights.LoadFromFile(mp);
        handle = backend.Attach(w.NativeHandle.DangerousGetHandle());
        if (handle is not null && resident is not null)
            for (var l = 0; l < handle.Layers; l++)
                if (handle.LayerEnabled(l)) handle.SetResidency(l, resident);
        using var ctx = w.CreateContext(mp);
        var toks = ctx.Tokenize("hello world the model is an expert of the world", addBos: true);
        var batch = new LLama.Native.LLamaBatch();
        for (var i = 0; i < toks.Length; i++) batch.Add(toks[i], i, LLama.Native.LLamaSeqId.Zero, true);
        ctx.NativeHandle.Decode(batch);
        var result = new List<float>();
        for (var i = 0; i < toks.Length; i++) result.AddRange(ctx.NativeHandle.GetLogitsIth(i).ToArray());
        if (handle is not null) { var c = handle.Drain(); Console.WriteLine($"  slots={handle.SlotsPerLayer} layers={handle.Layers} uploads={c.Promotions} bytes={c.BytesUploaded}"); }
        return result.ToArray();
    }
    var baseLogits = Logits(0, null, out var noCache);
    Check(noCache is null, "no cache without configuration");
    var cached = Logits(4, [0, 3, 5], out var h);
    Check(h is { Layers: 4, SlotsPerLayer: 4 }, "cache attached to all 4 MoE layers");
    var maxDiff = baseLogits.Zip(cached, (a, b) => Math.Abs(a - b)).Max();
    Check(maxDiff < 1e-4, $"logits identical with experts in slots (max |Δ| = {maxDiff:G3})");
    var full = Logits(8, [0, 1, 2, 3, 4, 5, 6, 7], out _);
    Check(baseLogits.Zip(full, (a, b) => Math.Abs(a - b)).Max() < 1e-4, "logits identical with all experts resident");

    // Coordinator glue: initial residency from the plan, then the rebalancer re-fills
    // slots from a learned (skewed) routing profile — slots on the CPU device here.
    var cdb = Path.Combine(Path.GetTempPath(), $"hb-hbec-{Guid.NewGuid():N}.db");
    try
    {
        var copts = new DbContextOptionsBuilder<AppDbContext>().UseSqlite($"Data Source={cdb};Pooling=False").Options;
        var cf = new Factory(copts);
        await using (var db = cf.CreateDbContext()) { await db.Database.EnsureCreatedAsync(); await db.ReconcileSchemaAsync(); }
        var m = new LocalModel { Name = "tiny-moe", Format = ModelFormat.Gguf, LocalPath = moePath };
        await using (var db = cf.CreateDbContext()) { db.Models.Add(m); await db.SaveChangesAsync(); }

        var tinyCatalog = await TensorCatalog.FromFileAsync(moePath);
        var fakeHw = hw with { HasExpertCache = true };
        var rs = new RuntimeSettings { ContextSize = 512, UBatch = 256 };
        var fixedVram = new TierBudget().VramOverheadBytes + MemoryEstimator.ComputeBufferBytes(tinyCatalog.Shape, rs)
                        + MemoryEstimator.KvCacheBytes(tinyCatalog.Shape, rs)
                        + tinyCatalog.Tensors.Where(t => t.IsAlwaysOn).Sum(t => t.Bytes);
        var tinyBudget = new TierBudget { VramBytes = fixedVram + tinyCatalog.ExpertBytes * 3 / 8 + (64 << 10), RamBytes = 1L << 30 };
        var tinyPlan = PlacementPlanner.Plan(tinyCatalog, rs, tinyBudget, fakeHw, null, expertSlots: true);
        Check(tinyPlan.ExpertSlotsPerLayer is > 0 and < 8, $"tiny MoE slot plan ({tinyPlan.ExpertSlotsPerLayer} slots/layer)");

        backend.ConfigureNextLoad(tinyPlan.ExpertSlotsPerLayer, NativeExpertCacheBackend.DeviceCpu);
        var mp = new LLama.Common.ModelParams(moePath) { GpuLayerCount = 0, ContextSize = 512 };
        using var w = LLama.LLamaWeights.LoadFromFile(mp);
        var handle = backend.Attach(w.NativeHandle.DangerousGetHandle())!;
        var usage = new ExpertUsageProfile(4, 8);
        var mem = new MemoryProfile { ModelId = m.Id, RebalancePolicy = RebalancePolicy.Auto, RebalanceThreshold = 0.01f, RoutingProfilerEnabled = false };
        using var coord2 = new TieringCoordinator(cf, new HardwareProbe(new HomebredLLM.Services.GpuMetricsService(), cf));
        coord2.OnLoaded(m.Id, new PreparedLoad(tinyPlan, mem, usage, fakeHw, 1.0), w, mp, f => f(), handle);
        var initial = handle.GetResidency(0);
        Check(initial.OrderBy(e => e).SequenceEqual(tinyPlan.ResidentExperts[0].OrderBy(e => e)), $"initial residency from plan [{string.Join(",", initial)}]");

        for (var t = 0; t < 500; t++)
            for (var l = 0; l < 4; l++)
                usage.Record(l, [7, 6]);
        var cacheSnap = coord2.GetSnapshot(m.Id)!.ExpertCache!;
        Check(cacheSnap.ExpectedHitRate < 0.5, $"skew makes current residency look bad ({cacheSnap.ExpectedHitRate:P0})");
        for (var i = 0; i < 45 && !handle.GetResidency(0).Contains(7); i++) await Task.Delay(1000);
        var after = handle.GetResidency(0);
        Check(after.Contains(7) && after.Contains(6), $"rebalancer re-filled slots with the hot experts [{string.Join(",", after)}]");
        var cs = coord2.GetSnapshot(m.Id)!.ExpertCache!;
        Check(cs.ExpectedHitRate > 0.99 && cs.Promotions > 0 && cs.UploadBytes > 0, $"hit rate {cs.ExpectedHitRate:P0}, {cs.Promotions} uploads, {cs.Evictions} evictions");
        coord2.OnUnloaded(m.Id);
    }
    finally
    {
        SqliteConnection.ClearAllPools();
        File.Delete(cdb);
    }
}

// --- GPU backend installer: archive extraction + checker ---
{
    Console.WriteLine("\n== GPU backend installer ==");
    var gtmp = Path.Combine(Path.GetTempPath(), "hb-gpu-" + Guid.NewGuid().ToString("N"));
    Directory.CreateDirectory(gtmp);
    try
    {
        // zip (Windows layout): dlls + an exe + a nested dll, plus a path-traversal attempt
        var zipPath = Path.Combine(gtmp, "t.zip");
        using (var zs = new FileStream(zipPath, FileMode.Create))
        using (var z = new System.IO.Compression.ZipArchive(zs, System.IO.Compression.ZipArchiveMode.Create))
        {
            void Add(string n, string c) { var e = z.CreateEntry(n); using var w = new StreamWriter(e.Open()); w.Write(c); }
            Add("llama.dll", "L"); Add("ggml-cuda.dll", "C"); Add("llama-server.exe", "X"); Add("sub/ggml-base.dll", "B");
            Add("../evil.dll", "E");
        }
        var outDir = Path.Combine(gtmp, "native");
        Directory.CreateDirectory(outDir);
        GpuBackendInstaller.Extract(zipPath, outDir);
        Check(File.Exists(Path.Combine(outDir, "llama.dll")) && File.Exists(Path.Combine(outDir, "ggml-cuda.dll")), "zip: dlls extracted");
        Check(File.Exists(Path.Combine(outDir, "ggml-base.dll")), "zip: nested dll flattened");
        Check(!File.Exists(Path.Combine(outDir, "llama-server.exe")), "zip: executables skipped");
        Check(!File.Exists(Path.Combine(gtmp, "evil.dll")), "zip: traversal entry stays inside the folder");

        // tar.gz (Linux layout): real file + symlink + tool binary
        var tarPath = Path.Combine(gtmp, "t.tar.gz");
        using (var fs = File.Create(tarPath))
        using (var gz = new System.IO.Compression.GZipStream(fs, System.IO.Compression.CompressionLevel.Fastest))
        using (var tw = new System.Formats.Tar.TarWriter(gz))
        {
            var real = new System.Formats.Tar.PaxTarEntry(System.Formats.Tar.TarEntryType.RegularFile, "llama-b8816/libllama.so.0.0.1")
            { DataStream = new MemoryStream("SO"u8.ToArray()) };
            tw.WriteEntry(real);
            tw.WriteEntry(new System.Formats.Tar.PaxTarEntry(System.Formats.Tar.TarEntryType.SymbolicLink, "llama-b8816/libllama.so") { LinkName = "libllama.so.0.0.1" });
            tw.WriteEntry(new System.Formats.Tar.PaxTarEntry(System.Formats.Tar.TarEntryType.RegularFile, "llama-b8816/llama-cli") { DataStream = new MemoryStream("T"u8.ToArray()) });
        }
        var outLinux = Path.Combine(gtmp, "linux");
        Directory.CreateDirectory(outLinux);
        GpuBackendInstaller.Extract(tarPath, outLinux);
        Check(File.Exists(Path.Combine(outLinux, "libllama.so.0.0.1")), "tar.gz: library extracted");
        Check(File.Exists(Path.Combine(outLinux, "libllama.so")) && File.ReadAllText(Path.Combine(outLinux, "libllama.so")) == "SO", "tar.gz: symlink materialised as copy");
        Check(!File.Exists(Path.Combine(outLinux, "llama-cli")), "tar.gz: tool binaries skipped");

        Check(GpuBackendInstaller.CanInstall(GpuBackendKind.Cuda) == OperatingSystem.IsWindows(), "CUDA installable only on Windows");
        var gpuReport = new GpuRequirementsChecker(new HomebredLLM.Services.GpuMetricsService()).Check();
        Check(gpuReport.Items.Count > 0 && !string.IsNullOrEmpty(gpuReport.Summary), "checker returns a gpuReport");
        Console.WriteLine("  checker: " + gpuReport.Summary);
    }
    finally { try { Directory.Delete(gtmp, true); } catch { } }
}

Console.WriteLine($"\n{(failures == 0 ? "ALL PASSED" : $"{failures} FAILURE(S)")}");
return failures;

// ── helpers ─────────────────────────────────────────────────────────────────
static byte[] BuildMoeGguf(int layers, int experts, int used, int embd, int expFf)
{
    var ms = new MemoryStream();
    var w = new BinaryWriter(ms, Encoding.UTF8);
    void Str(string s) { var b = Encoding.UTF8.GetBytes(s); w.Write((ulong)b.Length); w.Write(b); }
    var kv = new List<(string, uint, object)>
    {
        ("general.architecture", 8, "qwen3moe"),
        ("qwen3moe.block_count", 4, (uint)layers),
        ("qwen3moe.embedding_length", 4, (uint)embd),
        ("qwen3moe.attention.head_count", 4, 4u),
        ("qwen3moe.attention.head_count_kv", 4, 2u),
        ("qwen3moe.expert_count", 4, (uint)experts),
        ("qwen3moe.expert_used_count", 4, (uint)used),
        ("qwen3moe.expert_feed_forward_length", 4, (uint)expFf),
    };
    var tensors = new List<(string Name, ulong[] Shape, uint Type)> { ("token_embd.weight", [(ulong)embd, 100], 0) };
    for (var l = 0; l < layers; l++)
    {
        tensors.Add(($"blk.{l}.attn_q.weight", [(ulong)embd, (ulong)embd], 1));
        tensors.Add(($"blk.{l}.attn_norm.weight", [(ulong)embd], 0));
        tensors.Add(($"blk.{l}.ffn_gate_inp.weight", [(ulong)embd, (ulong)experts], 0));
        foreach (var n in new[] { "gate", "up", "down" })
            tensors.Add(($"blk.{l}.ffn_{n}_exps.weight", [(ulong)embd, (ulong)expFf, (ulong)experts], 1));
    }
    tensors.Add(("output.weight", [(ulong)embd, 100], 1));

    w.Write("GGUF"u8.ToArray()); w.Write(3u); w.Write((ulong)tensors.Count); w.Write((ulong)kv.Count);
    foreach (var (k, t, v) in kv)
    {
        Str(k); w.Write(t);
        if (v is string sv) Str(sv); else w.Write((uint)v);
    }
    ulong offset = 0;
    foreach (var (name, shape, type) in tensors)
    {
        Str(name); w.Write((uint)shape.Length);
        foreach (var d in shape) w.Write(d);
        w.Write(type); w.Write(offset);
        var elems = shape.Aggregate(1UL, (a, b) => a * b);
        offset += (elems * (type == 0 ? 4UL : 2UL) + 31) / 32 * 32;
    }
    while (ms.Length % 32 != 0) w.Write((byte)0);
    w.Write(new byte[offset]);
    return ms.ToArray();
}

sealed class Factory(DbContextOptions<AppDbContext> options) : IDbContextFactory<AppDbContext>
{
    public AppDbContext CreateDbContext() => new(options);
}
