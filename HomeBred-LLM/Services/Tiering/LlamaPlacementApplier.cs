using System.Text;
using System.Text.RegularExpressions;
using LLama.Abstractions;
using LLama.Common;
using LLama.Native;

namespace HomebredLLM.Services.Tiering;

/// <summary>
/// Translates a <see cref="PlacementPlan"/> into LLamaSharp load parameters.
/// GPU placement uses llama.cpp's per-tensor buffer overrides
/// (<c>--override-tensor</c>): every layer is nominally offloaded and the
/// tensors the plan keeps CPU-side are pinned to the "CPU" buffer type.
/// CPU-side tensors stay memory-mapped, which is what makes the RAM/disk
/// split (page cache + <see cref="WarmTierManager"/>) work.
/// </summary>
public static class LlamaPlacementApplier
{
    public static void Apply(ModelParams p, PlacementPlan plan)
    {
        p.UseMemorymap = true;   // CPU tensors must stay file-backed for the warm/cold split
        p.UseMemoryLock = false; // the warm tier locks selectively instead of the whole file

        if (plan.UsesGpu)
        {
            p.GpuLayerCount = plan.Catalog.Shape.LayerCount + 1; // all repeating layers + output
            p.TensorBufferOverrides = BuildOverrides(plan)
                .Select(pattern => new TensorBufferOverride(pattern, "CPU"))
                .ToList();
        }
        else
        {
            p.GpuLayerCount = 0;
            p.TensorBufferOverrides = [];
        }

        p.NoKqvOffload = !plan.KvOnGpu;
        var kvType = ToGgml(plan.Settings.KvType);
        p.TypeK = kvType;
        p.TypeV = kvType;
        // llama.cpp requires flash attention for a quantized V cache.
        p.FlashAttention = plan.Settings.FlashAttention ||
                           plan.Settings.KvType is KvCacheType.Q8_0 or KvCacheType.Q4_0;
    }

    public static GGMLType ToGgml(KvCacheType t) => t switch
    {
        KvCacheType.F32 => (GGMLType)0,
        KvCacheType.Q4_0 => (GGMLType)2,
        KvCacheType.Q8_0 => (GGMLType)8,
        _ => (GGMLType)1,
    };

    /// <summary>
    /// Compact regex patterns matching exactly the CPU-side tensors of the plan.
    /// Whole layers become <c>^blk\.(3|4)\.</c>; partial layers become one
    /// pattern per tensor suffix listing the layers (<c>^blk\.(0|1)\.ffn_up_exps\.weight$</c>).
    /// </summary>
    public static List<string> BuildOverrides(PlacementPlan plan)
    {
        var patterns = new List<string>();
        var cpu = new HashSet<string>(plan.CpuTensors
            .Where(t => t.Kind != TensorKind.Embedding) // llama.cpp keeps the input layer on CPU anyway
            .Select(t => t.Name), StringComparer.Ordinal);
        if (cpu.Count == 0) return patterns;

        var byLayer = plan.Catalog.Tensors.Where(t => t.Layer >= 0).GroupBy(t => t.Layer).ToList();
        var wholeLayers = byLayer.Where(g => g.All(t => cpu.Contains(t.Name))).Select(g => g.Key).OrderBy(l => l).ToList();
        if (wholeLayers.Count > 0)
            patterns.Add($@"^blk\.({string.Join('|', wholeLayers)})\.");

        var whole = new HashSet<int>(wholeLayers);
        var partial = new SortedDictionary<string, SortedSet<int>>(StringComparer.Ordinal);
        foreach (var t in plan.CpuTensors.Where(t => t.Layer >= 0 && !whole.Contains(t.Layer) && cpu.Contains(t.Name)))
        {
            var suffix = t.Name[(t.Name.IndexOf('.', 4) + 1)..];
            if (!partial.TryGetValue(suffix, out var layers)) partial[suffix] = layers = [];
            layers.Add(t.Layer);
        }
        foreach (var (suffix, layers) in partial)
            patterns.Add($@"^blk\.({string.Join('|', layers)})\.{Regex.Escape(suffix)}$");

        foreach (var t in plan.CpuTensors.Where(t => t.Layer < 0 && cpu.Contains(t.Name)))
            patterns.Add($"^{Regex.Escape(t.Name)}$");

        return patterns;
    }

    /// <summary>Human-readable summary of the overrides, for the load log / UI.</summary>
    public static string Describe(PlacementPlan plan)
    {
        var sb = new StringBuilder();
        foreach (var p in BuildOverrides(plan)) sb.Append(p).Append(" → CPU; ");
        return sb.Length == 0 ? "all weights on the GPU" : sb.ToString();
    }
}
