using System.IO.Compression;

namespace HomebredLLM.Services.Tiering;

/// <summary>
/// Per-layer × per-expert routing counts with exponential decay, so the
/// "hot set" follows how the model is actually being used. Thread-safe:
/// the routing profiler writes from llama.cpp's eval callback thread while
/// the planner/UI read.
/// </summary>
public sealed class ExpertUsageProfile
{
    private readonly float[] _counts;
    private readonly double[] _layerTotals;
    private readonly object _gate = new();

    public int Layers { get; }
    public int Experts { get; }

    private int _picksPerToken = 1;

    /// <summary>Approximate number of routed tokens observed (decayed with the counts).</summary>
    public double ObservedTokens
    {
        get { lock (_gate) return _layerTotals.Max() / _picksPerToken; }
    }

    public DateTime UpdatedAt { get; private set; } = DateTime.UtcNow;

    public ExpertUsageProfile(int layers, int experts)
    {
        if (layers <= 0 || experts <= 0) throw new ArgumentOutOfRangeException(nameof(layers));
        Layers = layers;
        Experts = experts;
        _counts = new float[layers * experts];
        _layerTotals = new double[layers];
    }

    /// <summary>Records that one token in <paramref name="layer"/> was routed to <paramref name="expertIds"/>.</summary>
    public void Record(int layer, ReadOnlySpan<int> expertIds)
    {
        if ((uint)layer >= (uint)Layers) return;
        lock (_gate)
        {
            var row = layer * Experts;
            foreach (var e in expertIds)
            {
                if ((uint)e >= (uint)Experts) continue;
                _counts[row + e] += 1;
                _layerTotals[layer] += 1;
            }
            if (expertIds.Length > 0) _picksPerToken = expertIds.Length;
            UpdatedAt = DateTime.UtcNow;
        }
    }

    /// <summary>Multiplies all counts by <paramref name="factor"/> (0..1) so old usage fades.</summary>
    public void Decay(float factor)
    {
        factor = Math.Clamp(factor, 0f, 1f);
        lock (_gate)
        {
            for (var i = 0; i < _counts.Length; i++) _counts[i] *= factor;
            for (var l = 0; l < Layers; l++) _layerTotals[l] *= factor;
        }
    }

    public bool HasData(int layer)
    {
        lock (_gate) return (uint)layer < (uint)Layers && _layerTotals[layer] > 0;
    }

    public bool HasAnyData
    {
        get { lock (_gate) return _layerTotals.Any(t => t > 0); }
    }

    /// <summary>
    /// Probability-like share of routed picks that went to <paramref name="expert"/>
    /// in <paramref name="layer"/> (shares in a layer sum to 1). Uniform when no data.
    /// </summary>
    public double Share(int layer, int expert)
    {
        if ((uint)layer >= (uint)Layers || (uint)expert >= (uint)Experts) return 0;
        lock (_gate)
        {
            var total = _layerTotals[layer];
            return total > 0 ? _counts[layer * Experts + expert] / total : 1.0 / Experts;
        }
    }

    public double[] LayerShares(int layer)
    {
        var shares = new double[Experts];
        for (var e = 0; e < Experts; e++) shares[e] = Share(layer, e);
        return shares;
    }

    /// <summary>
    /// Fraction of routing mass captured by the top <paramref name="topFraction"/>
    /// of experts, averaged over layers with data — a skew measure (0.2 → "top 20%
    /// of experts receive X% of tokens"). Returns <paramref name="topFraction"/> for uniform routing.
    /// </summary>
    public double Concentration(double topFraction = 0.2)
    {
        var k = Math.Max(1, (int)Math.Round(Experts * topFraction));
        double sum = 0;
        var n = 0;
        for (var l = 0; l < Layers; l++)
        {
            if (!HasData(l)) continue;
            sum += LayerShares(l).OrderByDescending(x => x).Take(k).Sum();
            n++;
        }
        return n == 0 ? (double)k / Experts : sum / n;
    }

    public ExpertUsageProfile Clone()
    {
        var c = new ExpertUsageProfile(Layers, Experts);
        lock (_gate)
        {
            Array.Copy(_counts, c._counts, _counts.Length);
            Array.Copy(_layerTotals, c._layerTotals, _layerTotals.Length);
            c._picksPerToken = _picksPerToken;
            c.UpdatedAt = UpdatedAt;
        }
        return c;
    }

    private const uint Magic = 0x50555848; // "HXUP"

    /// <summary>Compact binary form (Brotli-compressed float matrix).</summary>
    public byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using (var brotli = new BrotliStream(ms, CompressionLevel.Fastest, leaveOpen: true))
        using (var w = new BinaryWriter(brotli))
        {
            lock (_gate)
            {
                w.Write(Magic);
                w.Write(Layers);
                w.Write(Experts);
                w.Write(_picksPerToken);
                foreach (var c in _counts) w.Write(c);
            }
        }
        return ms.ToArray();
    }

    public static ExpertUsageProfile? Deserialize(byte[]? data)
    {
        if (data is null || data.Length == 0) return null;
        try
        {
            using var ms = new MemoryStream(data);
            using var brotli = new BrotliStream(ms, CompressionMode.Decompress);
            using var r = new BinaryReader(brotli);
            if (r.ReadUInt32() != Magic) return null;
            var layers = r.ReadInt32();
            var experts = r.ReadInt32();
            if (layers <= 0 || experts <= 0 || (long)layers * experts > 10_000_000) return null;
            var p = new ExpertUsageProfile(layers, experts) { _picksPerToken = Math.Max(1, r.ReadInt32()) };
            for (var l = 0; l < layers; l++)
            {
                double total = 0;
                for (var e = 0; e < experts; e++)
                {
                    var v = r.ReadSingle();
                    p._counts[l * experts + e] = v;
                    total += v;
                }
                p._layerTotals[l] = total;
            }
            return p;
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or EndOfStreamException)
        {
            return null;
        }
    }
}
