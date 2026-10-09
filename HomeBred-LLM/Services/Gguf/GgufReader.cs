using System.Text;

namespace HomebredLLM.Services.Gguf;

/// <summary>GGUF metadata value type tags, as written in the file.</summary>
public enum GgufValueType : uint
{
    UInt8 = 0, Int8 = 1, UInt16 = 2, Int16 = 3, UInt32 = 4, Int32 = 5,
    Float32 = 6, Bool = 7, String = 8, Array = 9, UInt64 = 10, Int64 = 11, Float64 = 12,
}

/// <summary>
/// ggml tensor element type (quantization scheme). Mirrors llama.cpp's
/// <c>enum ggml_type</c>. Only the tag is needed for metadata purposes —
/// dequantization is handled by the native llama.cpp library at load time.
/// </summary>
public enum GgmlType : uint
{
    F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3,
    Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9,
    Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, Q8_K = 15,
    IQ2_XXS = 16, IQ2_XS = 17, IQ3_XXS = 18, IQ1_S = 19, IQ4_NL = 20,
    IQ3_S = 21, IQ2_S = 22, IQ4_XS = 23,
    I8 = 24, I16 = 25, I32 = 26, I64 = 27, F64 = 28, IQ1_M = 29, BF16 = 30,
    TQ1_0 = 34, TQ2_0 = 35, MXFP4 = 39,
}

/// <summary>One entry in a GGUF file's tensor directory (name/shape/type/offset only — no data).</summary>
public sealed record GgufTensorInfo(string Name, uint[] Shape, GgmlType Type, ulong RelativeOffset)
{
    public long ElementCount => Shape.Length == 0 ? 0 : Shape.Aggregate(1L, (acc, d) => acc * d);
}

/// <summary>Parsed contents of a GGUF v2/v3 file, excluding tensor weight data.</summary>
public sealed record GgufFile(
    uint Version,
    ulong TensorCount,
    ulong MetadataKvCount,
    IReadOnlyDictionary<string, object?> Metadata,
    IReadOnlyList<GgufTensorInfo> Tensors,
    uint Alignment,
    long TensorDataStartOffset)
{
    public T? Get<T>(string key) => Metadata.TryGetValue(key, out var v) && v is T t ? t : default;
    public string? GetString(string key) => Get<string>(key);

    public long? GetInt(string key) =>
        Metadata.TryGetValue(key, out var v) && v is not null and not List<object?>
            ? Convert.ToInt64(v)
            : null;
}

/// <summary>
/// Pure C# reader for the GGUF binary model format used by llama.cpp / ggml,
/// versions 2 and 3. It reads only the header, key-value metadata block and
/// tensor directory — it never loads tensor weight data — so it's cheap
/// enough to run against every file in the Model Library.
///
/// Actual weight loading and inference is handled separately by
/// LlamaCppInferenceService via LLamaSharp, which uses llama.cpp's own
/// (native, C++) GGUF loader; this class exists purely for fast metadata
/// extraction and import-time validation.
///
/// GGUF v1 (which used 32-bit tensor/kv counts, string lengths and array
/// lengths instead of 64-bit) is intentionally not supported: essentially
/// every GGUF file produced since late 2023 is v2 or v3, and the two
/// versions share an identical binary layout for everything read here.
/// </summary>
public static class GgufReader
{
    private static readonly byte[] Magic = "GGUF"u8.ToArray();

    public static async Task<GgufFile> ReadAsync(string path, CancellationToken ct = default)
    {
        await using var fs = new FileStream(
            path, FileMode.Open, FileAccess.Read, FileShare.Read, bufferSize: 1 << 16, useAsync: true);
        return await Task.Run(() => Read(fs), ct);
    }

    public static GgufFile Read(Stream stream)
    {
        using var br = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);

        var magic = br.ReadBytes(4);
        if (magic.Length < 4 || !magic.AsSpan().SequenceEqual(Magic))
            throw new InvalidDataException("Not a GGUF file (missing 'GGUF' magic header).");

        var version = br.ReadUInt32();
        if (version is not (2 or 3))
            throw new NotSupportedException(
                $"Unsupported GGUF version {version}. Only GGUF v2 and v3 are supported.");

        var tensorCount = br.ReadUInt64();
        var kvCount = br.ReadUInt64();

        if (tensorCount > 5_000_000 || kvCount > 5_000_000)
            throw new InvalidDataException("GGUF header counts are implausibly large; file may be corrupt.");

        var metadata = new Dictionary<string, object?>(StringComparer.Ordinal);
        for (ulong i = 0; i < kvCount; i++)
        {
            var key = ReadGgufString(br);
            metadata[key] = ReadValue(br);
        }

        var tensors = new List<GgufTensorInfo>((int)Math.Min(tensorCount, 100_000));
        for (ulong i = 0; i < tensorCount; i++)
        {
            var name = ReadGgufString(br);
            var nDims = br.ReadUInt32();
            if (nDims > 8)
                throw new InvalidDataException($"Tensor '{name}' reports {nDims} dimensions; file may be corrupt.");

            var shape = new uint[nDims];
            for (var d = 0; d < nDims; d++)
                shape[d] = checked((uint)br.ReadUInt64());

            var ggmlType = (GgmlType)br.ReadUInt32();
            var offset = br.ReadUInt64();
            tensors.Add(new GgufTensorInfo(name, shape, ggmlType, offset));
        }

        var alignment = metadata.TryGetValue("general.alignment", out var a) && a is not null
            ? Convert.ToUInt32(a)
            : 32u;
        if (alignment == 0) alignment = 32;

        var pos = stream.Position;
        var dataStart = (long)((ulong)pos + alignment - 1) / alignment * alignment;

        return new GgufFile(version, tensorCount, kvCount, metadata, tensors, alignment, dataStart);
    }

    private static string ReadGgufString(BinaryReader br)
    {
        var len = br.ReadUInt64();
        if (len > 10_000_000)
            throw new InvalidDataException("GGUF string length is implausibly large; file may be corrupt.");
        var bytes = br.ReadBytes((int)len);
        return Encoding.UTF8.GetString(bytes);
    }

    private static object? ReadValue(BinaryReader br) => ReadValue(br, (GgufValueType)br.ReadUInt32());

    private static object? ReadValue(BinaryReader br, GgufValueType type) => type switch
    {
        GgufValueType.UInt8 => br.ReadByte(),
        GgufValueType.Int8 => br.ReadSByte(),
        GgufValueType.UInt16 => br.ReadUInt16(),
        GgufValueType.Int16 => br.ReadInt16(),
        GgufValueType.UInt32 => br.ReadUInt32(),
        GgufValueType.Int32 => br.ReadInt32(),
        GgufValueType.Float32 => br.ReadSingle(),
        GgufValueType.Bool => br.ReadByte() != 0,
        GgufValueType.String => ReadGgufString(br),
        GgufValueType.UInt64 => br.ReadUInt64(),
        GgufValueType.Int64 => br.ReadInt64(),
        GgufValueType.Float64 => br.ReadDouble(),
        GgufValueType.Array => ReadArray(br),
        _ => throw new InvalidDataException($"Unknown GGUF value type tag {(uint)type}; file may be corrupt."),
    };

    private static List<object?> ReadArray(BinaryReader br)
    {
        var elementType = (GgufValueType)br.ReadUInt32();
        var len = br.ReadUInt64();
        if (len > 5_000_000)
            throw new InvalidDataException("GGUF array length is implausibly large; file may be corrupt.");

        var list = new List<object?>((int)Math.Min(len, 1_000_000));
        for (ulong i = 0; i < len; i++)
            list.Add(ReadValue(br, elementType));
        return list;
    }
}
