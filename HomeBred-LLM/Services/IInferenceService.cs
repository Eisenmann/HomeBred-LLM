using HomebredLLM.Models;

namespace HomebredLLM.Services;

public record InferenceRequest(
    Guid ModelId,
    string ModelPath,
    ModelConfiguration Config,
    IReadOnlyList<(MessageRole Role, string Content)> Messages);

public record InferenceStats(
    float TokensPerSecond,
    float TimeToFirstTokenMs,
    float TotalMs,
    int PromptTokens,
    int OutputTokens);

public interface IInferenceService
{
    bool IsLoaded(Guid modelId);
    Task LoadAsync(Guid modelId, string modelPath, ModelConfiguration config, IProgress<string>? progress = null);
    void Unload(Guid modelId);
    void UnloadAll();

    /// <summary>Streams tokens. Yields each text fragment as it is generated. Returns final stats.</summary>
    IAsyncEnumerable<string> ChatStreamAsync(
        InferenceRequest request,
        Action<InferenceStats> onDone,
        CancellationToken ct = default);

    /// <summary>Returns one embedding vector per input string, in order.</summary>
    Task<IReadOnlyList<float[]>> GetEmbeddingsAsync(
        Guid modelId, IReadOnlyList<string> inputs, CancellationToken ct = default);
}
