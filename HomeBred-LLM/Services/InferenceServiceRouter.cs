using System.Collections.Concurrent;
using HomebredLLM.Data;
using HomebredLLM.Models;
using Microsoft.EntityFrameworkCore;

namespace HomebredLLM.Services;

/// <summary>
/// Dispatches <see cref="IInferenceService"/> calls to <see cref="OnnxRuntimeService"/>
/// or <see cref="LlamaCppInferenceService"/> depending on the target model's
/// <c>LocalModel.Format</c>. Registered as the app's single <see cref="IInferenceService"/>
/// so ViewModels don't need to know which engine backs a given model.
/// </summary>
public sealed class InferenceServiceRouter(
    IDbContextFactory<AppDbContext> dbFactory,
    OnnxRuntimeService onnx,
    LlamaCppInferenceService llama) : IInferenceService, IDisposable
{
    private readonly ConcurrentDictionary<Guid, ModelFormat> _formatCache = new();

    // Used for calls that only carry a modelId (IsLoaded/SupportsVision/Unload):
    // trust the cache populated by LoadAsync; if we've never loaded this model
    // in this process, neither engine has it loaded, so the choice is moot for
    // "is it loaded" queries and Unload is a safe no-op on the wrong engine.
    private IInferenceService CachedOrDefault(Guid modelId) =>
        _formatCache.TryGetValue(modelId, out var f) && f == ModelFormat.Gguf ? llama : onnx;

    private async Task<IInferenceService> ResolveForLoadAsync(Guid modelId, string modelPath)
    {
        var format = modelPath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)
            ? ModelFormat.Gguf
            : ModelFormat.Onnx;

        // The file extension is normally decisive, but fall back to the DB
        // record (set at import time) in case a model was registered from a
        // path without a recognizable extension.
        if (format == ModelFormat.Onnx && !Directory.Exists(modelPath) && !File.Exists(modelPath))
        {
            await using var db = await dbFactory.CreateDbContextAsync();
            var model = await db.Models.AsNoTracking().FirstOrDefaultAsync(m => m.Id == modelId);
            if (model is not null) format = model.Format;
        }

        _formatCache[modelId] = format;
        return format == ModelFormat.Gguf ? llama : onnx;
    }

    public bool IsLoaded(Guid modelId) => CachedOrDefault(modelId).IsLoaded(modelId);
    public bool SupportsVision(Guid modelId) => CachedOrDefault(modelId).SupportsVision(modelId);

    public async Task LoadAsync(Guid modelId, string modelPath, ModelConfiguration config,
        string? mmprojPath = null, IReadOnlyList<LoraSpec>? loraAdapters = null,
        IProgress<string>? progress = null)
    {
        var svc = await ResolveForLoadAsync(modelId, modelPath);
        await svc.LoadAsync(modelId, modelPath, config, mmprojPath, loraAdapters, progress);
    }

    public void Unload(Guid modelId)
    {
        // Unload on both — whichever one doesn't actually have it loaded is a no-op.
        onnx.Unload(modelId);
        llama.Unload(modelId);
        _formatCache.TryRemove(modelId, out _);
    }

    public void UnloadAll()
    {
        onnx.UnloadAll();
        llama.UnloadAll();
        _formatCache.Clear();
    }

    public IAsyncEnumerable<string> ChatStreamAsync(
        InferenceRequest request, Action<InferenceStats> onDone, CancellationToken ct = default)
        => CachedOrDefault(request.ModelId).ChatStreamAsync(request, onDone, ct);

    public Task<IReadOnlyList<float[]>> GetEmbeddingsAsync(
        Guid modelId, IReadOnlyList<string> inputs, CancellationToken ct = default)
        => CachedOrDefault(modelId).GetEmbeddingsAsync(modelId, inputs, ct);

    public void Dispose()
    {
        onnx.Dispose();
        llama.Dispose();
    }
}
