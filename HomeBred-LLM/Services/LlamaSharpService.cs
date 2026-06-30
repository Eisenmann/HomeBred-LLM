using HomebredLLM.Models;
using LLama;
using LLama.Common;
using LLama.Sampling;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace HomebredLLM.Services;

/// <summary>
/// In-process LLM inference via LLamaSharp (llama.cpp P/Invoke bindings).
/// Each loaded model occupies a slot; inference runs on a background thread
/// so the WPF UI thread is never blocked.
/// </summary>
public sealed class LlamaSharpService : IInferenceService, IDisposable
{
    private sealed record LoadedModel(LLamaWeights Weights, LLamaContext Context, SemaphoreSlim Lock);
    private readonly ConcurrentDictionary<Guid, LoadedModel> _loaded = new();

    public bool IsLoaded(Guid modelId) => _loaded.ContainsKey(modelId);

    public async Task LoadAsync(Guid modelId, string modelPath, ModelConfiguration config, IProgress<string>? progress = null)
    {
        if (_loaded.ContainsKey(modelId)) return;

        progress?.Report("Loading model weights…");

        var parameters = new ModelParams(modelPath)
        {
            ContextSize = (uint)config.ContextSize,
            GpuLayerCount = config.GpuLayerCount,
            BatchSize = (uint)config.BatchSize,
            Threads = (int?)config.ThreadCount,
            FlashAttention = true,
        };

        // Loading is CPU/disk-intensive — run off the UI thread
        var (weights, context) = await Task.Run(() =>
        {
            var w = LLamaWeights.LoadFromFile(parameters);
            var c = w.CreateContext(parameters);
            return (w, c);
        });

        _loaded[modelId] = new LoadedModel(weights, context, new SemaphoreSlim(1, 1));
        progress?.Report("Model loaded.");
    }

    public void Unload(Guid modelId)
    {
        if (_loaded.TryRemove(modelId, out var m))
        {
            m.Context.Dispose();
            m.Weights.Dispose();
            m.Lock.Dispose();
        }
    }

    public void UnloadAll()
    {
        foreach (var id in _loaded.Keys.ToArray())
            Unload(id);
    }

    public async IAsyncEnumerable<string> ChatStreamAsync(
        InferenceRequest request,
        Action<InferenceStats> onDone,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        if (!_loaded.TryGetValue(request.ModelId, out var loaded))
            throw new InvalidOperationException("Model is not loaded. Call LoadAsync first.");

        await loaded.Lock.WaitAsync(ct);
        try
        {
            var executor = new InteractiveExecutor(loaded.Context);

            // Build prompt from message history
            var chatHistory = new ChatHistory();
            foreach (var (role, content) in request.Messages)
            {
                chatHistory.AddMessage(role switch
                {
                    MessageRole.User => AuthorRole.User,
                    MessageRole.Assistant => AuthorRole.Assistant,
                    MessageRole.System => AuthorRole.System,
                    _ => AuthorRole.User,
                }, content);
            }

            var cfg = request.Config;
            var inferenceParams = new InferenceParams
            {
                MaxTokens = cfg.MaxTokens,
                SamplingPipeline = new DefaultSamplingPipeline
                {
                    Temperature = cfg.Temperature,
                    TopP = cfg.TopP,
                    TopK = cfg.TopK,
                    RepeatPenalty = cfg.RepeatPenalty,
                },
            };

            var session = new LLama.ChatSession(executor, chatHistory);

            var sw = Stopwatch.StartNew();
            float? ttft = null;
            int outputTokens = 0;
            string fullReply = "";

            // Build the last user message as the prompt
            var lastUser = request.Messages.LastOrDefault(m => m.Role == MessageRole.User);
            var prompt = lastUser.Content ?? "";

            await foreach (var text in session.ChatAsync(
                new ChatHistory.Message(AuthorRole.User, prompt),
                inferenceParams,
                ct))
            {
                if (string.IsNullOrEmpty(text)) continue;
                if (ttft is null)
                    ttft = (float)sw.Elapsed.TotalMilliseconds;

                outputTokens++;
                fullReply += text;
                yield return text;
            }

            sw.Stop();
            var totalMs = (float)sw.Elapsed.TotalMilliseconds;
            var tps = totalMs > 0 ? outputTokens / (totalMs / 1000f) : 0;

            onDone(new InferenceStats(tps, ttft ?? 0, totalMs, 0, outputTokens));
        }
        finally
        {
            loaded.Lock.Release();
        }
    }

    public void Dispose() => UnloadAll();
}
