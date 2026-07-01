using HomebredLLM.Models;
using LLama;
using LLama.Common;
using LLama.Native;
using LLama.Sampling;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;

namespace HomebredLLM.Services;

/// <summary>
/// In-process LLM inference via LLamaSharp (llama.cpp P/Invoke bindings).
/// Each loaded model occupies a slot; inference runs on a background thread
/// so the WPF UI thread is never blocked.
/// </summary>
public sealed class LlamaSharpService : IInferenceService, IDisposable
{
    private sealed class LoadedModel(LLamaWeights weights, LLamaContext context, string modelPath, ModelConfiguration config)
    {
        public LLamaWeights Weights { get; } = weights;
        public LLamaContext Context { get; } = context;
        public string ModelPath { get; } = modelPath;
        public ModelConfiguration Config { get; } = config;
        public SemaphoreSlim ChatLock { get; } = new(1, 1);

        // Set when the model has a companion mmproj (vision projector) file.
        public MtmdWeights? Mtmd;
        public MtmdContextParams? MtmdParams;

        // The embedder needs its own context (created with Embeddings=true), so it's
        // built lazily on first use rather than at load time — most models are only
        // ever used for chat.
        public LLamaEmbedder? Embedder;
        public SemaphoreSlim EmbedderInitLock { get; } = new(1, 1);
        public SemaphoreSlim EmbedLock { get; } = new(1, 1);

        public void Dispose()
        {
            Embedder?.Dispose();
            Mtmd?.Dispose();
            Context.Dispose();
            Weights.Dispose();
            ChatLock.Dispose();
            EmbedderInitLock.Dispose();
            EmbedLock.Dispose();
        }
    }

    private readonly ConcurrentDictionary<Guid, LoadedModel> _loaded = new();

    public bool IsLoaded(Guid modelId) => _loaded.ContainsKey(modelId);

    public bool SupportsVision(Guid modelId) =>
        _loaded.TryGetValue(modelId, out var m) && m.Mtmd?.SupportsVision == true;

    public async Task LoadAsync(Guid modelId, string modelPath, ModelConfiguration config, string? mmprojPath = null, IProgress<string>? progress = null)
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

        var loaded = new LoadedModel(weights, context, modelPath, config);

        if (mmprojPath is not null && File.Exists(mmprojPath))
        {
            progress?.Report("Loading vision projector…");
            var mtmdParams = MtmdContextParams.Default();
            mtmdParams.UseGpu = config.GpuLayerCount != 0;
            loaded.Mtmd = await MtmdWeights.LoadFromFileAsync(mmprojPath, weights, mtmdParams);
            loaded.MtmdParams = mtmdParams;
        }

        _loaded[modelId] = loaded;
        progress?.Report("Model loaded.");
    }

    public void Unload(Guid modelId)
    {
        if (_loaded.TryRemove(modelId, out var m))
        {
            m.Dispose();
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

        if (request.Messages.Count == 0)
            throw new InvalidOperationException("No message to send.");

        await loaded.ChatLock.WaitAsync(ct);
        try
        {
            var executor = loaded.Mtmd is not null
                ? new InteractiveExecutor(loaded.Context, loaded.Mtmd)
                : new InteractiveExecutor(loaded.Context);

            // The current turn (last message) is passed to ChatAsync below, which
            // appends it to the session history itself — so history must exclude it
            // here, or the model sees the current turn twice. Every message (not just
            // the current turn) is re-run through BuildEffectiveContent because this
            // method re-tokenizes the whole conversation from text on every call —
            // there's no persisted KV cache across calls — so a prior turn's image
            // attachments must be re-queued via Mtmd.LoadMedia every time too.
            var priorMessages = request.Messages.Take(request.Messages.Count - 1);
            var currentTurn = request.Messages[^1];

            var chatHistory = new ChatHistory();
            foreach (var (role, content, attachments) in priorMessages)
            {
                chatHistory.AddMessage(role switch
                {
                    MessageRole.User => AuthorRole.User,
                    MessageRole.Assistant => AuthorRole.Assistant,
                    MessageRole.System => AuthorRole.System,
                    _ => AuthorRole.User,
                }, BuildEffectiveContent(loaded, content, attachments));
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

            var currentTurnContent = BuildEffectiveContent(loaded, currentTurn.Content, currentTurn.Attachments);
            var promptText = string.Join("\n", request.Messages.Select(m => m.Content));
            var promptTokens = loaded.Context.Tokenize(promptText, addBos: true, special: true).Length;

            var sw = Stopwatch.StartNew();
            float? ttft = null;
            int outputTokens = 0;

            await foreach (var text in session.ChatAsync(
                new ChatHistory.Message(AuthorRole.User, currentTurnContent),
                inferenceParams,
                ct))
            {
                if (string.IsNullOrEmpty(text)) continue;
                if (ttft is null)
                    ttft = (float)sw.Elapsed.TotalMilliseconds;

                outputTokens++;
                yield return text;
            }

            sw.Stop();
            var totalMs = (float)sw.Elapsed.TotalMilliseconds;
            var tps = totalMs > 0 ? outputTokens / (totalMs / 1000f) : 0;

            onDone(new InferenceStats(tps, ttft ?? 0, totalMs, promptTokens, outputTokens));
        }
        finally
        {
            loaded.ChatLock.Release();
        }
    }

    private const int MaxAttachmentTextChars = 200_000;

    // Queues any image attachments into the Mtmd pending-media buffer (consumed by the
    // executor's own tokenizer the next time it processes this text) and inlines any
    // text attachments directly into the message content.
    private static string BuildEffectiveContent(LoadedModel loaded, string content, IReadOnlyList<InferenceAttachment> attachments)
    {
        if (attachments.Count == 0) return content;

        var sb = new StringBuilder();

        var images = attachments.Where(a => a.Kind == AttachmentKind.Image).ToList();
        if (images.Count > 0)
        {
            if (loaded.Mtmd is null)
                throw new InvalidOperationException("This model has no vision projector loaded — it can't accept image attachments.");

            foreach (var image in images)
            {
                loaded.Mtmd.LoadMedia(image.Path);
                sb.Append(loaded.MtmdParams!.MediaMarker).Append('\n');
            }
        }

        foreach (var file in attachments.Where(a => a.Kind == AttachmentKind.Text))
        {
            sb.Append("\n[Attached: ").Append(Path.GetFileName(file.Path)).Append("]\n```\n")
              .Append(ReadTextAttachment(file.Path)).Append("\n```\n");
        }

        sb.Append(content);
        return sb.ToString();
    }

    private static string ReadTextAttachment(string path)
    {
        try
        {
            var text = File.ReadAllText(path);
            return text.Length > MaxAttachmentTextChars
                ? text[..MaxAttachmentTextChars] + "\n…(truncated)"
                : text;
        }
        catch (Exception ex)
        {
            return $"(failed to read attachment: {ex.Message})";
        }
    }

    public async Task<IReadOnlyList<float[]>> GetEmbeddingsAsync(
        Guid modelId, IReadOnlyList<string> inputs, CancellationToken ct = default)
    {
        if (!_loaded.TryGetValue(modelId, out var loaded))
            throw new InvalidOperationException("Model is not loaded. Call LoadAsync first.");

        var embedder = await GetOrCreateEmbedderAsync(loaded, ct);

        await loaded.EmbedLock.WaitAsync(ct);
        try
        {
            var results = new List<float[]>(inputs.Count);
            foreach (var input in inputs)
            {
                var vectors = await embedder.GetEmbeddings(input, ct);
                results.Add(vectors.Count > 0 ? vectors[0] : []);
            }
            return results;
        }
        finally
        {
            loaded.EmbedLock.Release();
        }
    }

    private static async Task<LLamaEmbedder> GetOrCreateEmbedderAsync(LoadedModel loaded, CancellationToken ct)
    {
        if (loaded.Embedder is not null) return loaded.Embedder;

        await loaded.EmbedderInitLock.WaitAsync(ct);
        try
        {
            loaded.Embedder ??= await Task.Run(() =>
            {
                var embedParams = new ModelParams(loaded.ModelPath)
                {
                    ContextSize = (uint)loaded.Config.ContextSize,
                    GpuLayerCount = loaded.Config.GpuLayerCount,
                    Embeddings = true,
                };
                return new LLamaEmbedder(loaded.Weights, embedParams);
            }, ct);
        }
        finally
        {
            loaded.EmbedderInitLock.Release();
        }

        return loaded.Embedder;
    }

    public void Dispose() => UnloadAll();
}
