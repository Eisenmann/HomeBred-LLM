using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;
using HomebredLLM.Models;
using Microsoft.ML.OnnxRuntimeGenAI;

namespace HomebredLLM.Services;

public sealed class OnnxRuntimeService : IInferenceService, IDisposable
{
    private sealed class LoadedModel : IDisposable
    {
        public Model Model { get; set; } = null!;
        public Tokenizer Tokenizer { get; set; } = null!;
        public ModelConfiguration ConfigSnapshot { get; set; } = null!;
        public SemaphoreSlim ChatLock { get; } = new(1, 1);

        public void Dispose()
        {
            Tokenizer?.Dispose();
            Model?.Dispose();
            ChatLock?.Dispose();
        }
    }

    private readonly ConcurrentDictionary<Guid, LoadedModel> _loaded = new();

    public bool IsLoaded(Guid modelId) => _loaded.ContainsKey(modelId);

    public bool SupportsVision(Guid modelId) => false;

    public async Task LoadAsync(Guid modelId, string modelPath, ModelConfiguration config,
        string? mmprojPath = null, IReadOnlyList<LoraSpec>? loraAdapters = null,
        IProgress<string>? progress = null)
    {
        if (_loaded.ContainsKey(modelId)) return;

        progress?.Report("Loading ONNX model...");

        var modelDir = File.Exists(modelPath) ? Path.GetDirectoryName(modelPath)! : modelPath;
        if (!Directory.Exists(modelDir))
            throw new DirectoryNotFoundException($"Model directory not found: {modelDir}");

        if (!File.Exists(Path.Combine(modelDir, "model.onnx")) &&
            !File.Exists(Path.Combine(modelDir, "model.onnx.data")))
            throw new InvalidDataException(
                "No model.onnx or model.onnx.data found in the model directory.");

        await Task.Run(() =>
        {
            try
            {
                var model = new Model(modelDir);
                var tokenizer = new Tokenizer(model);

                _loaded[modelId] = new LoadedModel
                {
                    Model = model,
                    Tokenizer = tokenizer,
                    ConfigSnapshot = config,
                };
                progress?.Report("Model loaded.");
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException(
                    $"Failed to load ONNX model from {modelDir}. " +
                    $"Ensure the directory contains a valid ONNX export. " +
                    $"Error: {ex.Message}", ex);
            }
        });
    }

    public void Unload(Guid modelId)
    {
        if (_loaded.TryRemove(modelId, out var m))
            m.Dispose();
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
            var cfg = request.Config;
            var prompt = BuildChatPrompt(request.Messages, cfg.SystemPrompt);

            var sw = Stopwatch.StartNew();
            float? ttft = null;
            int outputTokens = 0;

            var sequences = loaded.Tokenizer.Encode(prompt);

            using var generatorParams = new GeneratorParams(loaded.Model);
            generatorParams.SetSearchOption("max_length", cfg.MaxTokens);
            generatorParams.SetSearchOption("temperature", cfg.Temperature);
            generatorParams.SetSearchOption("top_p", cfg.TopP);
            generatorParams.SetSearchOption("top_k", cfg.TopK);
            generatorParams.SetSearchOption("repetition_penalty", cfg.RepeatPenalty);

            using var tokenizerStream = loaded.Tokenizer.CreateStream();
            using var generator = new Generator(loaded.Model, generatorParams);
            generator.AppendTokenSequences(sequences);
            var promptTokens = (int)generator.TokenCount();

            while (!generator.IsDone())
            {
                ct.ThrowIfCancellationRequested();
                generator.GenerateNextToken();

                var token = generator.GetSequence(0)[^1];
                var text = tokenizerStream.Decode(token);

                if (!string.IsNullOrEmpty(text))
                {
                    if (ttft is null)
                        ttft = (float)sw.Elapsed.TotalMilliseconds;
                    outputTokens++;
                    yield return text;
                }
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

    private static string BuildChatPrompt(
        IReadOnlyList<(MessageRole Role, string Content, IReadOnlyList<InferenceAttachment> Attachments)> messages,
        string systemPrompt)
    {
        var sb = new StringBuilder();

        if (!string.IsNullOrWhiteSpace(systemPrompt))
            sb.Append("<|im_start|>system\n").Append(systemPrompt).Append("<|im_end|>\n");

        foreach (var (role, content, _) in messages)
        {
            var tag = role switch
            {
                MessageRole.User => "<|im_start|>user\n",
                MessageRole.Assistant => "<|im_start|>assistant\n",
                MessageRole.System => "<|im_start|>system\n",
                _ => "<|im_start|>user\n",
            };
            sb.Append(tag).Append(content).Append("<|im_end|>\n");
        }

        sb.Append("<|im_start|>assistant\n");
        return sb.ToString();
    }

    public Task<IReadOnlyList<float[]>> GetEmbeddingsAsync(
        Guid modelId, IReadOnlyList<string> inputs, CancellationToken ct = default)
        => throw new NotSupportedException(
            "Embeddings are not supported by the ONNX Runtime GenAI inference service.");

    public void Dispose() => UnloadAll();
}