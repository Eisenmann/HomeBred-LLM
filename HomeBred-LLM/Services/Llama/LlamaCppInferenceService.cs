using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;
using HomebredLLM.Models;
using System.Globalization;
using System.Text.RegularExpressions;
using HomebredLLM.Services.Gguf;
using HomebredLLM.Services.Tiering;
using LLama;
using LLama.Common;
using LLama.Sampling;

namespace HomebredLLM.Services;

/// <summary>
/// GGUF inference backed by llama.cpp via the LLamaSharp managed bindings
/// (NuGet: LLamaSharp + a native backend package — LLamaSharp.Backend.Cpu,
/// .Cuda12, .Vulkan, or .Metal — see HomeBred-LLM.csproj). This runs
/// alongside OnnxRuntimeService; <see cref="InferenceServiceRouter"/> routes
/// a model here when its <c>LocalModel.Format</c> is <c>ModelFormat.Gguf</c>.
///
/// NOTE ON API SURFACE: LLamaSharp's public API has moved between minor
/// versions (executor/session shape, sampling pipeline types, embedder
/// construction). This file was build-verified against LLamaSharp 0.21.0
/// (pinned in the csproj) — build was clean (0 errors). Specifics that
/// were confirmed valid in this version: DefaultSamplingPipeline with
/// Temperature/TopP/TopK/RepeatPenalty properties, InferenceParams.
/// AntiPrompts taking a string[], new InteractiveExecutor(context),
/// LLamaWeights.LoadFromFile(ModelParams), and
/// new LLamaEmbedder(LLamaWeights, ModelParams) with a ContextSize.
/// LLamaSharp 0.21.0 does NOT expose LoRA in its managed API (no
/// ModelParams.LoraAdapters property). If you bump that version, re-check
/// these against LLamaSharp's changelog.
/// </summary>
public sealed class LlamaCppInferenceService : IInferenceService, IDisposable
{
    private sealed class LoadedModel : IDisposable
    {
        public required LLamaWeights Weights { get; init; }
        public required LLamaContext Context { get; init; }
        public required InteractiveExecutor Executor { get; init; }
        public LLamaEmbedder? Embedder { get; set; }
        public SemaphoreSlim ChatLock { get; } = new(1, 1);

        public void Dispose()
        {
            Embedder?.Dispose();
            Context.Dispose();
            Weights.Dispose();
            ChatLock.Dispose();
        }
    }

    private readonly ConcurrentDictionary<Guid, LoadedModel> _loaded = new();
    private readonly ConcurrentDictionary<Guid, bool> _vision = new();

    /// <summary>
    /// Captures llama.cpp's native log lines (real tensor names, architecture
    /// strings, "expected X got Y" dims) that the native loader prints instead
    /// of putting into the managed exception. Folded into the load-failure
    /// exception below so the running app can actually see why a load failed.
    /// Registered once at startup (see <see cref="NativeLogBuffer"/>).
    /// </summary>
    private readonly NativeLogBuffer _nativeLogBuffer;

    /// <summary>
    /// Tiered memory (docs/tiered-memory-architecture.md): plans VRAM/RAM/disk
    /// placement before load, then runs the warm tier, routing profiler and
    /// rebalancer for the loaded model.
    /// </summary>
    private readonly TieringCoordinator? _tiering;

    private readonly HardwareProbe? _hardware;

    public LlamaCppInferenceService(NativeLogBuffer nativeLogBuffer, TieringCoordinator tiering, HardwareProbe hardware)
    {
        _nativeLogBuffer = nativeLogBuffer;
        _tiering = tiering;
        _hardware = hardware;
    }

    /// <summary>Without tiering (legacy GPU-layer placement only) — used by tests and tools.</summary>
    public LlamaCppInferenceService(NativeLogBuffer nativeLogBuffer)
    {
        _nativeLogBuffer = nativeLogBuffer;
    }

    public bool IsLoaded(Guid modelId) => _loaded.ContainsKey(modelId);

    // mmproj (LLaVA-style vision projector) loading isn't wired into the
    // generation path here — only tracked so the UI can reflect whether a
    // vision-capable pairing was supplied at load time.
    public bool SupportsVision(Guid modelId) => _vision.TryGetValue(modelId, out var v) && v;

    public async Task LoadAsync(Guid modelId, string modelPath, ModelConfiguration config,
        string? mmprojPath = null, IReadOnlyList<LoraSpec>? loraAdapters = null,
        IProgress<string>? progress = null)
    {
        if (_loaded.ContainsKey(modelId)) return;
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"GGUF model file not found: {modelPath}", modelPath);

        progress?.Report("Validating tensor names...");
        await ValidateTensorNaming(modelPath);

        progress?.Report("Validating vocab size...");
        ValidateVocabSize(modelPath);

        progress?.Report("Validating attention dimensions...");
        ValidateAttentionDimensions(modelPath);

        PreparedLoad? tiered = null;
        try
        {
            if (_tiering is not null)
                tiered = await _tiering.PrepareAsync(modelId, modelPath, config, progress);
        }
        catch (InvalidOperationException)
        {
            throw; // "does not fit the budgets" — the user has to change settings
        }
        catch (Exception ex)
        {
            // Planning is an optimisation; never let it block a load that would work the legacy way.
            progress?.Report($"Tier planning skipped ({ex.Message}); using GPU layer count.");
        }

        progress?.Report("Loading GGUF model...");

        // Clear prior capture so the lines we read on failure belong to *this*
        // load attempt, not a previous one. Native lines only arrive while the
        // native loader runs (below), so clearing right before is correct.
        _nativeLogBuffer.Clear();

        await Task.Run(() =>
        {
            var parameters = new ModelParams(modelPath)
            {
                ContextSize = (uint)Math.Max(512, config.ContextSize),
                GpuLayerCount = config.GpuLayerCount,
                Threads = config.ThreadCount > 0 ? config.ThreadCount : null,
                BatchSize = (uint)Math.Max(1, config.BatchSize),
            };

            if (tiered is not null)
                LlamaPlacementApplier.Apply(parameters, tiered.Plan);

            if (loraAdapters is { Count: > 0 })
            {
                // NOTE: LLamaSharp 0.21.0 does not expose LoRA in its managed API
                // (no ModelParams.LoraAdapters property, no LoraAdapter type). The
                // adapters are accepted for interface parity with OnnxRuntimeService
                // (which likewise does not apply them) but are not applied here.
                // Wiring this up requires a newer LLamaSharp or calling llama.cpp
                // native APIs (llama_model_apply_lora_from_file) directly.
                progress?.Report(
                    $"Note: {loraAdapters.Count} LoRA adapter(s) supplied, but the " +
                    "pinned LLamaSharp backend does not expose LoRA in its managed API; " +
                    "they will not be applied to this GGUF model.");
            }

            try
            {
                progress?.Report("Reading weights...");
                var weights = LLamaWeights.LoadFromFile(parameters);
                var context = weights.CreateContext(parameters);
                var executor = new InteractiveExecutor(context);

                var loadedModel = new LoadedModel { Weights = weights, Context = context, Executor = executor };
                _loaded[modelId] = loadedModel;
                _vision[modelId] = !string.IsNullOrWhiteSpace(mmprojPath) && File.Exists(mmprojPath);

                if (tiered is not null && _tiering is not null)
                {
                    CalibrateComputeBuffer(tiered.Plan);
                    _tiering.OnLoaded(modelId, tiered, weights, parameters,
                        action => RunExclusiveAsync(loadedModel, action));
                }

                progress?.Report("Model loaded.");
            }
            catch (Exception ex)
            {
                var nativeLog = string.Join('\n', _nativeLogBuffer.TakeRecentLines());
                throw new InvalidOperationException(
                    $"Failed to load GGUF model from {modelPath}. " +
                    $"Ensure it's a valid GGUF v2/v3 file and that a native llama.cpp " +
                    $"backend package matching this machine (CPU/CUDA/Vulkan/Metal) is installed. " +
                    $"Error: {ex.Message}" +
                    (string.IsNullOrWhiteSpace(nativeLog) ? "" :
                        $"\n\nNative llama.cpp log (most recent lines):\n{nativeLog}"), ex);
            }
        });
    }

    /// <summary>Runs background work (routing profiling) only while no chat is generating.</summary>
    private static async Task RunExclusiveAsync(LoadedModel model, Func<Task> action)
    {
        await model.ChatLock.WaitAsync();
        try { await action(); }
        finally { model.ChatLock.Release(); }
    }

    private static readonly Regex ComputeBufferLine = new(
        @"(\S+) compute buffer size =\s*([0-9.]+) MiB", RegexOptions.Compiled);

    /// <summary>Compares llama.cpp's reported GPU compute buffers with the estimate to calibrate future plans.</summary>
    private void CalibrateComputeBuffer(PlacementPlan plan)
    {
        if (!plan.UsesGpu || plan.VramComputeBytes <= 0) return;
        double gpuMiB = 0;
        foreach (var line in _nativeLogBuffer.RecentLines())
        {
            var m = ComputeBufferLine.Match(line);
            if (!m.Success) continue;
            var device = m.Groups[1].Value;
            if (device.StartsWith("CPU", StringComparison.OrdinalIgnoreCase) ||
                device.EndsWith("_Host", StringComparison.OrdinalIgnoreCase)) continue;
            gpuMiB += double.Parse(m.Groups[2].Value, CultureInfo.InvariantCulture);
        }
        if (gpuMiB > 0 && _hardware is not null)
            _ = _hardware.RecordComputeBufferSampleAsync(gpuMiB * (1 << 20) / plan.VramComputeBytes);
    }

    public void Unload(Guid modelId)
    {
        _tiering?.OnUnloaded(modelId); // profiler context must go before the weights it uses
        if (_loaded.TryRemove(modelId, out var m))
            m.Dispose();
        _vision.TryRemove(modelId, out _);
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

            var inferenceParams = new InferenceParams
            {
                MaxTokens = cfg.MaxTokens,
                AntiPrompts = ["<|im_end|>", "<|eot_id|>", "</s>"],
                SamplingPipeline = new DefaultSamplingPipeline
                {
                    Temperature = cfg.Temperature,
                    TopP = cfg.TopP,
                    TopK = cfg.TopK,
                    RepeatPenalty = cfg.RepeatPenalty,
                },
            };

            var sw = Stopwatch.StartNew();
            float? ttft = null;
            var outputTokens = 0;
            var promptTokens = loaded.Context.Tokenize(prompt, addBos: true).Length;
            var completion = new StringBuilder();
            _tiering?.NotifyActivity(request.ModelId);

            await foreach (var text in loaded.Executor.InferAsync(prompt, inferenceParams, ct))
            {
                ct.ThrowIfCancellationRequested();
                if (string.IsNullOrEmpty(text)) continue;
                if (ttft is null) ttft = (float)sw.Elapsed.TotalMilliseconds;
                outputTokens++;
                completion.Append(text);
                yield return text;
            }

            sw.Stop();
            var totalMs = (float)sw.Elapsed.TotalMilliseconds;
            var tps = totalMs > 0 ? outputTokens / (totalMs / 1000f) : 0;
            var stats = new InferenceStats(tps, ttft ?? 0, totalMs, promptTokens, outputTokens);
            _tiering?.OnChatCompleted(request.ModelId, prompt + completion, stats);
            onDone(stats);
        }
        finally
        {
            loaded.ChatLock.Release();
        }
    }

    /// <summary>
    /// Cheap pre-flight check that runs before llama.cpp's native loader is
    /// touched. Reads the file's tensor directory with our pure-C# reader and
    /// rejects files whose tokenizer vocabulary count
    /// (tokenizer.ggml.tokens) does not match the number of rows in the
    /// token_embd.weight tensor. llama.cpp's native loader hard-fails such a
    /// file with check_tensor_dims, so this converts that opaque
    /// LoadWeightsFailedException into a clear, actionable message. A file
    /// with this mismatch is a data problem in the GGUF itself (commonly a
    /// Qwen-family embedding matrix padded with reserved token slots) — it
    /// cannot be repaired on this app's side, only regenerated with a
    /// corrected converter.
    ///
    /// The ground truth for the vocab dimension is the tensor directory's
    /// actual shape (Shape[1] is n_vocab in this reader's GGUF shape
    /// convention), NOT a tokenizer.ggml.* or *_vocab_size metadata field,
    /// which could itself be wrong or absent. A parse failure from
    /// <see cref="GgufReader"/> is deliberately NOT swallowed here — it
    /// surfaces as a separate, clearly-labeled InvalidDataException about a
    /// malformed file, consistent with <see cref="ValidateTensorNaming"/>.
    /// </summary>
    private static void ValidateVocabSize(string modelPath)
    {
        GgufFile gguf;
        try
        {
            gguf = GgufReader.Read(File.OpenRead(modelPath));
        }
        catch (Exception ex) when (ex is InvalidDataException or NotSupportedException)
        {
            throw new InvalidDataException(
                $"'{Path.GetFileName(modelPath)}' failed GGUF header/tensor-directory parsing " +
                $"while checking vocab size: {ex.GetType().Name}: {ex.Message}", ex);
        }

        // tokenizer.ggml.tokens — the actual tokenizer vocabulary array.
        var tokenCount = gguf.Metadata.TryGetValue("tokenizer.ggml.tokens", out var toks)
            && toks is List<object?> tokList
            ? tokList.Count
            : (int?)null;

        // token_embd.weight's actual tensor shape as read from the tensor
        // directory. Shape is stored row-major-per-this-reader: Shape[1] is
        // the vocab dimension (confirmed by gen_test_gguf.py and llama.cpp's
        // "expected 1024, 151643, got 1024, 151936" diagnostic).
        var embedTensor = gguf.Tensors.FirstOrDefault(t => t.Name == "token_embd.weight");
        long? embedRows = embedTensor is { Shape.Length: >= 2 } ? embedTensor.Shape[1] : null;

        if (tokenCount is null || embedRows is null)
            return; // can't check without both pieces present; let the native loader report its own error

        if (tokenCount != embedRows)
        {
            throw new InvalidDataException(
                $"'{Path.GetFileName(modelPath)}' has a tokenizer/embedding mismatch: " +
                $"tokenizer.ggml.tokens has {tokenCount} entries but token_embd.weight " +
                $"has {embedRows} rows. These must match exactly for llama.cpp to load " +
                $"this model. This is a data problem in the GGUF file itself (a common " +
                $"cause for Qwen-family models: the embedding matrix is padded with " +
                $"reserved token slots the tokenizer list is missing) — it needs to be " +
                $"fixed by regenerating the file with a corrected converter, not by " +
                $"anything on this app's side.");
        }
    }

    /// <summary>
    /// Mirrors llama.cpp's own GGML_ASSERT(n_embd_head == n_rot) invariant,
    /// checked here in managed code before the native loader ever runs.
    /// That assertion, if it fails natively, calls abort() and kills the
    /// whole process with no .NET exception and no chance to recover —
    /// there is no "catch" for it, so it must never be allowed to fire.
    /// </summary>
    private static void ValidateAttentionDimensions(string modelPath)
    {
        GgufFile gguf;
        try
        {
            using var stream = File.OpenRead(modelPath);
            gguf = GgufReader.Read(stream);
        }
        catch (Exception ex)
        {
            throw new InvalidDataException(
                $"Could not parse '{Path.GetFileName(modelPath)}' while checking " +
                $"attention dimensions: {ex.GetType().Name}: {ex.Message}", ex);
        }

        var arch = gguf.GetString("general.architecture");
        if (arch is null) return; // let the native loader report its own error

        var embeddingLength = gguf.GetInt($"{arch}.embedding_length");
        var headCount = gguf.GetInt($"{arch}.attention.head_count");
        var keyLength = gguf.GetInt($"{arch}.attention.key_length");
        var ropeDim = gguf.GetInt($"{arch}.rope.dimension_count");

        // n_embd_head: explicit key_length if present (this is exactly the
        // field some architectures, e.g. Qwen3, need because their head_dim
        // isn't embedding_length / head_count) -- otherwise the same default
        // llama.cpp itself falls back to.
        long? embdHead = keyLength
            ?? (embeddingLength is > 0 && headCount is > 0
                ? embeddingLength / headCount
                : null);

        // Only check when both sides of the invariant are actually present
        // and rope is even in use for this architecture (ropeDim == 0 is a
        // legitimate "no rotary embedding" value for some architectures, not
        // a bug -- don't flag that as a mismatch).
        if (embdHead is null or 0 || ropeDim is null or 0)
            return;

        if (embdHead != ropeDim)
            throw new InvalidDataException(
                $"'{Path.GetFileName(modelPath)}' has an inconsistent attention " +
                $"configuration: computed head dimension is {embdHead} " +
                $"(from {(keyLength is not null ? $"{arch}.attention.key_length" : $"{arch}.embedding_length / {arch}.attention.head_count")}) " +
                $"but {arch}.rope.dimension_count is {ropeDim}. llama.cpp requires " +
                $"these to be equal and will hard-crash the process (not throw a " +
                $"catchable exception) if they aren't. This is a converter bug in " +
                $"whatever produced this GGUF, not something fixable on load — " +
                $"the file needs to be regenerated with matching key_length/" +
                $"value_length and rope.dimension_count values.");
    }

    /// <summary>
    /// Cheap pre-flight check that runs before llama.cpp's native loader is
    /// touched. Reads the file's tensor directory with our pure-C# reader and
    /// rejects files whose tensor names look like HuggingFace PyTorch names
    /// (e.g. "model.layers.0.self_attn.q_proj.weight") instead of the
    /// llama.cpp names the loader requires (e.g. "blk.0.attn_q.weight").
    /// A file produced by Distillery's older tensor-mapping bug fails this.
    /// Any parse failure from <see cref="GgufReader"/> is deliberately NOT
    /// swallowed here — it surfaces as a separate, clearly-labeled
    /// InvalidDataException about a malformed file, so a corrupt/unsupported
    /// GGUF is distinguishable from a naming problem.
    /// </summary>
    private static async Task ValidateTensorNaming(string modelPath)
    {
        GgufFile gguf;
        try
        {
            gguf = await GgufReader.ReadAsync(modelPath);
        }
        catch (Exception ex) when (ex is InvalidDataException or NotSupportedException)
        {
            throw new InvalidDataException(
                $"'{Path.GetFileName(modelPath)}' failed GGUF header/tensor-directory parsing: {ex.Message}",
                ex);
        }

        const int sampleLimit = 5000;
        var count = 0;
        foreach (var t in gguf.Tensors)
        {
            if (++count > sampleLimit) break;
            var n = t.Name;
            if (n.StartsWith("model.", StringComparison.Ordinal) ||
                n.Contains(".self_attn.", StringComparison.Ordinal) ||
                n.Contains(".mlp.", StringComparison.Ordinal))
            {
                throw new InvalidDataException(
                    $"'{Path.GetFileName(modelPath)}' uses HuggingFace-style tensor names " +
                    $"(e.g. '{n}') instead of llama.cpp names (e.g. 'blk.0.attn_q.weight'). " +
                    "This file was likely produced with a tensor-name-mapping bug and cannot " +
                    "be loaded by llama.cpp. Repair it (e.g. with repair_gguf.py) and re-import.");
            }
        }
    }

    private static string BuildChatPrompt(
        IReadOnlyList<(MessageRole Role, string Content, IReadOnlyList<InferenceAttachment> Attachments)> messages,
        string systemPrompt)
    {
        // Same ChatML-style framing OnnxRuntimeService uses. Most GGUF chat
        // models ship their own Jinja chat template under the
        // "tokenizer.chat_template" GGUF metadata key (see GgufModelMetadata.
        // HasChatTemplate) — swap this for that template if you need exact
        // per-model prompt formatting instead of a generic ChatML frame.
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

    public async Task<IReadOnlyList<float[]>> GetEmbeddingsAsync(
        Guid modelId, IReadOnlyList<string> inputs, CancellationToken ct = default)
    {
        if (!_loaded.TryGetValue(modelId, out var loaded))
            throw new InvalidOperationException("Model is not loaded. Call LoadAsync first.");

        loaded.Embedder ??= new LLamaEmbedder(loaded.Weights, new ModelParams(string.Empty)
        {
            ContextSize = loaded.Context.ContextSize,
        });

        var results = new List<float[]>(inputs.Count);
        foreach (var input in inputs)
        {
            ct.ThrowIfCancellationRequested();
            var embedding = await loaded.Embedder.GetEmbeddings(input);
            results.Add(embedding.First());
        }
        return results;
    }

    public void Dispose() => UnloadAll();
}
