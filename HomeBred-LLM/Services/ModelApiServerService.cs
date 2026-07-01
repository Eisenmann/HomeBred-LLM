using HomebredLLM.Data;
using HomebredLLM.Models;
using Microsoft.EntityFrameworkCore;
using System.Collections.Concurrent;
using System.Net;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace HomebredLLM.Services;

// ── OpenAI-compatible chat completions DTOs ────────────────────────────────

public record OaiMessage(
    [property: JsonPropertyName("role")] string Role,
    [property: JsonPropertyName("content")] string Content);

public record OaiChatRequest(
    [property: JsonPropertyName("model")] string? Model,
    [property: JsonPropertyName("messages")] List<OaiMessage>? Messages,
    [property: JsonPropertyName("temperature")] float? Temperature,
    [property: JsonPropertyName("top_p")] float? TopP,
    [property: JsonPropertyName("max_tokens")] int? MaxTokens,
    [property: JsonPropertyName("stream")] bool? Stream);

public record OaiUsage(
    [property: JsonPropertyName("prompt_tokens")] int PromptTokens,
    [property: JsonPropertyName("completion_tokens")] int CompletionTokens,
    [property: JsonPropertyName("total_tokens")] int TotalTokens);

public record OaiDelta(
    [property: JsonPropertyName("role")] string? Role,
    [property: JsonPropertyName("content")] string? Content);

public record OaiChoice(
    [property: JsonPropertyName("index")] int Index,
    [property: JsonPropertyName("message")] OaiMessage? Message,
    [property: JsonPropertyName("delta")] OaiDelta? Delta,
    [property: JsonPropertyName("finish_reason")] string? FinishReason);

public record OaiChatResponse(
    [property: JsonPropertyName("id")] string Id,
    [property: JsonPropertyName("object")] string Object,
    [property: JsonPropertyName("created")] long Created,
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("choices")] List<OaiChoice> Choices,
    [property: JsonPropertyName("usage")] OaiUsage? Usage);

public record OaiErrorBody([property: JsonPropertyName("message")] string Message);
public record OaiError([property: JsonPropertyName("error")] OaiErrorBody Error);

// ── OpenAI-compatible embeddings DTOs ──────────────────────────────────────

public record OaiEmbeddingRequest(
    [property: JsonPropertyName("model")] string? Model,
    [property: JsonPropertyName("input")] JsonElement Input);

public record OaiEmbeddingItem(
    [property: JsonPropertyName("object")] string Object,
    [property: JsonPropertyName("index")] int Index,
    [property: JsonPropertyName("embedding")] float[] Embedding);

public record OaiEmbeddingResponse(
    [property: JsonPropertyName("object")] string Object,
    [property: JsonPropertyName("data")] List<OaiEmbeddingItem> Data,
    [property: JsonPropertyName("model")] string Model);

/// <summary>
/// Exposes each running model as a local OpenAI-compatible HTTP endpoint
/// (POST /v1/chat/completions with streaming, POST /v1/embeddings) on its own
/// configured port. Bound to 127.0.0.1 only — never reachable from the network.
/// </summary>
public sealed class ModelApiServerService(
    IInferenceService inference,
    IDbContextFactory<AppDbContext> dbFactory) : IDisposable
{
    private static readonly JsonSerializerOptions JsonOpts = new(JsonSerializerDefaults.Web);

    private sealed record RunningServer(HttpListener Listener, CancellationTokenSource Cts, int Port);
    private readonly ConcurrentDictionary<Guid, RunningServer> _servers = new();

    public bool IsRunning(Guid modelId) => _servers.ContainsKey(modelId);

    public string? EndpointUrl(Guid modelId) =>
        _servers.TryGetValue(modelId, out var s) ? $"http://127.0.0.1:{s.Port}" : null;

    public void Start(Guid modelId, int port)
    {
        if (_servers.ContainsKey(modelId)) return;

        var listener = new HttpListener();
        listener.Prefixes.Add($"http://127.0.0.1:{port}/");
        listener.Start(); // throws HttpListenerException if the port is taken

        var cts = new CancellationTokenSource();
        _servers[modelId] = new RunningServer(listener, cts, port);
        _ = AcceptLoopAsync(modelId, listener, cts.Token);
    }

    public void Stop(Guid modelId)
    {
        if (_servers.TryRemove(modelId, out var s))
        {
            s.Cts.Cancel();
            try { s.Listener.Stop(); s.Listener.Close(); } catch { /* already closed */ }
        }
    }

    public void StopAll()
    {
        foreach (var id in _servers.Keys.ToArray()) Stop(id);
    }

    private async Task AcceptLoopAsync(Guid modelId, HttpListener listener, CancellationToken ct)
    {
        while (!ct.IsCancellationRequested && listener.IsListening)
        {
            HttpListenerContext ctx;
            try { ctx = await listener.GetContextAsync(); }
            catch { break; } // listener was stopped/disposed

            _ = HandleRequestAsync(ctx, modelId, ct);
        }
    }

    private async Task HandleRequestAsync(HttpListenerContext ctx, Guid modelId, CancellationToken serverCt)
    {
        var req = ctx.Request;
        var resp = ctx.Response;

        try
        {
            if (req.HttpMethod == "GET" && req.Url?.AbsolutePath == "/v1/models")
            {
                await WriteJsonAsync(resp, 200, new { data = new[] { new { id = modelId.ToString(), @object = "model" } } });
                return;
            }

            if (!inference.IsLoaded(modelId))
            {
                await WriteJsonAsync(resp, 503, new OaiError(new OaiErrorBody("Model is not loaded.")));
                return;
            }

            if (req.HttpMethod == "POST" && req.Url?.AbsolutePath == "/v1/embeddings")
            {
                await HandleEmbeddingsAsync(ctx, modelId, serverCt);
                return;
            }

            if (req.HttpMethod != "POST" || req.Url?.AbsolutePath != "/v1/chat/completions")
            {
                await WriteJsonAsync(resp, 404, new OaiError(new OaiErrorBody(
                    "Unknown endpoint. Use POST /v1/chat/completions or POST /v1/embeddings.")));
                return;
            }

            using var bodyReader = new StreamReader(req.InputStream, req.ContentEncoding ?? Encoding.UTF8);
            var bodyText = await bodyReader.ReadToEndAsync();
            var body = JsonSerializer.Deserialize<OaiChatRequest>(bodyText, JsonOpts);

            if (body?.Messages is null || body.Messages.Count == 0)
            {
                await WriteJsonAsync(resp, 400, new OaiError(new OaiErrorBody("\"messages\" is required.")));
                return;
            }

            await using var db = await dbFactory.CreateDbContextAsync(serverCt);
            var cfg = await db.ModelConfigurations.FirstOrDefaultAsync(c => c.ModelId == modelId, serverCt)
                      ?? new ModelConfiguration();

            var messages = new List<(MessageRole Role, string Content)>();
            if (!string.IsNullOrWhiteSpace(cfg.SystemPrompt) &&
                !body.Messages.Any(m => string.Equals(m.Role, "system", StringComparison.OrdinalIgnoreCase)))
            {
                messages.Add((MessageRole.System, cfg.SystemPrompt));
            }
            messages.AddRange(body.Messages.Select(m => (ParseRole(m.Role), m.Content)));

            var effectiveCfg = new ModelConfiguration
            {
                ContextSize = cfg.ContextSize,
                Temperature = body.Temperature ?? cfg.Temperature,
                TopP = body.TopP ?? cfg.TopP,
                TopK = cfg.TopK,
                RepeatPenalty = cfg.RepeatPenalty,
                GpuLayerCount = cfg.GpuLayerCount,
                ThreadCount = cfg.ThreadCount,
                BatchSize = cfg.BatchSize,
                MaxTokens = body.MaxTokens ?? cfg.MaxTokens,
            };

            var request = new InferenceRequest(modelId, "", effectiveCfg, messages);
            var id = "chatcmpl-" + Guid.NewGuid().ToString("N");
            var created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
            var modelLabel = body.Model ?? modelId.ToString();

            using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(serverCt);

            if (body.Stream == true)
            {
                resp.ContentType = "text/event-stream";
                resp.Headers.Add("Cache-Control", "no-cache");
                var writer = new StreamWriter(resp.OutputStream, new UTF8Encoding(false)) { AutoFlush = true };

                await writer.WriteAsync(SseChunk(new OaiChatResponse(id, "chat.completion.chunk", created, modelLabel,
                    [new OaiChoice(0, null, new OaiDelta("assistant", ""), null)], null)));

                await foreach (var token in inference.ChatStreamAsync(request, _ => { }, linkedCts.Token))
                {
                    await writer.WriteAsync(SseChunk(new OaiChatResponse(id, "chat.completion.chunk", created, modelLabel,
                        [new OaiChoice(0, null, new OaiDelta(null, token), null)], null)));
                }

                await writer.WriteAsync(SseChunk(new OaiChatResponse(id, "chat.completion.chunk", created, modelLabel,
                    [new OaiChoice(0, null, new OaiDelta(null, null), "stop")], null)));
                await writer.WriteAsync("data: [DONE]\n\n");
                await writer.FlushAsync();
                resp.OutputStream.Close();
            }
            else
            {
                var sb = new StringBuilder();
                InferenceStats? stats = null;
                await foreach (var token in inference.ChatStreamAsync(request, s => stats = s, linkedCts.Token))
                    sb.Append(token);

                var result = new OaiChatResponse(id, "chat.completion", created, modelLabel,
                    [new OaiChoice(0, new OaiMessage("assistant", sb.ToString()), null, "stop")],
                    stats is null ? null : new OaiUsage(stats.PromptTokens, stats.OutputTokens, stats.PromptTokens + stats.OutputTokens));

                await WriteJsonAsync(resp, 200, result);
            }
        }
        catch (Exception ex)
        {
            try { await WriteJsonAsync(resp, 500, new OaiError(new OaiErrorBody(ex.Message))); }
            catch { /* response may already be closed */ }
        }
    }

    private async Task HandleEmbeddingsAsync(HttpListenerContext ctx, Guid modelId, CancellationToken serverCt)
    {
        var req = ctx.Request;
        var resp = ctx.Response;

        using var bodyReader = new StreamReader(req.InputStream, req.ContentEncoding ?? Encoding.UTF8);
        var bodyText = await bodyReader.ReadToEndAsync();
        var body = JsonSerializer.Deserialize<OaiEmbeddingRequest>(bodyText, JsonOpts);

        if (body is null)
        {
            await WriteJsonAsync(resp, 400, new OaiError(new OaiErrorBody("Invalid JSON body.")));
            return;
        }

        List<string> inputs = body.Input.ValueKind switch
        {
            JsonValueKind.String => [body.Input.GetString() ?? ""],
            JsonValueKind.Array => body.Input.EnumerateArray().Select(e => e.GetString() ?? "").ToList(),
            _ => [],
        };

        if (inputs.Count == 0)
        {
            await WriteJsonAsync(resp, 400, new OaiError(new OaiErrorBody("\"input\" must be a non-empty string or array of strings.")));
            return;
        }

        using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(serverCt);
        var vectors = await inference.GetEmbeddingsAsync(modelId, inputs, linkedCts.Token);

        var data = vectors.Select((v, i) => new OaiEmbeddingItem("embedding", i, v)).ToList();
        var modelLabel = body.Model ?? modelId.ToString();

        await WriteJsonAsync(resp, 200, new OaiEmbeddingResponse("list", data, modelLabel));
    }

    private static string SseChunk(OaiChatResponse chunk) => $"data: {JsonSerializer.Serialize(chunk, JsonOpts)}\n\n";

    private static async Task WriteJsonAsync(HttpListenerResponse resp, int statusCode, object body)
    {
        resp.StatusCode = statusCode;
        resp.ContentType = "application/json";
        var bytes = JsonSerializer.SerializeToUtf8Bytes(body, JsonOpts);
        await resp.OutputStream.WriteAsync(bytes);
        resp.OutputStream.Close();
    }

    private static MessageRole ParseRole(string role) => role.ToLowerInvariant() switch
    {
        "system" => MessageRole.System,
        "assistant" => MessageRole.Assistant,
        _ => MessageRole.User,
    };

    public void Dispose() => StopAll();
}
