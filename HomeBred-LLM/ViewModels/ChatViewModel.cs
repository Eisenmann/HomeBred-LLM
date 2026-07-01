using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using HomebredLLM.Data;
using HomebredLLM.Models;
using HomebredLLM.Services;
using Microsoft.EntityFrameworkCore;
using System.Collections.ObjectModel;

namespace HomebredLLM.ViewModels;

public partial class ChatViewModel(
    IDbContextFactory<AppDbContext> dbFactory,
    IInferenceService inference,
    MetricsCollectorService metricsCollector) : ObservableObject
{
    [ObservableProperty] private ObservableCollection<ChatSession> _sessions = [];
    [ObservableProperty] private ChatSession? _selectedSession;
    [ObservableProperty] private ObservableCollection<ChatMessage> _messages = [];
    [ObservableProperty] private ObservableCollection<LocalModel> _runningModels = [];
    [ObservableProperty] private LocalModel? _selectedModel;
    [ObservableProperty] private string _userInput = "";
    [ObservableProperty] private bool _isGenerating;
    [ObservableProperty] private string _streamingReply = "";
    [ObservableProperty] private string _statusText = "";

    private CancellationTokenSource? _cts;

    public async Task InitializeAsync()
    {
        await using var db = await dbFactory.CreateDbContextAsync();
        var sessions = await db.ChatSessions.OrderByDescending(s => s.UpdatedAt).ToListAsync();
        Sessions = new ObservableCollection<ChatSession>(sessions);

        await RefreshRunningModelsAsync();
    }

    // Models started/stopped in the Model Library after startup don't otherwise
    // reach this singleton VM, so the nav command re-calls this on every visit.
    public async Task RefreshRunningModelsAsync()
    {
        await using var db = await dbFactory.CreateDbContextAsync();
        var running = await db.Models.Where(m => m.Status == ModelStatus.Running).ToListAsync();
        var keepSelected = SelectedModel is not null && running.Any(m => m.Id == SelectedModel.Id);

        RunningModels = new ObservableCollection<LocalModel>(running);
        SelectedModel = keepSelected
            ? running.First(m => m.Id == SelectedModel!.Id)
            : running.FirstOrDefault();
    }

    [RelayCommand]
    private async Task NewSessionAsync()
    {
        if (SelectedModel is null) { StatusText = "Start a model first."; return; }

        await using var db = await dbFactory.CreateDbContextAsync();
        var session = new ChatSession { ModelId = SelectedModel.Id, Title = "New Chat" };
        db.ChatSessions.Add(session);
        await db.SaveChangesAsync();

        Sessions.Insert(0, session);
        await SelectSessionAsync(session);
    }

    // Replaces WPF Interaction.Triggers on ListBox SelectionChanged
    partial void OnSelectedSessionChanged(ChatSession? value)
    {
        if (value is not null) _ = SelectSessionAsync(value);
    }

    [RelayCommand]
    private async Task SelectSessionAsync(ChatSession session)
    {
        SelectedSession = session;
        await using var db = await dbFactory.CreateDbContextAsync();
        var msgs = await db.ChatMessages
            .Where(m => m.SessionId == session.Id)
            .OrderBy(m => m.CreatedAt)
            .ToListAsync();
        Messages = new ObservableCollection<ChatMessage>(msgs);
        StreamingReply = "";
    }

    [RelayCommand]
    private async Task DeleteSessionAsync(ChatSession session)
    {
        await using var db = await dbFactory.CreateDbContextAsync();
        var s = await db.ChatSessions.FindAsync(session.Id);
        if (s is not null) { db.ChatSessions.Remove(s); await db.SaveChangesAsync(); }
        Sessions.Remove(session);
        if (SelectedSession?.Id == session.Id)
        {
            SelectedSession = null;
            Messages.Clear();
        }
    }

    [RelayCommand]
    private async Task SendAsync()
    {
        if (string.IsNullOrWhiteSpace(UserInput) || IsGenerating) return;
        if (SelectedSession is null) await NewSessionAsync();
        if (SelectedSession is null || SelectedModel is null) return;
        if (!inference.IsLoaded(SelectedModel.Id)) { StatusText = "Model not loaded."; return; }

        var userText = UserInput.Trim();
        UserInput = "";
        IsGenerating = true;
        StreamingReply = "";

        // Persist user message
        var userMsg = new ChatMessage { SessionId = SelectedSession.Id, Role = MessageRole.User, Content = userText };
        await using (var db = await dbFactory.CreateDbContextAsync())
        {
            db.ChatMessages.Add(userMsg);
            await db.SaveChangesAsync();
        }
        // Build history before queuing the UI update below — Dispatcher.Post runs
        // later on the message loop, so reading Messages right after posting would
        // miss the message that was just sent.
        var history = Messages
            .Select(m => (m.Role, m.Content))
            .Append((userMsg.Role, userMsg.Content))
            .ToList();

        Dispatcher.UIThread.Post(() => Messages.Add(userMsg));

        await using var db2 = await dbFactory.CreateDbContextAsync();
        var cfg = await db2.ModelConfigurations.FirstOrDefaultAsync(c => c.ModelId == SelectedModel.Id)
                  ?? new ModelConfiguration();

        var request = new InferenceRequest(SelectedModel.Id, SelectedModel.LocalPath!, cfg, history);
        _cts = new CancellationTokenSource();

        InferenceStats? stats = null;
        try
        {
            await foreach (var token in inference.ChatStreamAsync(request, s => stats = s, _cts.Token))
            {
                Dispatcher.UIThread.Post(() => StreamingReply += token);
            }
        }
        catch (OperationCanceledException) { StreamingReply += " [cancelled]"; }

        // Persist assistant reply
        var assistantMsg = new ChatMessage
        {
            SessionId = SelectedSession.Id,
            Role = MessageRole.Assistant,
            Content = StreamingReply,
            Tokens = stats?.OutputTokens,
            TokensPerSecond = stats?.TokensPerSecond,
            InferenceTimeMs = stats?.TotalMs,
        };
        await using (var db3 = await dbFactory.CreateDbContextAsync())
        {
            db3.ChatMessages.Add(assistantMsg);
            var s = await db3.ChatSessions.FindAsync(SelectedSession.Id);
            if (s != null)
            {
                s.UpdatedAt = DateTime.UtcNow;
                if (s.Title == "New Chat" && userText.Length > 0)
                    s.Title = userText.Length > 60 ? userText[..60] + "…" : userText;
            }
            await db3.SaveChangesAsync();
        }

        Dispatcher.UIThread.Post(() =>
        {
            Messages.Add(assistantMsg);
            StreamingReply = "";
            if (stats is not null) metricsCollector.LastInferenceStats = stats;
            if (SelectedSession != null) SelectedSession.UpdatedAt = DateTime.UtcNow;
        });

        IsGenerating = false;
    }

    [RelayCommand]
    private void CancelGeneration() => _cts?.Cancel();
}
