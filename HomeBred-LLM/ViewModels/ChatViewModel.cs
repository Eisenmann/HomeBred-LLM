using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using HomebredLLM.Data;
using HomebredLLM.Models;
using HomebredLLM.Services;
using Microsoft.EntityFrameworkCore;
using System.Collections.ObjectModel;

namespace HomebredLLM.ViewModels;

public sealed record PendingAttachment(string DisplayName, string SourcePath, AttachmentKind Kind, long SizeBytes);

public partial class ChatViewModel(
    IDbContextFactory<AppDbContext> dbFactory,
    IInferenceService inference,
    MetricsCollectorService metricsCollector) : ObservableObject
{
    private static readonly HashSet<string> ImageExtensions = new(StringComparer.OrdinalIgnoreCase)
        { ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp" };

    private static readonly HashSet<string> TextExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".txt", ".md", ".markdown", ".cs", ".py", ".js", ".jsx", ".ts", ".tsx", ".json", ".xml",
        ".yaml", ".yml", ".csv", ".html", ".htm", ".css", ".java", ".c", ".h", ".cpp", ".hpp",
        ".go", ".rs", ".rb", ".php", ".sql", ".sh", ".ps1", ".log", ".ini", ".toml", ".cfg", ".config",
    };

    private static readonly HashSet<string> SkippedDirNames = new(StringComparer.OrdinalIgnoreCase)
        { "node_modules", "bin", "obj", ".git", ".vs", ".idea" };

    private const int MaxAttachedFiles = 50;
    private const int MaxAttachedImages = 10;
    private const long MaxAttachedTextBytes = 25L * 1024 * 1024;

    [ObservableProperty] private ObservableCollection<ChatSession> _sessions = [];
    [ObservableProperty] private ChatSession? _selectedSession;
    [ObservableProperty] private ObservableCollection<ChatMessage> _messages = [];
    [ObservableProperty] private ObservableCollection<LocalModel> _runningModels = [];
    [ObservableProperty] private LocalModel? _selectedModel;
    [ObservableProperty] private string _userInput = "";
    [ObservableProperty] private bool _isGenerating;
    [ObservableProperty] private string _streamingReply = "";
    [ObservableProperty] private string _statusText = "";
    [ObservableProperty] private ObservableCollection<PendingAttachment> _pendingAttachments = [];
    [ObservableProperty] private bool _selectedModelSupportsVision;

    private CancellationTokenSource? _cts;

    partial void OnSelectedModelChanged(LocalModel? value) =>
        SelectedModelSupportsVision = value is not null && inference.SupportsVision(value.Id);

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
            .Include(m => m.Attachments)
            .Where(m => m.SessionId == session.Id)
            .OrderBy(m => m.CreatedAt)
            .ToListAsync();
        Messages = new ObservableCollection<ChatMessage>(msgs);
        StreamingReply = "";
        PendingAttachments.Clear();
    }

    public void AddFiles(IReadOnlyList<string> paths)
    {
        foreach (var path in paths) AddSingleFile(path);
    }

    public void AddFolder(string folderPath)
    {
        if (!Directory.Exists(folderPath)) return;

        int added = 0, skipped = 0;
        foreach (var path in EnumerateFolderFiles(folderPath))
        {
            var before = PendingAttachments.Count;
            AddSingleFile(path);
            if (PendingAttachments.Count > before) added++; else skipped++;
        }

        StatusText = skipped > 0
            ? $"Attached {added} file(s) from folder — skipped {skipped} (unsupported type or over the attachment limit)."
            : $"Attached {added} file(s) from folder.";
    }

    [RelayCommand]
    private void RemovePendingAttachment(PendingAttachment attachment) => PendingAttachments.Remove(attachment);

    private void AddSingleFile(string path)
    {
        if (!File.Exists(path)) return;

        var fileName = Path.GetFileName(path);
        var ext = Path.GetExtension(path);
        long size;
        try { size = new FileInfo(path).Length; }
        catch (Exception ex) { StatusText = $"Skipped {fileName} — {ex.Message}"; return; }

        if (ImageExtensions.Contains(ext))
        {
            if (!SelectedModelSupportsVision)
            {
                StatusText = $"Skipped {fileName} — the running model doesn't support image input.";
                return;
            }
            if (PendingAttachments.Count(a => a.Kind == AttachmentKind.Image) >= MaxAttachedImages)
            {
                StatusText = $"Skipped {fileName} — image attachment limit ({MaxAttachedImages}) reached.";
                return;
            }
            PendingAttachments.Add(new PendingAttachment(fileName, path, AttachmentKind.Image, size));
            return;
        }

        if (TextExtensions.Contains(ext))
        {
            if (PendingAttachments.Count >= MaxAttachedFiles)
            {
                StatusText = $"Skipped {fileName} — attachment limit ({MaxAttachedFiles} files) reached.";
                return;
            }
            var currentTextBytes = PendingAttachments.Where(a => a.Kind == AttachmentKind.Text).Sum(a => a.SizeBytes);
            if (currentTextBytes + size > MaxAttachedTextBytes)
            {
                StatusText = $"Skipped {fileName} — attached text would exceed the {MaxAttachedTextBytes / (1024 * 1024)} MB cap.";
                return;
            }
            PendingAttachments.Add(new PendingAttachment(fileName, path, AttachmentKind.Text, size));
            return;
        }

        StatusText = $"Skipped {fileName} — unsupported file type.";
    }

    private static IEnumerable<string> EnumerateFolderFiles(string root)
    {
        var stack = new Stack<string>();
        stack.Push(root);
        while (stack.Count > 0)
        {
            var dir = stack.Pop();
            IEnumerable<string> entries;
            try { entries = Directory.EnumerateFileSystemEntries(dir); }
            catch { continue; }

            foreach (var entry in entries)
            {
                if (Directory.Exists(entry))
                {
                    var name = Path.GetFileName(entry);
                    if (name.StartsWith('.') || SkippedDirNames.Contains(name)) continue;
                    stack.Push(entry);
                }
                else
                {
                    yield return entry;
                }
            }
        }
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
        if ((string.IsNullOrWhiteSpace(UserInput) && PendingAttachments.Count == 0) || IsGenerating) return;
        if (SelectedSession is null) await NewSessionAsync();
        if (SelectedSession is null || SelectedModel is null) return;
        if (!inference.IsLoaded(SelectedModel.Id)) { StatusText = "Model not loaded."; return; }

        var userText = UserInput.Trim();
        var attachmentsToSend = PendingAttachments.ToList();
        UserInput = "";
        PendingAttachments.Clear();
        IsGenerating = true;
        StreamingReply = "";

        // Persist user message + copy attachments into permanent storage so the
        // chat history stays valid even if the original source file moves or is deleted.
        var userMsg = new ChatMessage { SessionId = SelectedSession.Id, Role = MessageRole.User, Content = userText };
        if (attachmentsToSend.Count > 0)
        {
            var sessionDir = Path.Combine(AppPaths.AttachmentsDirectory, SelectedSession.Id.ToString());
            Directory.CreateDirectory(sessionDir);

            foreach (var pending in attachmentsToSend)
            {
                var storedPath = Path.Combine(sessionDir, $"{Guid.NewGuid():N}_{pending.DisplayName}");
                try { File.Copy(pending.SourcePath, storedPath, overwrite: true); }
                catch (Exception ex) { StatusText = $"Failed to attach {pending.DisplayName}: {ex.Message}"; continue; }

                userMsg.Attachments.Add(new ChatAttachment
                {
                    MessageId = userMsg.Id,
                    FileName = pending.DisplayName,
                    StoredPath = storedPath,
                    Kind = pending.Kind,
                    SizeBytes = pending.SizeBytes,
                });
            }
        }

        await using (var db = await dbFactory.CreateDbContextAsync())
        {
            db.ChatMessages.Add(userMsg);
            await db.SaveChangesAsync();
        }
        // Build history before queuing the UI update below — Dispatcher.Post runs
        // later on the message loop, so reading Messages right after posting would
        // miss the message that was just sent.
        var history = Messages
            .Append(userMsg)
            .Select(m => (m.Role, m.Content, (IReadOnlyList<InferenceAttachment>)m.Attachments
                .Select(a => new InferenceAttachment(a.StoredPath, a.Kind)).ToList()))
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
