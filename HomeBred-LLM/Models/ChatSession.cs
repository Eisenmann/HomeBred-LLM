using System.ComponentModel.DataAnnotations;

namespace HomebredLLM.Models;

public class ChatSession
{
    [Key] public Guid Id { get; set; } = Guid.NewGuid();
    public Guid ModelId { get; set; }
    public LocalModel? Model { get; set; }
    public string Title { get; set; } = "New Chat";
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;

    public List<ChatMessage> Messages { get; set; } = [];
}

public enum MessageRole { User, Assistant, System }

public class ChatMessage
{
    [Key] public Guid Id { get; set; } = Guid.NewGuid();
    public Guid SessionId { get; set; }
    public ChatSession? Session { get; set; }
    public MessageRole Role { get; set; }
    public string Content { get; set; } = "";
    public int? Tokens { get; set; }
    public float? TokensPerSecond { get; set; }
    public float? InferenceTimeMs { get; set; }
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    public List<ChatAttachment> Attachments { get; set; } = [];
}

public enum AttachmentKind { Image, Text, Other }

public class ChatAttachment
{
    [Key] public Guid Id { get; set; } = Guid.NewGuid();
    public Guid MessageId { get; set; }
    public ChatMessage? Message { get; set; }
    public string FileName { get; set; } = "";
    public string StoredPath { get; set; } = "";
    public AttachmentKind Kind { get; set; }
    public string? MimeType { get; set; }
    public long SizeBytes { get; set; }
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
}
