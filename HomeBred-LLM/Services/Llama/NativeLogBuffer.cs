using System.Collections.Concurrent;
using LLama.Native;

namespace HomebredLLM.Services;

/// <summary>
/// Thread-safe ring buffer that captures the most recent llama.cpp native
/// log lines (via the LLamaSharp native log callback). HomeBred-LLM is an
/// Avalonia GUI app with no attached console, so anything llama.cpp prints
/// to stderr is invisible to the running app. This buffer is the pipe that
/// rescues those diagnostics (real tensor names, architecture strings,
/// "expected X got Y" dims, etc.) so they can be folded into the
/// <see cref="InvalidOperationException"/> thrown by
/// <see cref="LlamaCppInferenceService.LoadAsync"/> when the native loader
/// fails.
///
/// Native logging is a single global/static stream in llama.cpp, so this is
/// registered once at app startup (App.axaml.cs) as a singleton and is
/// inherently shared across every model load attempt. It is the only place
/// the log callback can be registered (registering it more than once would
/// simply replace the previous callback, losing the earlier capture).
/// </summary>
public sealed class NativeLogBuffer
{
    private const int DefaultCapacity = 100;

    private readonly ConcurrentQueue<string> _lines = new();
    private readonly int _capacity;
    private readonly object _trimLock = new();

    /// <summary>
    /// A reference kept alive for the whole app lifetime. llama.cpp's native
    /// callback must not be garbage-collected while the native library still
    /// holds a pointer to it; keeping a strong field here guarantees that.
    /// </summary>
    private NativeLogConfig.LLamaLogCallback? _registeredCallback;

    public NativeLogBuffer(int capacity = DefaultCapacity)
    {
        _capacity = Math.Max(1, capacity);
    }

    /// <summary>
    /// True once the native callback has been installed. Guards against
    /// double registration (the native API replaces the previous callback,
    /// so a second registration would silently drop the first).
    /// </summary>
    public bool IsRegistered { get; private set; }

    /// <summary>
    /// Registers the llama.cpp log callback that appends to this buffer.
    /// Must be called once, early in app startup, before any model is loaded.
    /// Deliberately idempotent: a second call is a no-op so a re-entrant or
    /// accidental double-registration can't clobber the buffer.
    /// </summary>
    public void RegisterNativeLogCallback()
    {
        if (IsRegistered) return;

        _registeredCallback = (level, message) =>
        {
            // The message may span multiple lines (llama.cpp emits multi-line
            // messages with a single continuation log level). We keep each
            // physical line so the exception message reads naturally.
            foreach (var part in (message ?? string.Empty).Split('\n'))
            {
                if (string.IsNullOrEmpty(part)) continue;
                Enqueue(part);
            }
        };

        NativeLogConfig.llama_log_set(_registeredCallback);
        IsRegistered = true;
    }

    /// <summary>Appends a line to the buffer, trimming to the capacity.</summary>
    public void Enqueue(string line)
    {
        if (string.IsNullOrEmpty(line)) return;
        _lines.Enqueue(line);

        lock (_trimLock)
        {
            while (_lines.Count > _capacity && _lines.TryDequeue(out _)) { }
        }
    }

    /// <summary>
    /// Clears the buffer. Call at the start of each load attempt so the
    /// captured lines belong to *this* load, not a previous one.
    /// </summary>
    public void Clear() => _lines.Clear();

    /// <summary>
    /// Snapshots and clears the buffer. Returns the captured lines (most
    /// recent last) — designed for the catch block in LoadAsync.
    /// </summary>
    public IReadOnlyList<string> TakeRecentLines()
    {
        var snapshot = _lines.ToArray();
        _lines.Clear();
        return snapshot;
    }

    /// <summary>
    /// Returns the current buffered lines without clearing (for diagnostics).
    /// </summary>
    public IReadOnlyList<string> RecentLines() => _lines.ToArray();
}