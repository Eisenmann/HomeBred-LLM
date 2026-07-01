using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CommunityToolkit.Mvvm.Messaging;
using HomebredLLM.Models;

namespace HomebredLLM.ViewModels;

// Sent by ModelLibraryViewModel when the user clicks "⚙ Config" on a model row
public record OpenConfigMessage(LocalModel Model);

public partial class MainViewModel : ObservableObject
{
    private readonly ModelLibraryViewModel _libraryVm;
    private readonly ChatViewModel         _chatVm;
    private readonly AnalyticsViewModel    _analyticsVm;
    private readonly ModelConfigViewModel  _configVm;

    [ObservableProperty] private object _currentView = null!;
    [ObservableProperty] private string _statusMessage = "Ready";

    public MainViewModel(
        ModelLibraryViewModel library,
        ChatViewModel         chat,
        AnalyticsViewModel    analytics,
        ModelConfigViewModel  config)
    {
        _libraryVm   = library;
        _chatVm      = chat;
        _analyticsVm = analytics;
        _configVm    = config;
        _currentView = library;

        // Config nav driven by a message from ModelLibraryViewModel
        WeakReferenceMessenger.Default.Register<OpenConfigMessage>(this, (_, msg) =>
        {
            _ = _configVm.LoadAsync(msg.Model.Id);
            CurrentView = _configVm;
        });
    }

    [RelayCommand]
    private async Task NavigateToAsync(string page)
    {
        if (page == "Chat") await _chatVm.RefreshRunningModelsAsync();

        CurrentView = page switch
        {
            "Chat"      => (object)_chatVm,
            "Analytics" => _analyticsVm,
            "Config"    => _configVm,
            _           => _libraryVm,
        };
    }

    public void SetStatus(string msg) => StatusMessage = msg;
}
