using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;
using HomebredLLM.ViewModels;

namespace HomebredLLM.Views;

public partial class ModelConfigView : UserControl
{
    public ModelConfigView()
    {
        InitializeComponent();
    }

    private async void AddAdapter_Click(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not ModelConfigViewModel vm) return;
        var storage = TopLevel.GetTopLevel(this)?.StorageProvider;
        if (storage is null) return;

        var files = await storage.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            AllowMultiple = false,
            Title = "Select a GGUF LoRA adapter",
            FileTypeFilter =
            [
                new FilePickerFileType("GGUF LoRA adapter") { Patterns = ["*.gguf"] },
            ],
        });
        if (files.Count > 0)
            await vm.AddAdapterAsync(files[0].Path.LocalPath);
    }
}
