using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;
using HomebredLLM.ViewModels;

namespace HomebredLLM.Views;

public partial class ModelLibraryView : UserControl
{
    public ModelLibraryView()
    {
        InitializeComponent();
    }

    private async void ImportLocalModel_Click(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not ModelLibraryViewModel vm) return;
        var storage = TopLevel.GetTopLevel(this)?.StorageProvider;
        if (storage is null) return;

        // ONNX models are directories containing model.onnx, config.json, and
        // tokenizer files — pick the model.onnx file and the import service
        // resolves the parent directory. GGUF models are a single .gguf file
        // and are imported directly.
        var files = await storage.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            AllowMultiple = false,
            Title = "Select a model.onnx file, or a .gguf model file",
            FileTypeFilter =
            [
                new FilePickerFileType("Model files") { Patterns = ["*.onnx", "*.gguf"] },
                new FilePickerFileType("ONNX model") { Patterns = ["*.onnx"] },
                new FilePickerFileType("GGUF model") { Patterns = ["*.gguf"] },
                new FilePickerFileType("All files") { Patterns = ["*"] },
            ],
        });
        if (files.Count > 0)
            await vm.ImportLocalModelAsync(files[0].Path.LocalPath);
    }
}