using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;
using HomebredLLM.ViewModels;

namespace HomebredLLM.Views;

public partial class ChatView : UserControl
{
    public ChatView()
    {
        InitializeComponent();
        AddHandler(DragDrop.DragOverEvent, OnDragOver);
        AddHandler(DragDrop.DragLeaveEvent, OnDragLeave);
        AddHandler(DragDrop.DropEvent, OnDrop);
    }

    private async void Attach_Click(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not ChatViewModel vm) return;
        var storage = TopLevel.GetTopLevel(this)?.StorageProvider;
        if (storage is null) return;

        var files = await storage.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            AllowMultiple = true,
            Title = "Attach files",
        });
        if (files.Count > 0)
            vm.AddFiles(files.Select(f => f.Path.LocalPath).ToList());
    }

    private async void AttachFolder_Click(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not ChatViewModel vm) return;
        var storage = TopLevel.GetTopLevel(this)?.StorageProvider;
        if (storage is null) return;

        var folders = await storage.OpenFolderPickerAsync(new FolderPickerOpenOptions
        {
            AllowMultiple = false,
            Title = "Attach folder",
        });
        if (folders.Count > 0)
            vm.AddFolder(folders[0].Path.LocalPath);
    }

    private void OnDragOver(object? sender, DragEventArgs e)
    {
        e.DragEffects = e.Data.Contains(DataFormats.Files) ? DragDropEffects.Copy : DragDropEffects.None;
        if (e.DragEffects != DragDropEffects.None)
            DropHighlight.Opacity = 1;
    }

    private void OnDragLeave(object? sender, DragEventArgs e) => DropHighlight.Opacity = 0;

    private void OnDrop(object? sender, DragEventArgs e)
    {
        DropHighlight.Opacity = 0;
        if (DataContext is not ChatViewModel vm) return;

        var items = e.Data.GetFiles();
        if (items is null) return;

        foreach (var item in items)
        {
            if (item is IStorageFolder folder)
                vm.AddFolder(folder.Path.LocalPath);
            else
                vm.AddFiles([item.Path.LocalPath]);
        }
    }
}
