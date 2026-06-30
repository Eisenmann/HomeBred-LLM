using Avalonia.Controls;
using HomebredLLM.ViewModels;

namespace HomebredLLM;

public partial class MainWindow : Window
{
    public MainWindow(MainViewModel vm)
    {
        InitializeComponent();
        DataContext = vm;
    }
}
