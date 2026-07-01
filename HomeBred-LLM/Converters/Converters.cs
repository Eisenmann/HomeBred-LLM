using Avalonia.Data.Converters;
using Avalonia.Layout;
using Avalonia.Media;
using Avalonia.Media.Imaging;
using HomebredLLM.Models;
using System.Globalization;

namespace HomebredLLM.Converters;

// Loads a file path into a Bitmap for inline image attachment previews.
// Swallows failures (missing/corrupt file) so a bad attachment can't crash the binding.
public class PathToBitmapConverter : IValueConverter
{
    public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        if (value is not string path || string.IsNullOrEmpty(path)) return null;
        try { return new Bitmap(path); }
        catch { return null; }
    }

    public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture) =>
        throw new NotSupportedException();
}

public class StatusToColorConverter : IValueConverter
{
    public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture) =>
        value is ModelStatus status
            ? status switch
            {
                ModelStatus.Running     => new SolidColorBrush(Color.FromRgb(16, 185, 129)),
                ModelStatus.Ready       => new SolidColorBrush(Color.FromRgb(99, 102, 241)),
                ModelStatus.Downloading => new SolidColorBrush(Color.FromRgb(245, 158, 11)),
                ModelStatus.Error       => new SolidColorBrush(Color.FromRgb(239, 68, 68)),
                _                       => new SolidColorBrush(Color.FromRgb(107, 114, 128)),
            }
            : new SolidColorBrush(Color.FromRgb(107, 114, 128));

    public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture) =>
        throw new NotSupportedException();
}

public class BytesToReadableConverter : IValueConverter
{
    public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        if (value is not long bytes) return "—";
        return bytes >= 1_000_000_000
            ? $"{bytes / 1_000_000_000.0:F1} GB"
            : $"{bytes / 1_000_000.0:F0} MB";
    }

    public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture) =>
        throw new NotSupportedException();
}

// Returns bool — Avalonia binds IsVisible to bool directly
public class BoolToVisibilityConverter : IValueConverter
{
    public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        bool boolVal = value switch
        {
            bool b => b,
            int i  => i > 0,
            _      => false,
        };
        bool inverse = parameter?.ToString() == "inverse";
        return boolVal ^ inverse;
    }

    public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture) =>
        value is bool b && b;
}

public class RoleToBubbleColorConverter : IValueConverter
{
    public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture) =>
        value is MessageRole role
            ? role switch
            {
                MessageRole.User      => new SolidColorBrush(Color.FromRgb(14, 165, 233)),
                MessageRole.Assistant => new SolidColorBrush(Color.FromRgb(31, 41, 55)),
                MessageRole.System    => new SolidColorBrush(Color.FromRgb(55, 65, 81)),
                _                     => Brushes.Gray,
            }
            : Brushes.Gray;

    public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture) =>
        throw new NotSupportedException();
}

public class RoleToAlignConverter : IValueConverter
{
    public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture) =>
        value is MessageRole role && role == MessageRole.User
            ? HorizontalAlignment.Right
            : HorizontalAlignment.Left;

    public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture) =>
        throw new NotSupportedException();
}
