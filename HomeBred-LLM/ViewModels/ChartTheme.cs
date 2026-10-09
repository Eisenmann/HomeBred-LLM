using LiveChartsCore.SkiaSharpView;
using LiveChartsCore.SkiaSharpView.Painting;
using SkiaSharp;

namespace HomebredLLM.ViewModels;

/// <summary>Dark-theme paints for LiveCharts (its defaults are dark text on our dark cards).</summary>
public static class ChartTheme
{
    public static SolidColorPaint LegendPaint { get; } = new(SKColor.Parse("#D1D5DB"));
    public static SolidColorPaint LabelPaint { get; } = new(SKColor.Parse("#9CA3AF"));

    public static Axis Axis(string? name = null, Func<double, string>? labeler = null, string[]? labels = null)
    {
        var axis = new Axis
        {
            Name = name,
            NamePaint = name is null ? null : new SolidColorPaint(SKColor.Parse("#9CA3AF")),
            NameTextSize = 12,
            TextSize = 11,
            LabelsPaint = new SolidColorPaint(SKColor.Parse("#9CA3AF")),
            SeparatorsPaint = new SolidColorPaint(SKColor.Parse("#374151")),
            Labels = labels,
        };
        if (labeler is not null) axis.Labeler = labeler;
        return axis;
    }
}
