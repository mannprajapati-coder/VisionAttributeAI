namespace VisionAttributeAI.Models.AI;

/// <summary>
/// Execution and timing metrics for the image analysis pipeline.
/// </summary>
public class AnalysisMetrics
{
    public double ImageDecodeMs { get; set; }
    public double DetectionMs { get; set; }
    public double AttributeRecognitionMs { get; set; }
    public double TotalElapsedMs { get; set; }
}

/// <summary>
/// Unified output of the image analysis pipeline.
/// </summary>
public class ImageAnalysisResult
{
    public DateTime ProcessedAtUtc { get; init; } = DateTime.UtcNow;
    public int ImageWidth { get; init; }
    public int ImageHeight { get; init; }
    public IReadOnlyList<AnalyzedPerson> Persons { get; init; } = [];
    public AnalysisMetrics Metrics { get; init; } = new();
}
