namespace VisionAttributeAI.Models.AI;

/// <summary>
/// Represents the result of an object/person detection.
/// </summary>
public class DetectionResult
{
    public required BoundingBox Box { get; init; }
    public float Confidence { get; init; }
    public int ClassId { get; init; }
    public string Label { get; init; } = "person";
    public int? TrackingId { get; init; }

    public DetectionResult() { }

    [System.Diagnostics.CodeAnalysis.SetsRequiredMembers]
    public DetectionResult(BoundingBox box, float confidence, string label = "person")
    {
        Box = box;
        Confidence = confidence;
        Label = label;
    }
}
