namespace VisionAttributeAI.Models.Pose;

/// <summary>
/// A single 2D pose keypoint with coordinate positions, confidence, and anatomical name.
/// </summary>
public record Keypoint(int Index, string Name, float X, float Y, float Confidence)
{
    public bool IsVisible(float minConfidence = 0.35f) => Confidence >= minConfidence && X > 0 && Y > 0;
}
