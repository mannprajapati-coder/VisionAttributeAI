namespace VisionAttributeAI.Models.Tracking;

/// <summary>
/// Quality evaluation result for a single detected person crop.
/// </summary>
public record TrackQualityResult
{
    public bool IsAccepted { get; init; }
    public string RejectionReason { get; init; } = string.Empty;
    public double SharpnessVariance { get; init; }
    public float QualityScore { get; init; }
    public float ComputedWeight { get; init; }

    public static TrackQualityResult Accepted(double sharpness, float qualityScore, float computedWeight) =>
        new()
        {
            IsAccepted = true,
            RejectionReason = string.Empty,
            SharpnessVariance = sharpness,
            QualityScore = qualityScore,
            ComputedWeight = computedWeight
        };

    public static TrackQualityResult Rejected(string reason, double sharpness = 0.0) =>
        new()
        {
            IsAccepted = false,
            RejectionReason = reason,
            SharpnessVariance = sharpness,
            QualityScore = 0.0f,
            ComputedWeight = 0.0f
        };
}
