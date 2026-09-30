using OpenCvSharp;
using VisionAttributeAI.Models.AI;

namespace VisionAttributeAI.Services.Watch;

/// <summary>
/// Result of watch inspection on high-resolution left and right wrist crops.
/// States: NotVisible, InsufficientVisualEvidence, NoWatchDetected, WatchDetected, Unknown.
/// </summary>
public record WatchResult
{
    public bool HasWatch { get; init; }
    public float Confidence { get; init; }
    public string Status { get; init; } = "Unknown"; // "WatchDetected", "NoWatchDetected", "InsufficientVisualEvidence", "NotVisible", "Unknown"
    public string Details { get; init; } = string.Empty;

    public bool LeftWristAnalyzed { get; init; }
    public bool LeftWristDetected { get; init; }
    public float LeftWristConfidence { get; init; }

    public bool RightWristAnalyzed { get; init; }
    public bool RightWristDetected { get; init; }
    public float RightWristConfidence { get; init; }
}

/// <summary>
/// Contract for dedicated watch detection on localized wrist crops.
/// Evaluates left and right wrist crops independently.
/// </summary>
public interface IWatchDetectionService
{
    /// <summary>
    /// Evaluates left and right wrist crops independently for the presence of a watch using object detection.
    /// </summary>
    Task<WatchResult> DetectWatchAsync(Mat? leftWristCrop, Mat? rightWristCrop, CancellationToken cancellationToken = default);
}
