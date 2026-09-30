using OpenCvSharp;
using VisionAttributeAI.Models.AI;
using VisionAttributeAI.Models.Pose;

namespace VisionAttributeAI.Services.Pose;

/// <summary>
/// Contract for multi-person pose estimation, keypoint extraction, and body region localization.
/// </summary>
public interface IPoseEstimationService : IDisposable
{
    bool IsModelLoaded { get; }

    /// <summary>
    /// Performs pose keypoint estimation on an image and extracts anatomical body visibility flags and region bounding boxes.
    /// </summary>
    Task<IReadOnlyList<PersonPoseResult>> EstimatePoseAsync(Mat image, IReadOnlyList<DetectionResult> detections, CancellationToken cancellationToken = default);
}
