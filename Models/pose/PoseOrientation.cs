namespace VisionAttributeAI.Models.Pose;

/// <summary>
/// Estimated pose orientation determined from facial and shoulder keypoints.
/// </summary>
public enum PoseOrientation
{
    Front,
    NearFrontal,
    Side,
    Back,
    Uncertain
}
