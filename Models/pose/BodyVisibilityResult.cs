namespace VisionAttributeAI.Models.Pose;

/// <summary>
/// Anatomical visibility flags and frame-boundary clipping indicators determined from keypoints.
/// </summary>
public record BodyVisibilityResult
{
    public bool HeadVisible { get; init; }
    public bool UpperBodyVisible { get; init; }
    public bool LeftArmVisible { get; init; }
    public bool RightArmVisible { get; init; }
    public bool LeftWristVisible { get; init; }
    public bool RightWristVisible { get; init; }
    public bool LowerBodyVisible { get; init; }
    public bool LeftLegVisible { get; init; }
    public bool RightLegVisible { get; init; }
    public bool FeetVisible { get; init; }

    public float OverallPoseConfidence { get; init; }
    public int VisibleKeypointCount { get; init; }

    // Gestures / Raised Hands (Anatomical: Wrist Y < Shoulder Y)
    public bool LeftHandRaised { get; init; }
    public bool RightHandRaised { get; init; }

    // Pose Orientation for Appearance Gating
    public PoseOrientation Orientation { get; init; } = PoseOrientation.Uncertain;

    // Frame Boundary Clipping Flags
    public bool HeadClipped { get; init; }
    public bool UpperClipped { get; init; }
    public bool LowerClipped { get; init; }
    public bool FeetClipped { get; init; }
    public bool LeftWristClipped { get; init; }
    public bool RightWristClipped { get; init; }
}
