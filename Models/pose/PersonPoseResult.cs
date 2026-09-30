using VisionAttributeAI.Models.AI;

namespace VisionAttributeAI.Models.Pose;

/// <summary>
/// Complete 17-keypoint pose estimation and derived body region analysis for a detected person.
/// </summary>
public record PersonPoseResult
{
    public BoundingBox BoundingBox { get; init; } = new(0, 0, 0, 0);
    public float DetectionConfidence { get; init; }
    public List<Keypoint> Keypoints { get; init; } = new();
    public BodyVisibilityResult Visibility { get; init; } = new();
    public BodyRegions Regions { get; init; } = new();

    public static readonly string[] CocoKeypointNames =
    [
        "Nose",          // 0
        "LeftEye",       // 1
        "RightEye",      // 2
        "LeftEar",       // 3
        "RightEar",      // 4
        "LeftShoulder",  // 5
        "RightShoulder", // 6
        "LeftElbow",     // 7
        "RightElbow",    // 8
        "LeftWrist",     // 9
        "RightWrist",    // 10
        "LeftHip",       // 11
        "RightHip",      // 12
        "LeftKnee",      // 13
        "RightKnee",     // 14
        "LeftAnkle",     // 15
        "RightAnkle"     // 16
    ];
}
