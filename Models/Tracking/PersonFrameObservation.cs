using VisionAttributeAI.Models.AI;
using VisionAttributeAI.Models.Pose;

namespace VisionAttributeAI.Models.Tracking;

/// <summary>
/// Explicit frame-level binding between a YOLO person detection, its corresponding YOLO-Pose keypoints/regions,
/// and its associated multi-object track state. Ensures zero ambiguity or array-index drift.
/// </summary>
public class PersonFrameObservation
{
    public int DetectionIndex { get; set; }
    public DetectionResult Detection { get; set; } = default!;
    public PersonPoseResult Pose { get; set; } = default!;
    public int MatchedTrackId { get; set; }
    public TrackedPersonState TrackState { get; set; } = default!;
    public float AssociationScore { get; set; }
    public float IoU { get; set; }
    public float CenterDistance { get; set; }
    public float NormalizedDisplacement { get; set; }
    public float AreaRatio { get; set; }
    public bool PossibleIdSwitch { get; set; }
    public string AssociationReason { get; set; } = string.Empty;
    public bool IsNewTrack { get; set; }
    public bool AssociationAccepted { get; set; } = true;
    public bool IsConfirmed => TrackState?.IsConfirmed ?? false;
    public BoundingBox BoundingBox => Detection.Box;
}
