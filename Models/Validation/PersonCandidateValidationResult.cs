using VisionAttributeAI.Models.AI;
using VisionAttributeAI.Models.Pose;

namespace VisionAttributeAI.Models.Validation;

public enum PersonCandidateTier
{
    Rejected = 0,
    ProbablePerson = 1,
    StrongPerson = 2
}

/// <summary>
/// Detailed evaluation result for a detected candidate bounding box prior to track initialization or attribute recognition.
/// </summary>
public record PersonCandidateValidationResult
{
    public int CandidateIndex { get; init; }
    public bool IsValidPerson { get; init; }
    public float ValidationScore { get; init; }
    public PersonCandidateTier Tier { get; init; } = PersonCandidateTier.Rejected;
    public string RejectReason { get; init; } = string.Empty;

    // Geometric Metrics
    public float Width { get; init; }
    public float Height { get; init; }
    public float Area { get; init; }
    public float AspectRatio { get; init; } // Width / Height
    public float FrameAreaFraction { get; init; }

    // Pose & Human Anatomical Metrics
    public int ValidKeypointCount { get; init; }
    public int UpperBodyKeypointCount { get; init; }
    public int LowerBodyKeypointCount { get; init; }
    public float AnatomicalScore { get; init; }
    public bool HasHeadEvidence { get; init; }
    public bool HasShoulderEvidence { get; init; }
    public bool HasTorsoEvidence { get; init; }
    public bool HasLimbEvidence { get; init; }

    // YOLO Detector Metrics
    public float YoloConfidence { get; init; }
    public BoundingBox BoundingBox { get; init; } = default!;
}
