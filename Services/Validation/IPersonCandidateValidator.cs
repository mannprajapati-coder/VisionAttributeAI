using VisionAttributeAI.Models.AI;
using VisionAttributeAI.Models.Pose;
using VisionAttributeAI.Models.Validation;

namespace VisionAttributeAI.Services.Validation;

/// <summary>
/// Service contract for validating whether raw YOLO detections represent authentic human candidates
/// before initializing/updating person tracks or generating user-facing person results.
/// </summary>
public interface IPersonCandidateValidator
{
    /// <summary>
    /// Evaluates a single candidate detection and its associated pose keypoints against geometric,
    /// anatomical, and multi-signal human body structure rules.
    /// </summary>
    PersonCandidateValidationResult ValidateCandidate(
        DetectionResult detection,
        PersonPoseResult? pose,
        int frameWidth,
        int frameHeight,
        int candidateIndex = 0);

    /// <summary>
    /// Batch validates multiple candidate detections and pairs them with valid poses.
    /// Returns only validated human candidates.
    /// </summary>
    IReadOnlyList<ValidatedPersonCandidate> FilterValidCandidates(
        IReadOnlyList<DetectionResult> detections,
        IReadOnlyList<PersonPoseResult>? poses,
        int frameWidth,
        int frameHeight);
}

/// <summary>
/// Represents a validated human candidate paired with its confirmed detection and pose.
/// </summary>
public record ValidatedPersonCandidate
{
    public int OriginalCandidateIndex { get; init; }
    public DetectionResult Detection { get; init; } = default!;
    public PersonPoseResult Pose { get; init; } = default!;
    public PersonCandidateValidationResult ValidationResult { get; init; } = default!;
}
