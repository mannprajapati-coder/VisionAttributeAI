using VisionAttributeAI.Models.AI;
using VisionAttributeAI.Models.Pose;
using VisionAttributeAI.Models.Tracking;

namespace VisionAttributeAI.Services.Tracking;

/// <summary>
/// Contract for multi-object person tracking across consecutive camera frames and video sequences.
/// </summary>
public interface IPersonTracker
{
    /// <summary>
    /// Updates tracking states with new YOLO detections and returns matched/created active tracks.
    /// </summary>
    IReadOnlyList<TrackedPersonState> UpdateTracks(IReadOnlyList<DetectionResult> detections, bool isOfflineVideo = false);

    /// <summary>
    /// Explicit 1-to-1 association that binds each YOLO detection to its matching YOLO-Pose keypoints/regions
    /// and associates with active multi-object tracks under strict gating rules.
    /// </summary>
    IReadOnlyList<PersonFrameObservation> AssociateAndTrack(
        IReadOnlyList<DetectionResult> detections,
        IReadOnlyList<PersonPoseResult>? poses,
        bool isOfflineVideo = false,
        long frameIndex = 0,
        double timestampSec = 0);

    /// <summary>
    /// Gets all currently active tracked persons.
    /// </summary>
    IReadOnlyList<TrackedPersonState> GetActiveTracks();

    /// <summary>
    /// Gets all finalized (retired) tracked persons whose lifetime has concluded.
    /// </summary>
    IReadOnlyList<TrackedPersonState> GetFinalizedTracks();

    /// <summary>
    /// Gets all tracked persons (active + finalized).
    /// </summary>
    IReadOnlyList<TrackedPersonState> GetAllTracks();

    /// <summary>
    /// Purges expired tracks based on missed frame counts and timeout.
    /// </summary>
    void PurgeExpiredTracks();

    /// <summary>
    /// Resets all active tracks and clears tracking state.
    /// </summary>
    void Reset();
}
