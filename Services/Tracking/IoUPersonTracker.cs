using System.Collections.Concurrent;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using VisionAttributeAI.Models.AI;
using VisionAttributeAI.Models.Pose;
using VisionAttributeAI.Models.Tracking;

namespace VisionAttributeAI.Services.Tracking;

/// <summary>
/// Multi-object person tracker with tentative-to-confirmed lifecycle management,
/// strict spatial & scale gating, ID-switch rejection, and zero public exposure of unconfirmed candidate artifacts.
/// </summary>
public class IoUPersonTracker : IPersonTracker
{
    private readonly LiveTrackingOptions _options;
    private readonly ILogger<IoUPersonTracker> _logger;
    private readonly object _lock = new();

    private readonly Dictionary<int, TrackedPersonState> _activeTracks = new();
    private readonly Dictionary<int, TrackedPersonState> _finalizedTracks = new();
    private int _nextInternalTrackId = 1;
    private int _nextPublicPersonId = 1;

    public IoUPersonTracker(
        IOptions<LiveTrackingOptions> options,
        ILogger<IoUPersonTracker> logger)
    {
        _options = options.Value;
        _logger = logger;
    }

    public IReadOnlyList<TrackedPersonState> UpdateTracks(IReadOnlyList<DetectionResult> detections, bool isOfflineVideo = false, CameraMotionResult? cameraMotion = null)
    {
        var observations = AssociateAndTrack(detections, null, isOfflineVideo, cameraMotion: cameraMotion);
        return observations.Where(o => o.IsConfirmed).Select(o => o.TrackState).ToList();
    }

    public IReadOnlyList<PersonFrameObservation> AssociateAndTrack(
        IReadOnlyList<DetectionResult> detections,
        IReadOnlyList<PersonPoseResult>? poses,
        bool isOfflineVideo = false,
        long frameIndex = 0,
        double timestampSec = 0,
        CameraMotionResult? cameraMotion = null)
    {
        lock (_lock)
        {
            var now = DateTime.UtcNow;

            // 1. Purge stale tracks exceeding timeout (Live camera only; video uses strict missed frame count)
            if (!isOfflineVideo)
            {
                PurgeExpiredTracksInternal(now);
            }

            if (detections == null || detections.Count == 0)
            {
                // Age all active tracks with motion awareness
                AgeAndRetireTracks(new HashSet<TrackedPersonState>(), isOfflineVideo, frameIndex, timestampSec, cameraMotion);
                return Array.Empty<PersonFrameObservation>();
            }

            // 2. Spatial Detection <-> Pose Mapping
            var detectionToPoseMap = new Dictionary<int, PersonPoseResult>();
            if (poses != null && poses.Count > 0)
            {
                var matchedPoseIndices = new HashSet<int>();
                for (int d = 0; d < detections.Count; d++)
                {
                    var det = detections[d];
                    int bestPoseIdx = -1;
                    float bestPoseIoU = 0f;

                    for (int p = 0; p < poses.Count; p++)
                    {
                        if (matchedPoseIndices.Contains(p)) continue;
                        float iou = CalculateIoU(det.Box, poses[p].BoundingBox);
                        if (iou > bestPoseIoU)
                        {
                            bestPoseIoU = iou;
                            bestPoseIdx = p;
                        }
                    }

                    if (bestPoseIdx >= 0 && bestPoseIoU >= 0.15f)
                    {
                        detectionToPoseMap[d] = poses[bestPoseIdx];
                        matchedPoseIndices.Add(bestPoseIdx);
                    }
                    else
                    {
                        detectionToPoseMap[d] = new PersonPoseResult
                        {
                            BoundingBox = det.Box,
                            DetectionConfidence = det.Confidence,
                            Visibility = new BodyVisibilityResult { HeadVisible = true, UpperBodyVisible = true, Orientation = PoseOrientation.Uncertain }
                        };
                    }
                }
            }
            else
            {
                for (int d = 0; d < detections.Count; d++)
                {
                    detectionToPoseMap[d] = new PersonPoseResult
                    {
                        BoundingBox = detections[d].Box,
                        DetectionConfidence = detections[d].Confidence,
                        Visibility = new BodyVisibilityResult { HeadVisible = true, UpperBodyVisible = true, Orientation = PoseOrientation.Uncertain }
                    };
                }
            }

            // 3. Evaluate Match Candidates against Active Tracks (with Camera Motion Compensation)
            var activeTracksList = _activeTracks.Values.Where(t => t.IsActive && !t.IsFinalized).ToList();
            var candidateMatches = new List<MatchCandidate>();

            float matchIouThresh = _options.MatchIouThreshold; // 0.25
            float maxDisplacementRatio = _options.MaxAllowedCenterDisplacementRatio; // 0.35
            float maxAreaRatio = _options.MaxAllowedAreaChangeRatio; // 2.2
            int maxMissedForProx = _options.MaxMissedFramesForProximity; // 2

            for (int t = 0; t < activeTracksList.Count; t++)
            {
                var track = activeTracksList[t];
                var prevBox = track.CurrentBox;

                // Apply Camera Motion Compensation (CMC) to predict new track bounding box
                var evalBox = prevBox;
                if (cameraMotion != null && cameraMotion.IsReliable)
                {
                    evalBox = new BoundingBox(
                        prevBox.X + cameraMotion.DeltaX,
                        prevBox.Y + cameraMotion.DeltaY,
                        prevBox.Width,
                        prevBox.Height);
                }

                float prevH = Math.Max(1f, evalBox.Height);
                float prevW = Math.Max(1f, evalBox.Width);
                float prevCenterX = evalBox.X + (prevW / 2f);
                float prevCenterY = evalBox.Y + (prevH / 2f);
                float prevArea = Math.Max(1f, prevW * prevH);
                float prevAspect = prevW / prevH;

                // Adjust allowed displacement based on missed frame age
                float effectiveMaxDisplacement = maxDisplacementRatio * (track.MissedFrames > 0
                    ? Math.Max(0.50f, 1.0f - (track.MissedFrames * 0.20f))
                    : 1.0f);

                for (int d = 0; d < detections.Count; d++)
                {
                    var det = detections[d];
                    var currBox = det.Box;
                    float currH = Math.Max(1f, currBox.Height);
                    float currW = Math.Max(1f, currBox.Width);
                    float currCenterX = currBox.X + (currW / 2f);
                    float currCenterY = currBox.Y + (currH / 2f);
                    float currArea = Math.Max(1f, currW * currH);
                    float currAspect = currW / currH;

                    float dx = currCenterX - prevCenterX;
                    float dy = currCenterY - prevCenterY;
                    float centerDistance = (float)Math.Sqrt(dx * dx + dy * dy);
                    float normalizedDisplacement = centerDistance / prevH;

                    float areaRatio = Math.Max(prevArea, currArea) / Math.Min(prevArea, currArea);
                    float aspectDiff = Math.Abs(currAspect - prevAspect);
                    float iou = CalculateIoU(evalBox, currBox);

                    // Gating Rules against CMC-compensated position
                    bool passesIoU = iou >= matchIouThresh;
                    bool passesDisplacement = normalizedDisplacement <= effectiveMaxDisplacement;
                    bool passesArea = areaRatio <= maxAreaRatio;
                    bool passesAspect = aspectDiff <= 0.40f;
                    bool canUseProximity = track.MissedFrames <= (maxMissedForProx + (cameraMotion?.MotionState == CameraMotionState.HighMotion ? 2 : 0));

                    bool isCandidate = false;
                    float score = 0f;
                    string rejectReason = string.Empty;
                    bool possibleIdSwitch = false;

                    // Direct IoU Match Path
                    if (passesIoU)
                    {
                        if (passesArea && passesAspect && normalizedDisplacement <= (effectiveMaxDisplacement * 1.5f))
                        {
                            isCandidate = true;
                            score = (iou * 0.70f) + (Math.Max(0f, 1.0f - normalizedDisplacement) * 0.30f);
                        }
                        else
                        {
                            rejectReason = $"IoU passed ({iou:F2}) but failed scale/aspect (AreaRatio: {areaRatio:F1}, AspectDiff: {aspectDiff:F2})";
                        }
                    }
                    // Proximity Fallback Match Path
                    else if (canUseProximity && passesDisplacement && passesArea && passesAspect)
                    {
                        isCandidate = true;
                        score = (Math.Max(0f, 1.0f - (normalizedDisplacement / effectiveMaxDisplacement)) * 0.60f) + (Math.Max(0f, iou) * 0.40f);
                    }
                    else
                    {
                        rejectReason = $"Failed gating (IoU: {iou:F2}, NormDisp: {normalizedDisplacement:F2}/{effectiveMaxDisplacement:F2}, AreaRatio: {areaRatio:F1}, Missed: {track.MissedFrames})";
                    }

                    // Suspicious ID-Switch Check
                    if (isCandidate)
                    {
                        if (iou < 0.15f && normalizedDisplacement > 0.28f)
                        {
                            possibleIdSwitch = true;
                            rejectReason = $"Suspicious match: low IoU ({iou:F2}) and high displacement ({normalizedDisplacement:F2})";
                        }
                        else if (areaRatio > 1.90f && iou < 0.20f)
                        {
                            possibleIdSwitch = true;
                            rejectReason = $"Suspicious match: sudden scale jump ({areaRatio:F1}x) with low IoU ({iou:F2})";
                        }
                        else if (track.MissedFrames >= 2 && normalizedDisplacement > 0.20f && (!cameraMotion?.IsReliable ?? true))
                        {
                            possibleIdSwitch = true;
                            rejectReason = $"Suspicious match: track missed for {track.MissedFrames} frames and reacquired with displacement ({normalizedDisplacement:F2}) without reliable CMC";
                        }
                    }

                    if (isCandidate)
                    {
                        candidateMatches.Add(new MatchCandidate
                        {
                            TrackInternalId = track.TrackAgeFrames,
                            TrackState = track,
                            DetectionIndex = d,
                            Detection = det,
                            IoU = iou,
                            CenterDistance = centerDistance,
                            NormalizedDisplacement = normalizedDisplacement,
                            AreaRatio = areaRatio,
                            Score = score,
                            PossibleIdSwitch = possibleIdSwitch,
                            RejectReason = rejectReason
                        });
                    }
                }
            }

            // 4. Greedy 1-to-1 Association
            candidateMatches.Sort((a, b) => b.Score.CompareTo(a.Score));

            var matchedTrackObjects = new HashSet<TrackedPersonState>();
            var matchedDetectionIndices = new HashSet<int>();
            var frameObservations = new List<PersonFrameObservation>();

            foreach (var match in candidateMatches)
            {
                if (matchedTrackObjects.Contains(match.TrackState) || matchedDetectionIndices.Contains(match.DetectionIndex))
                {
                    continue;
                }

                if (match.PossibleIdSwitch)
                {
                    _logger.LogWarning(
                        "Frame {Frame}: REJECTED candidate association for Track #{TrackId} on Detection #{DetIdx} due to ID-Switch risk: {Reason}. A new tentative track will be generated.",
                        frameIndex, match.TrackState.PersonId, match.DetectionIndex, match.RejectReason);
                    continue;
                }

                var track = match.TrackState;
                var prevBox = track.CurrentBox;
                track.PreviousBox = prevBox;
                track.CurrentBox = match.Detection.Box;
                track.DetectionConfidence = match.Detection.Confidence;
                track.LastSeenTimestampUtc = now;
                track.MissedFrames = 0;
                track.TotalFramesObserved++;
                track.ConfirmationHits++;
                track.LastMatchedIoU = match.IoU;
                track.CenterDisplacement = match.CenterDistance;
                track.NormalizedCenterDistance = match.NormalizedDisplacement;
                track.AreaChangeRatio = match.AreaRatio;
                track.PossibleIdSwitch = false;
                track.PossibleIdSwitchReason = string.Empty;

                // Re-acquisition of TemporarilyLost track (Preserves existing PersonId and Attribute History!)
                if (track.ConfirmationState == TrackConfirmationState.TemporarilyLost)
                {
                    track.ConfirmationState = TrackConfirmationState.Confirmed;
                    _logger.LogInformation(
                        "Frame {Frame} ({Time:F1}s): Confirmed Person #{PersonId} RE-ACQUIRED successfully after motion/blur grace period.",
                        frameIndex, timestampSec, track.PersonId);
                }
                // Track Confirmation Promotion
                else if (!track.IsConfirmed && track.ConfirmationHits >= 2)
                {
                    track.ConfirmationState = TrackConfirmationState.Confirmed;
                    track.PersonId = _nextPublicPersonId++;
                    _logger.LogInformation(
                        "Frame {Frame} ({Time:F1}s): Tentative track PROMOTED to Confirmed Person #{PersonId} after {Hits} observations.",
                        frameIndex, timestampSec, track.PersonId, track.ConfirmationHits);
                }

                matchedTrackObjects.Add(track);
                matchedDetectionIndices.Add(match.DetectionIndex);

                var pose = detectionToPoseMap.GetValueOrDefault(match.DetectionIndex) ?? new PersonPoseResult
                {
                    BoundingBox = match.Detection.Box,
                    DetectionConfidence = match.Detection.Confidence
                };

                frameObservations.Add(new PersonFrameObservation
                {
                    DetectionIndex = match.DetectionIndex,
                    Detection = match.Detection,
                    Pose = pose,
                    MatchedTrackId = track.PersonId,
                    TrackState = track,
                    AssociationScore = match.Score,
                    IoU = match.IoU,
                    CenterDistance = match.CenterDistance,
                    NormalizedDisplacement = match.NormalizedDisplacement,
                    AreaRatio = match.AreaRatio,
                    PossibleIdSwitch = false,
                    AssociationReason = track.IsConfirmed ? "Confirmed 1-to-1 Match" : "Tentative Track Match",
                    IsNewTrack = false,
                    AssociationAccepted = true
                });
            }

            // 5. Age and Retire Unmatched Existing Tracks with Motion Awareness
            AgeAndRetireTracks(matchedTrackObjects, isOfflineVideo, frameIndex, timestampSec, cameraMotion);

            // 6. Create Clean New Tracks for Unmatched Detections
            for (int d = 0; d < detections.Count; d++)
            {
                if (!matchedDetectionIndices.Contains(d))
                {
                    var det = detections[d];
                    int internalId = _nextInternalTrackId++;

                    // Single image mode promotes immediately; video/camera begins as Tentative
                    bool immediateConfirm = !isOfflineVideo && frameIndex == 0;
                    int assignedPublicId = immediateConfirm ? _nextPublicPersonId++ : 0;
                    var confirmState = immediateConfirm ? TrackConfirmationState.Confirmed : TrackConfirmationState.Tentative;

                    var newTrack = new TrackedPersonState(assignedPublicId, det.Box, det.Confidence)
                    {
                        ConfirmationState = confirmState,
                        ConfirmationHits = 1,
                        FirstSeenTimestampUtc = now,
                        LastSeenTimestampUtc = now,
                        LastMatchedIoU = 1.0f,
                        CenterDisplacement = 0f,
                        NormalizedCenterDistance = 0f,
                        AreaChangeRatio = 1.0f,
                        PossibleIdSwitch = false,
                        PossibleIdSwitchReason = "New Candidate Initialized"
                    };

                    _activeTracks[internalId] = newTrack;
                    matchedDetectionIndices.Add(d);

                    var pose = detectionToPoseMap.GetValueOrDefault(d) ?? new PersonPoseResult
                    {
                        BoundingBox = det.Box,
                        DetectionConfidence = det.Confidence
                    };

                    frameObservations.Add(new PersonFrameObservation
                    {
                        DetectionIndex = d,
                        Detection = det,
                        Pose = pose,
                        MatchedTrackId = newTrack.PersonId,
                        TrackState = newTrack,
                        AssociationScore = 1.0f,
                        IoU = 1.0f,
                        CenterDistance = 0f,
                        NormalizedDisplacement = 0f,
                        AreaRatio = 1.0f,
                        PossibleIdSwitch = false,
                        AssociationReason = immediateConfirm ? "New Confirmed Person Initialized" : "New Tentative Candidate Initialized",
                        IsNewTrack = true,
                        AssociationAccepted = true
                    });

                    _logger.LogDebug(
                        "Frame {Frame} ({Time:F1}s): New {State} Candidate initialized at [{Box}] (Conf: {Conf:P1})",
                        frameIndex, timestampSec, confirmState, det.Box, det.Confidence);
                }
            }

            return frameObservations.OrderBy(o => o.DetectionIndex).ToList();
        }
    }

    private void AgeAndRetireTracks(HashSet<TrackedPersonState> matchedTracks, bool isOfflineVideo, long frameIndex, double timestampSec, CameraMotionResult? cameraMotion)
    {
        int maxAllowedMissed = isOfflineVideo ? _options.VideoMaxMissedFrames : _options.MaxMissedFrames;
        bool isHighMotion = cameraMotion?.MotionState == CameraMotionState.HighMotion;
        int effectiveMaxMissed = isHighMotion ? maxAllowedMissed + 3 : maxAllowedMissed;

        var toRetire = new List<int>();
        var toDiscardTentative = new List<int>();

        foreach (var (internalId, track) in _activeTracks)
        {
            if (!track.IsActive || track.IsFinalized) continue;

            if (!matchedTracks.Contains(track))
            {
                track.MissedFrames++;

                // If tentative and missed without confirmation, discard silently
                if (track.ConfirmationState == TrackConfirmationState.Tentative && track.MissedFrames >= 2)
                {
                    toDiscardTentative.Add(internalId);
                }
                else if (track.ConfirmationState == TrackConfirmationState.Confirmed && isHighMotion)
                {
                    // Transition to TemporarilyLost state during high camera motion
                    track.ConfirmationState = TrackConfirmationState.TemporarilyLost;
                    _logger.LogDebug("Frame {Frame}: Confirmed Person #{PersonId} transitioned to TemporarilyLost due to camera motion.", frameIndex, track.PersonId);
                }
                
                if (track.MissedFrames > effectiveMaxMissed)
                {
                    toRetire.Add(internalId);
                }
            }
        }

        foreach (int internalId in toDiscardTentative)
        {
            if (_activeTracks.TryGetValue(internalId, out var track))
            {
                track.IsActive = false;
                _activeTracks.Remove(internalId);
                _logger.LogDebug("Frame {Frame}: Discarded unconfirmed tentative candidate after {Missed} missed frames.", frameIndex, track.MissedFrames);
            }
        }

        foreach (int internalId in toRetire)
        {
            if (_activeTracks.TryGetValue(internalId, out var track))
            {
                track.IsActive = false;
                track.IsFinalized = true;
                track.ConfirmationState = TrackConfirmationState.Retired;
                if (track.PersonId > 0)
                {
                    _finalizedTracks[track.PersonId] = track;
                    _logger.LogInformation(
                        "Frame {Frame} ({Time:F1}s): Person #{PersonId} retired and finalized after {Missed} missed frames.",
                        frameIndex, timestampSec, track.PersonId, track.MissedFrames);
                }
            }
        }
    }

    public IReadOnlyList<TrackedPersonState> GetActiveTracks()
    {
        lock (_lock)
        {
            return _activeTracks.Values.Where(t => t.IsActive && !t.IsFinalized && t.IsConfirmed).ToList();
        }
    }

    public IReadOnlyList<TrackedPersonState> GetFinalizedTracks()
    {
        lock (_lock)
        {
            return _finalizedTracks.Values.Where(t => t.IsConfirmed).ToList();
        }
    }

    public IReadOnlyList<TrackedPersonState> GetAllTracks()
    {
        lock (_lock)
        {
            var combined = new Dictionary<int, TrackedPersonState>(_finalizedTracks);
            foreach (var kvp in _activeTracks.Values.Where(t => t.IsConfirmed && t.PersonId > 0))
            {
                combined[kvp.PersonId] = kvp;
            }
            return combined.Values.OrderBy(t => t.PersonId).ToList();
        }
    }

    public void PurgeExpiredTracks()
    {
        lock (_lock)
        {
            PurgeExpiredTracksInternal(DateTime.UtcNow);
        }
    }

    private void PurgeExpiredTracksInternal(DateTime now)
    {
        var toRemove = new List<int>();
        foreach (var (internalId, track) in _activeTracks)
        {
            double elapsedSeconds = (now - track.LastSeenTimestampUtc).TotalSeconds;
            if (track.MissedFrames > _options.MaxMissedFrames || elapsedSeconds > _options.TrackTimeoutSeconds)
            {
                toRemove.Add(internalId);
            }
        }

        foreach (int internalId in toRemove)
        {
            if (_activeTracks.TryGetValue(internalId, out var track))
            {
                track.IsActive = false;
                track.IsFinalized = true;
                if (track.IsConfirmed && track.PersonId > 0)
                {
                    _finalizedTracks[track.PersonId] = track;
                }
                _activeTracks.Remove(internalId);
            }
        }
    }

    public void Reset()
    {
        lock (_lock)
        {
            _activeTracks.Clear();
            _finalizedTracks.Clear();
            _nextInternalTrackId = 1;
            _nextPublicPersonId = 1;
            _logger.LogInformation("Reset Person Tracker state, active tracks, and finalized histories.");
        }
    }

    private static float CalculateIoU(BoundingBox boxA, BoundingBox boxB)
    {
        float xA = Math.Max(boxA.X, boxB.X);
        float yA = Math.Max(boxA.Y, boxB.Y);
        float xB = Math.Min(boxA.X + boxA.Width, boxB.X + boxB.Width);
        float yB = Math.Min(boxA.Y + boxA.Height, boxB.Y + boxB.Height);

        float interArea = Math.Max(0, xB - xA) * Math.Max(0, yB - yA);
        if (interArea <= 0) return 0.0f;

        float boxAArea = boxA.Width * boxA.Height;
        float boxBArea = boxB.Width * boxB.Height;
        float unionArea = boxAArea + boxBArea - interArea;

        return unionArea > 0 ? interArea / unionArea : 0.0f;
    }

    private class MatchCandidate
    {
        public int TrackInternalId { get; set; }
        public TrackedPersonState TrackState { get; set; } = default!;
        public int DetectionIndex { get; set; }
        public DetectionResult Detection { get; set; } = default!;
        public float IoU { get; set; }
        public float CenterDistance { get; set; }
        public float NormalizedDisplacement { get; set; }
        public float AreaRatio { get; set; }
        public float Score { get; set; }
        public bool PossibleIdSwitch { get; set; }
        public string RejectReason { get; set; } = string.Empty;
    }
}
