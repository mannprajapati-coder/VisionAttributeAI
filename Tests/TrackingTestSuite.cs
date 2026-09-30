using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using VisionAttributeAI.Models.AI;
using VisionAttributeAI.Models.Pose;
using VisionAttributeAI.Models.Tracking;
using VisionAttributeAI.Models.Validation;
using VisionAttributeAI.Services.Tracking;
using VisionAttributeAI.Services.Validation;

namespace VisionAttributeAI.Tests;

/// <summary>
/// Comprehensive empirical unit test suite verifying Scenarios A through L for multi-person tracking & candidate validation correctness.
/// </summary>
public static class TrackingTestSuite
{
    public static List<TestResult> RunAllTests()
    {
        var results = new List<TestResult>();

        results.Add(TestA_SinglePersonRemainsInFrame());
        results.Add(TestB_PersonLeavesThenNewPersonEntersFiveSecondsLater());
        results.Add(TestC_TwoPersonsCrossPaths());
        results.Add(TestD_PersonTemporarilyOccluded());
        results.Add(TestE_PersonLeavesPermanentlyAndTrackFinalized());
        results.Add(TestF_NewPersonEntersAtExitLocation());
        results.Add(TestG_TwoPeopleSimilarAppearanceDoNotMerge());
        results.Add(TestH_PersonRapidlyApproachesCamera());
        results.Add(TestI_RejectSmallIsolatedHandOrCupCrop());
        results.Add(TestJ_TentativeTrackNeverPromotedIfSingleFrame());
        results.Add(TestK_ValidFullPersonPassed());
        results.Add(TestL_DistantPersonAcceptedWithSufficientEvidence());
        results.Add(TestM_CloseUpWebcamBustPersonAccepted());
        results.Add(TestN_RejectGhostBackgroundSmallBox());

        return results;
    }

    private static LiveTrackingOptions CreateOptions() => new LiveTrackingOptions
    {
        MatchIouThreshold = 0.25f,
        MaxAllowedCenterDisplacementRatio = 0.35f,
        MaxAllowedAreaChangeRatio = 2.2f,
        MaxMissedFrames = 10,
        VideoMaxMissedFrames = 4,
        MaxMissedFramesForProximity = 2
    };

    // TEST A: One person remains in frame. Expected: Exactly 1 PersonId throughout.
    public static TestResult TestA_SinglePersonRemainsInFrame()
    {
        var tracker = new IoUPersonTracker(Options.Create(CreateOptions()), NullLogger<IoUPersonTracker>.Instance);
        var seenIds = new HashSet<int>();

        for (int frame = 0; frame < 30; frame++)
        {
            float x = 100f + (frame * 2f); // slow walking
            var box = new BoundingBox(x, 200f, 80f, 200f);
            var obs = tracker.AssociateAndTrack(new[] { new DetectionResult(box, 0.85f, "person") }, null, isOfflineVideo: true, frameIndex: frame);
            foreach (var o in obs)
            {
                if (o.IsConfirmed) seenIds.Add(o.MatchedTrackId);
            }
        }

        bool passed = seenIds.Count == 1 && seenIds.Contains(1);
        return new TestResult("TEST A: Single Person In Frame", passed, $"Unique Person IDs observed: {seenIds.Count} (Expected: 1)");
    }

    // TEST B: Person A leaves. 5 seconds later Person B enters same location. Expected: 2 distinct PersonIds.
    public static TestResult TestB_PersonLeavesThenNewPersonEntersFiveSecondsLater()
    {
        var tracker = new IoUPersonTracker(Options.Create(CreateOptions()), NullLogger<IoUPersonTracker>.Instance);

        // Person A present frames 0..5
        for (int frame = 0; frame < 5; frame++)
        {
            var box = new BoundingBox(150f, 200f, 80f, 200f);
            tracker.AssociateAndTrack(new[] { new DetectionResult(box, 0.85f, "person") }, null, isOfflineVideo: true, frameIndex: frame);
        }

        // Empty frames (5 seconds = 15 frames at 3fps)
        for (int frame = 5; frame < 20; frame++)
        {
            tracker.AssociateAndTrack(Array.Empty<DetectionResult>(), null, isOfflineVideo: true, frameIndex: frame);
        }

        // Person B enters at same location at frames 20 and 21 (confirmed on frame 21)
        tracker.AssociateAndTrack(new[] { new DetectionResult(new BoundingBox(155f, 205f, 80f, 200f), 0.88f, "person") }, null, isOfflineVideo: true, frameIndex: 20);
        var obsB = tracker.AssociateAndTrack(new[] { new DetectionResult(new BoundingBox(155f, 205f, 80f, 200f), 0.88f, "person") }, null, isOfflineVideo: true, frameIndex: 21);

        int personBId = obsB[0].MatchedTrackId;
        bool passed = personBId == 2 && obsB[0].IsConfirmed;
        return new TestResult("TEST B: Stale Location Re-entry after 5s", passed, $"Person A ID=1, Person B ID={personBId} (Expected: ID 2, zero ID reuse)");
    }

    // TEST C: Person A and Person B cross paths. Expected: Histories remain independent.
    public static TestResult TestC_TwoPersonsCrossPaths()
    {
        var tracker = new IoUPersonTracker(Options.Create(CreateOptions()), NullLogger<IoUPersonTracker>.Instance);

        // Person A moves left-to-right (100 -> 300), Person B moves right-to-left (300 -> 100)
        for (int frame = 0; frame < 20; frame++)
        {
            float ax = 100f + (frame * 10f);
            float bx = 300f - (frame * 10f);

            var dets = new[]
            {
                new DetectionResult(new BoundingBox(ax, 200f, 60f, 180f), 0.85f, "person"),
                new DetectionResult(new BoundingBox(bx, 200f, 60f, 180f), 0.85f, "person")
            };

            var obs = tracker.AssociateAndTrack(dets, null, isOfflineVideo: true, frameIndex: frame);
        }

        var allTracks = tracker.GetAllTracks();
        bool passed = allTracks.Count == 2;
        return new TestResult("TEST C: Two Persons Crossing Paths", passed, $"Total Tracks: {allTracks.Count} (Expected: 2)");
    }

    // TEST D: Person temporarily disappears behind an obstacle (2 frames). Expected: Recover original identity.
    public static TestResult TestD_PersonTemporarilyOccluded()
    {
        var tracker = new IoUPersonTracker(Options.Create(CreateOptions()), NullLogger<IoUPersonTracker>.Instance);

        // Seen frames 0..3
        for (int frame = 0; frame < 3; frame++)
        {
            tracker.AssociateAndTrack(new[] { new DetectionResult(new BoundingBox(100f + frame * 5f, 200f, 70f, 190f), 0.85f, "person") }, null, isOfflineVideo: true, frameIndex: frame);
        }

        // Occluded for 2 frames (within VideoMaxMissedFrames = 4)
        tracker.AssociateAndTrack(Array.Empty<DetectionResult>(), null, isOfflineVideo: true, frameIndex: 3);
        tracker.AssociateAndTrack(Array.Empty<DetectionResult>(), null, isOfflineVideo: true, frameIndex: 4);

        // Reappears close to last position
        var obsReappear = tracker.AssociateAndTrack(new[] { new DetectionResult(new BoundingBox(120f, 200f, 70f, 190f), 0.85f, "person") }, null, isOfflineVideo: true, frameIndex: 5);

        int reacquiredId = obsReappear[0].MatchedTrackId;
        bool passed = reacquiredId == 1;
        return new TestResult("TEST D: Brief 2-Frame Occlusion Recovery", passed, $"Reacquired ID: {reacquiredId} (Expected: 1)");
    }

    // TEST E: Person leaves permanently. Expected: Track finalized and never modified again.
    public static TestResult TestE_PersonLeavesPermanentlyAndTrackFinalized()
    {
        var tracker = new IoUPersonTracker(Options.Create(CreateOptions()), NullLogger<IoUPersonTracker>.Instance);

        // Frame 0..2
        for (int frame = 0; frame < 3; frame++)
        {
            tracker.AssociateAndTrack(new[] { new DetectionResult(new BoundingBox(100f, 200f, 70f, 190f), 0.85f, "person") }, null, isOfflineVideo: true, frameIndex: frame);
        }

        // Person leaves (5 missed frames > VideoMaxMissedFrames: 4)
        for (int frame = 3; frame < 9; frame++)
        {
            tracker.AssociateAndTrack(Array.Empty<DetectionResult>(), null, isOfflineVideo: true, frameIndex: frame);
        }

        var finalized = tracker.GetFinalizedTracks();
        var active = tracker.GetActiveTracks();

        bool passed = finalized.Count == 1 && finalized[0].PersonId == 1 && finalized[0].IsFinalized && active.Count == 0;
        return new TestResult("TEST E: Track Finalization & Freeze", passed, $"Finalized Tracks: {finalized.Count}, Active: {active.Count} (Expected: Finalized=1, Active=0)");
    }

    // TEST F: New person enters exactly where previous person exited after expiration. Expected: New PersonId.
    public static TestResult TestF_NewPersonEntersAtExitLocation()
    {
        var tracker = new IoUPersonTracker(Options.Create(CreateOptions()), NullLogger<IoUPersonTracker>.Instance);

        // Person 1 exits at X=500
        for (int frame = 0; frame < 3; frame++)
        {
            tracker.AssociateAndTrack(new[] { new DetectionResult(new BoundingBox(500f, 200f, 70f, 190f), 0.85f, "person") }, null, isOfflineVideo: true, frameIndex: frame);
        }

        // 6 empty frames (track retires)
        for (int frame = 3; frame < 9; frame++)
        {
            tracker.AssociateAndTrack(Array.Empty<DetectionResult>(), null, isOfflineVideo: true, frameIndex: frame);
        }

        // Person 2 appears at X=500 (confirmed on 2nd hit)
        tracker.AssociateAndTrack(new[] { new DetectionResult(new BoundingBox(500f, 200f, 70f, 190f), 0.85f, "person") }, null, isOfflineVideo: true, frameIndex: 10);
        var obs = tracker.AssociateAndTrack(new[] { new DetectionResult(new BoundingBox(500f, 200f, 70f, 190f), 0.85f, "person") }, null, isOfflineVideo: true, frameIndex: 11);

        bool passed = obs[0].MatchedTrackId == 2 && obs[0].IsConfirmed;
        return new TestResult("TEST F: New Person at Old Exit Location", passed, $"Assigned ID: {obs[0].MatchedTrackId} (Expected: 2)");
    }

    // TEST G: Two people have similar clothing. Expected: Independent spatial tracking without merging.
    public static TestResult TestG_TwoPeopleSimilarAppearanceDoNotMerge()
    {
        var tracker = new IoUPersonTracker(Options.Create(CreateOptions()), NullLogger<IoUPersonTracker>.Instance);

        for (int frame = 0; frame < 10; frame++)
        {
            var dets = new[]
            {
                new DetectionResult(new BoundingBox(100f, 200f, 60f, 180f), 0.85f, "person"),
                new DetectionResult(new BoundingBox(400f, 200f, 60f, 180f), 0.85f, "person")
            };
            tracker.AssociateAndTrack(dets, null, isOfflineVideo: true, frameIndex: frame);
        }

        var active = tracker.GetActiveTracks();
        bool passed = active.Count == 2 && active[0].PersonId == 1 && active[1].PersonId == 2;
        return new TestResult("TEST G: Spatial Independence of Two People", passed, $"Active Tracks: {active.Count} (Expected: 2)");
    }

    // TEST H: Person rapidly approaches camera (bounding box growth). Expected: Smooth tracking without false ID switch.
    public static TestResult TestH_PersonRapidlyApproachesCamera()
    {
        var tracker = new IoUPersonTracker(Options.Create(CreateOptions()), NullLogger<IoUPersonTracker>.Instance);
        var seenIds = new HashSet<int>();

        // Box grows from height 150 to height 400 smoothly over 15 frames
        for (int frame = 0; frame < 15; frame++)
        {
            float h = 150f + (frame * 16f);
            float w = h * 0.40f;
            float x = 200f - (w / 2f);
            float y = 400f - h;

            var obs = tracker.AssociateAndTrack(new[] { new DetectionResult(new BoundingBox(x, y, w, h), 0.90f, "person") }, null, isOfflineVideo: true, frameIndex: frame);
            foreach (var o in obs)
            {
                if (o.IsConfirmed) seenIds.Add(o.MatchedTrackId);
            }
        }

        bool passed = seenIds.Count == 1 && seenIds.Contains(1);
        return new TestResult("TEST H: Rapid Camera Approach (Box Growth)", passed, $"Unique IDs: {seenIds.Count} (Expected: 1)");
    }

    // TEST I: Small square box around a cup / hand with no torso or head keypoints -> REJECTED.
    public static TestResult TestI_RejectSmallIsolatedHandOrCupCrop()
    {
        var options = Options.Create(new PersonAnalysisOptions());
        var validator = new PersonCandidateValidator(options, NullLogger<PersonCandidateValidator>.Instance);
        
        // Small 40x45 box, ratio ~0.89, with only isolated wrist keypoint
        var det = new DetectionResult(new BoundingBox(100f, 100f, 40f, 45f), 0.45f, "person");
        var pose = new PersonPoseResult
        {
            BoundingBox = det.Box,
            DetectionConfidence = det.Confidence,
            Keypoints = new List<Keypoint>
            {
                new Keypoint(9, "left_wrist", 110f, 120f, 0.65f)
            },
            Visibility = new BodyVisibilityResult { LeftWristVisible = true }
        };

        var result = validator.ValidateCandidate(det, pose, 1920, 1080);
        bool passed = !result.IsValidPerson && result.Tier == PersonCandidateTier.Rejected;
        return new TestResult("TEST I: Reject Isolated Hand/Cup Crop", passed, $"IsValid: {result.IsValidPerson}, Reason: {result.RejectReason}");
    }

    // TEST J: Single-frame noise candidate is never promoted to Confirmed PersonId.
    public static TestResult TestJ_TentativeTrackNeverPromotedIfSingleFrame()
    {
        var tracker = new IoUPersonTracker(Options.Create(CreateOptions()), NullLogger<IoUPersonTracker>.Instance);

        // Frame 0: Candidate appears
        var obs0 = tracker.AssociateAndTrack(new[] { new DetectionResult(new BoundingBox(100f, 100f, 80f, 200f), 0.85f, "person") }, null, isOfflineVideo: true, frameIndex: 0);

        bool frame0NotConfirmed = !obs0[0].IsConfirmed && obs0[0].MatchedTrackId == 0;

        // Frames 1..5: Candidate disappears immediately
        for (int f = 1; f <= 5; f++)
        {
            tracker.AssociateAndTrack(Array.Empty<DetectionResult>(), null, isOfflineVideo: true, frameIndex: f);
        }

        var active = tracker.GetActiveTracks();
        var finalized = tracker.GetFinalizedTracks();

        // Stale unconfirmed track is dropped silently with 0 confirmed finalized tracks
        bool passed = frame0NotConfirmed && active.Count == 0 && finalized.Count == 0;
        return new TestResult("TEST J: Tentative Noise Track Purged Without Public ID", passed, $"Frame 0 IsConfirmed: {obs0[0].IsConfirmed}, Final Confirmed Count: {finalized.Count} (Expected: 0)");
    }

    // TEST K: Full valid standing person with head, shoulders, hips, knees -> ACCEPTED.
    public static TestResult TestK_ValidFullPersonPassed()
    {
        var options = Options.Create(new PersonAnalysisOptions());
        var validator = new PersonCandidateValidator(options, NullLogger<PersonCandidateValidator>.Instance);

        var det = new DetectionResult(new BoundingBox(100f, 100f, 100f, 300f), 0.88f, "person");
        var pose = new PersonPoseResult
        {
            BoundingBox = det.Box,
            DetectionConfidence = det.Confidence,
            Keypoints = new List<Keypoint>
            {
                new Keypoint(0, "nose", 150f, 120f, 0.9f),
                new Keypoint(5, "left_shoulder", 130f, 150f, 0.85f),
                new Keypoint(6, "right_shoulder", 170f, 150f, 0.85f),
                new Keypoint(11, "left_hip", 135f, 230f, 0.80f),
                new Keypoint(12, "right_hip", 165f, 230f, 0.80f),
                new Keypoint(13, "left_knee", 135f, 300f, 0.75f),
                new Keypoint(14, "right_knee", 165f, 300f, 0.75f)
            },
            Visibility = new BodyVisibilityResult { HeadVisible = true, UpperBodyVisible = true, LowerBodyVisible = true }
        };

        var result = validator.ValidateCandidate(det, pose, 1920, 1080);
        bool passed = result.IsValidPerson && (result.Tier == PersonCandidateTier.StrongPerson || result.Tier == PersonCandidateTier.ProbablePerson);
        return new TestResult("TEST K: Valid Full Person Validation", passed, $"IsValid: {result.IsValidPerson}, Tier: {result.Tier}");
    }

    // TEST L: Distant person (small height/area) with vertical human aspect ratio and torso keypoints -> ACCEPTED.
    public static TestResult TestL_DistantPersonAcceptedWithSufficientEvidence()
    {
        var options = Options.Create(new PersonAnalysisOptions());
        var validator = new PersonCandidateValidator(options, NullLogger<PersonCandidateValidator>.Instance);

        // Distant person box: 25x70 px (aspect ratio 0.35)
        var det = new DetectionResult(new BoundingBox(500f, 300f, 25f, 70f), 0.75f, "person");
        var pose = new PersonPoseResult
        {
            BoundingBox = det.Box,
            DetectionConfidence = det.Confidence,
            Keypoints = new List<Keypoint>
            {
                new Keypoint(0, "nose", 512f, 305f, 0.70f),
                new Keypoint(5, "left_shoulder", 505f, 315f, 0.65f),
                new Keypoint(6, "right_shoulder", 520f, 315f, 0.65f),
                new Keypoint(11, "left_hip", 507f, 335f, 0.60f),
                new Keypoint(12, "right_hip", 518f, 335f, 0.60f)
            },
            Visibility = new BodyVisibilityResult { HeadVisible = true, UpperBodyVisible = true }
        };

        var result = validator.ValidateCandidate(det, pose, 1920, 1080);
        bool passed = result.IsValidPerson;
        return new TestResult("TEST L: Distant Person Accepted With Torso Keypoints", passed, $"IsValid: {result.IsValidPerson}, Tier: {result.Tier}");
    }

    // TEST M: Close-up webcam bust crop (horizontal aspect ratio ~1.40, nose + shoulders keypoints) -> ACCEPTED.
    public static TestResult TestM_CloseUpWebcamBustPersonAccepted()
    {
        var options = Options.Create(new PersonAnalysisOptions());
        var validator = new PersonCandidateValidator(options, NullLogger<PersonCandidateValidator>.Instance);

        // Webcam close-up: 480x340 (ratio 1.41) in 640x480 frame
        var det = new DetectionResult(new BoundingBox(80f, 100f, 480f, 340f), 0.72f, "person");
        var pose = new PersonPoseResult
        {
            BoundingBox = det.Box,
            DetectionConfidence = det.Confidence,
            Keypoints = new List<Keypoint>
            {
                new Keypoint(0, "nose", 320f, 180f, 0.88f),
                new Keypoint(1, "left_eye", 290f, 160f, 0.85f),
                new Keypoint(2, "right_eye", 350f, 160f, 0.85f),
                new Keypoint(5, "left_shoulder", 180f, 290f, 0.80f),
                new Keypoint(6, "right_shoulder", 460f, 290f, 0.80f)
            },
            Visibility = new BodyVisibilityResult { HeadVisible = true, UpperBodyVisible = true }
        };

        var result = validator.ValidateCandidate(det, pose, 640, 480);
        bool passed = result.IsValidPerson && (result.Tier == PersonCandidateTier.StrongPerson || result.Tier == PersonCandidateTier.ProbablePerson);
        return new TestResult("TEST M: Close-up Webcam Bust Person Accepted", passed, $"IsValid: {result.IsValidPerson}, Tier: {result.Tier}, AspectRatio: {result.AspectRatio:F2}");
    }

    // TEST N: Ghost detection in distant background (low confidence, 0 keypoints, small box) -> REJECTED.
    public static TestResult TestN_RejectGhostBackgroundSmallBox()
    {
        var options = Options.Create(new PersonAnalysisOptions());
        var validator = new PersonCandidateValidator(options, NullLogger<PersonCandidateValidator>.Instance);

        // Small 30x50 distant box, 0 keypoints, low confidence 0.42
        var det = new DetectionResult(new BoundingBox(550f, 150f, 30f, 50f), 0.42f, "person");
        var pose = new PersonPoseResult
        {
            BoundingBox = det.Box,
            DetectionConfidence = det.Confidence,
            Keypoints = new List<Keypoint>(),
            Visibility = new BodyVisibilityResult()
        };

        var result = validator.ValidateCandidate(det, pose, 1920, 1080);
        bool passed = !result.IsValidPerson && result.Tier == PersonCandidateTier.Rejected;
        return new TestResult("TEST N: Reject Ghost Background Small Box", passed, $"IsValid: {result.IsValidPerson}, Reason: {result.RejectReason}");
    }
}

public record TestResult(string Name, bool Passed, string Details);
