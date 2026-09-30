using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using OpenCvSharp;
using VisionAttributeAI.DTOs.LiveCamera;
using VisionAttributeAI.Models.AI;
using VisionAttributeAI.Models.Pose;
using VisionAttributeAI.Models.Tracking;
using VisionAttributeAI.Services.Aggregation;
using VisionAttributeAI.Services.Detection;
using VisionAttributeAI.Services.PersonAnalysis;
using VisionAttributeAI.Services.Pose;
using VisionAttributeAI.Services.Quality;
using VisionAttributeAI.Services.Tracking;
using VisionAttributeAI.Services.Validation;

namespace VisionAttributeAI.Tests;

public enum CameraMotionCategory
{
    Stable,
    ModerateMotion,
    HighMotion
}

public enum FrameSharpnessCategory
{
    Sharp,
    Blurry
}

public enum FrameQualityQuadrant
{
    Stable_Sharp,
    Moving_Sharp,
    Moving_Blurry,
    Stable_Blurry
}

public enum FailureClassification
{
    None,
    CaseA_DetectorFailure,
    CaseB_TrackerFailure,
    CaseC_BothDetectorAndTrackerFailure,
    CaseD_FalseDetection
}

public class FrameDiagnosticLog
{
    public int ScenarioIndex { get; set; }
    public string ScenarioName { get; set; } = string.Empty;
    public int FrameNumber { get; set; }
    public double TimestampSec { get; set; }
    public double LaplacianVariance { get; set; }
    public FrameSharpnessCategory SharpnessCategory { get; set; }
    public double CameraMotionScore { get; set; } // 0..1
    public CameraMotionCategory MotionCategory { get; set; }
    public FrameQualityQuadrant Quadrant { get; set; }

    // Real world truth
    public int GroundTruthPersonCount { get; set; }
    public List<BoundingBox> GroundTruthBoxes { get; set; } = new();

    // YOLO Detector outputs
    public int RawDetectionCount { get; set; }
    public int PostNmsDetectionCount { get; set; }
    public List<DetectionResult> Detections { get; set; } = new();

    // Tracker outputs
    public List<int> ActivePersonIds { get; set; } = new();
    public int MatchedDetectionsCount { get; set; }
    public int UnmatchedDetectionsCount { get; set; }
    public int UnmatchedTracksCount { get; set; }
    public float LastMatchedIoU { get; set; }
    public float LastCenterDisplacement { get; set; }
    public float LastNormalizedDisplacement { get; set; }
    public int TrackMissedCount { get; set; }
    public List<int> NewIdsCreated { get; set; } = new();
    public List<int> IdsExpired { get; set; } = new();

    // Pose & Quality
    public bool ValidPoseFound { get; set; }
    public bool CropQualityPassed { get; set; }
    public string CropQualityReason { get; set; } = string.Empty;

    // Display / UI outputs
    public int VisibleDetectionCount { get; set; }
    public int ActiveTrackCount { get; set; }
    public int DisplayedPersonCount { get; set; }

    // Failure Classification
    public FailureClassification FailureType { get; set; }
    public string FailureReason { get; set; } = string.Empty;
}

public class ScenarioSummary
{
    public string ScenarioName { get; set; } = string.Empty;
    public int TotalFrames { get; set; }
    public int GroundTruthPersonFrames { get; set; }
    public int YoloDetectionsTotal { get; set; }
    public int YoloMissesTotal { get; set; }
    public double YoloMissRate { get; set; }
    public int TrackerIdSwitches { get; set; }
    public int NewFalseIdsCreated { get; set; }
    public int FalsePersonDetections { get; set; }
    public int CaseACount { get; set; }
    public int CaseBCount { get; set; }
    public int CaseCCount { get; set; }
    public int CaseDCount { get; set; }
    public double AvgSharpnessOnFailures { get; set; }
    public double AvgCameraMotionOnFailures { get; set; }
    public List<FrameDiagnosticLog> Logs { get; set; } = new();
}

public static class CameraMotionDiagnosticRunner
{
    public static LiveTrackingOptions GetDefaultTrackingOptions() => new LiveTrackingOptions
    {
        MatchIouThreshold = 0.25f,
        MaxAllowedCenterDisplacementRatio = 0.35f,
        MaxAllowedAreaChangeRatio = 2.2f,
        MinDetectionConfidence = 0.40f,
        MinCropWidth = 32,
        MinCropHeight = 64,
        MinCropArea = 2048,
        MinSharpnessVariance = 45.0,
        MaxMissedFrames = 10,
        VideoMaxMissedFrames = 4,
        MaxMissedFramesForProximity = 2,
        TrackTimeoutSeconds = 3.0
    };

    public static PersonAnalysisOptions GetDefaultPersonOptions() => new PersonAnalysisOptions
    {
        MinTorsoWidth = 28,
        MinTorsoHeight = 40,
        MinLowerWidth = 28,
        MinLowerHeight = 45,
        MinHeadWidth = 20,
        MinHeadHeight = 24,
        ObservationWindowSize = 15,
        MinimumValidFrames = 3,
        MinimumConsensusRatio = 0.60f
    };

    /// <summary>
    /// Computes global camera motion between two consecutive frames using downscaled grayscale frame difference
    /// and normalized phase displacement.
    /// </summary>
    public static double ComputeCameraMotionScore(Mat prevFrame, Mat currFrame)
    {
        if (prevFrame == null || currFrame == null || prevFrame.Empty() || currFrame.Empty())
            return 0.0;

        try
        {
            using var prevSmall = new Mat();
            using var currSmall = new Mat();

            // Downscale to 128x96 for ultra-fast, robust global motion estimation
            Cv2.Resize(prevFrame, prevSmall, new Size(128, 96));
            Cv2.Resize(currFrame, currSmall, new Size(128, 96));

            using var prevGray = new Mat();
            using var currGray = new Mat();

            if (prevSmall.Channels() == 1) prevSmall.CopyTo(prevGray);
            else Cv2.CvtColor(prevSmall, prevGray, ColorConversionCodes.BGR2GRAY);

            if (currSmall.Channels() == 1) currSmall.CopyTo(currGray);
            else Cv2.CvtColor(currSmall, currGray, ColorConversionCodes.BGR2GRAY);

            using var diff = new Mat();
            Cv2.Absdiff(prevGray, currGray, diff);

            Scalar meanDiff = Cv2.Mean(diff);
            // Normalization: mean grayscale change of ~40+ represents intense global camera sweep
            double rawMotion = meanDiff.Val0 / 40.0;
            return Math.Clamp(rawMotion, 0.0, 1.0);
        }
        catch
        {
            return 0.0;
        }
    }

    public static double ComputeLaplacianVariance(Mat frame)
    {
        if (frame == null || frame.Empty()) return 0.0;
        try
        {
            using var gray = new Mat();
            if (frame.Channels() == 1) frame.CopyTo(gray);
            else Cv2.CvtColor(frame, gray, ColorConversionCodes.BGR2GRAY);

            using var laplacian = new Mat();
            Cv2.Laplacian(gray, laplacian, MatType.CV_64F);

            Cv2.MeanStdDev(laplacian, out _, out Scalar stddev);
            return stddev.Val0 * stddev.Val0;
        }
        catch
        {
            return 0.0;
        }
    }

    public static CameraMotionCategory CategorizeMotion(double motionScore)
    {
        if (motionScore < 0.20) return CameraMotionCategory.Stable;
        if (motionScore < 0.55) return CameraMotionCategory.ModerateMotion;
        return CameraMotionCategory.HighMotion;
    }

    public static FrameSharpnessCategory CategorizeSharpness(double laplacianVariance)
    {
        return laplacianVariance >= 45.0 ? FrameSharpnessCategory.Sharp : FrameSharpnessCategory.Blurry;
    }

    public static FrameQualityQuadrant CategorizeQuadrant(double motionScore, double laplacianVariance)
    {
        bool isMoving = motionScore >= 0.20;
        bool isSharp = laplacianVariance >= 45.0;

        if (!isMoving && isSharp) return FrameQualityQuadrant.Stable_Sharp;
        if (isMoving && isSharp) return FrameQualityQuadrant.Moving_Sharp;
        if (isMoving && !isSharp) return FrameQualityQuadrant.Moving_Blurry;
        return FrameQualityQuadrant.Stable_Blurry;
    }

    public static ScenarioSummary RunSimulation(
        string scenarioName,
        int totalFrames,
        Func<int, (Mat Frame, List<BoundingBox> GroundTruth, List<DetectionResult> SimulatedDetections, bool ValidPose)> frameGenerator)
    {
        var trackingOptions = GetDefaultTrackingOptions();
        var tracker = new IoUPersonTracker(Options.Create(trackingOptions), NullLogger<IoUPersonTracker>.Instance);
        var qualityFilter = new FrameQualityFilter(Options.Create(trackingOptions), NullLogger<FrameQualityFilter>.Instance);
        var aggregator = new TemporalAppearanceAggregator(Options.Create(GetDefaultPersonOptions()), NullLogger<TemporalAppearanceAggregator>.Instance);

        var summary = new ScenarioSummary { ScenarioName = scenarioName, TotalFrames = totalFrames };
        Mat? prevMat = null;

        int previousObservedConfirmedId = -1;
        var knownConfirmedIds = new HashSet<int>();

        for (int frameIdx = 0; frameIdx < totalFrames; frameIdx++)
        {
            double timestampSec = frameIdx * 0.10; // 10 FPS
            var (currMat, groundTruthList, detections, validPose) = frameGenerator(frameIdx);

            double sharpness = ComputeLaplacianVariance(currMat);
            double motionScore = prevMat != null ? ComputeCameraMotionScore(prevMat, currMat) : 0.0;
            var motionCat = CategorizeMotion(motionScore);
            var sharpCat = CategorizeSharpness(sharpness);
            var quadrant = CategorizeQuadrant(motionScore, sharpness);

            summary.GroundTruthPersonFrames += groundTruthList.Count;
            summary.YoloDetectionsTotal += detections.Count;

            // Track associations
            var observations = tracker.AssociateAndTrack(
                detections,
                null,
                isOfflineVideo: false,
                frameIndex: frameIdx,
                timestampSec: timestampSec);

            var activeTracks = tracker.GetActiveTracks();
            var allTracks = tracker.GetAllTracks();

            var log = new FrameDiagnosticLog
            {
                ScenarioIndex = 0,
                ScenarioName = scenarioName,
                FrameNumber = frameIdx,
                TimestampSec = timestampSec,
                LaplacianVariance = sharpness,
                SharpnessCategory = sharpCat,
                CameraMotionScore = motionScore,
                MotionCategory = motionCat,
                Quadrant = quadrant,
                GroundTruthPersonCount = groundTruthList.Count,
                GroundTruthBoxes = groundTruthList,
                RawDetectionCount = detections.Count,
                PostNmsDetectionCount = detections.Count,
                Detections = detections,
                ActivePersonIds = activeTracks.Select(t => t.PersonId).ToList(),
                MatchedDetectionsCount = observations.Count(o => !o.IsNewTrack && o.AssociationAccepted),
                UnmatchedDetectionsCount = observations.Count(o => o.IsNewTrack),
                UnmatchedTracksCount = activeTracks.Count(t => t.MissedFrames > 0),
                ValidPoseFound = validPose,
                VisibleDetectionCount = observations.Count(o => o.IsConfirmed && o.MatchedTrackId > 0),
                ActiveTrackCount = activeTracks.Count,
                DisplayedPersonCount = observations.Count(o => o.IsConfirmed && o.MatchedTrackId > 0) // UI strictly uses confirmed detections in frame
            };

            // Calculate IoU & Displacement details from observations
            if (observations.Count > 0)
            {
                var bestObs = observations.OrderByDescending(o => o.AssociationScore).First();
                log.LastMatchedIoU = bestObs.IoU;
                log.LastCenterDisplacement = bestObs.CenterDistance;
                log.LastNormalizedDisplacement = bestObs.NormalizedDisplacement;
            }

            // Quality evaluation on detections
            if (detections.Count > 0)
            {
                var qRes = qualityFilter.EvaluateCropQuality(currMat, detections[0].Box, detections[0].Confidence);
                log.CropQualityPassed = qRes.IsAccepted;
                log.CropQualityReason = qRes.RejectionReason ?? string.Empty;
            }

            // Track IDs Created & ID switches
            var newlyConfirmed = observations.Where(o => o.IsConfirmed && !knownConfirmedIds.Contains(o.MatchedTrackId)).Select(o => o.MatchedTrackId).ToList();
            foreach (var nid in newlyConfirmed)
            {
                knownConfirmedIds.Add(nid);
                log.NewIdsCreated.Add(nid);
            }

            // Analyze Failures
            if (groundTruthList.Count > 0 && detections.Count == 0)
            {
                // Detector Failure
                log.FailureType = FailureClassification.CaseA_DetectorFailure;
                log.FailureReason = "YOLO missed person due to high motion blur or severe camera shift";
                summary.YoloMissesTotal++;
                summary.CaseACount++;
            }
            else if (groundTruthList.Count > 0 && detections.Count > 0)
            {
                // Check if tracker broke continuity for the same continuous entity
                var confirmedObs = observations.FirstOrDefault(o => o.IsConfirmed && o.MatchedTrackId > 0);
                if (confirmedObs != null)
                {
                    if (previousObservedConfirmedId > 0 && confirmedObs.MatchedTrackId != previousObservedConfirmedId)
                    {
                        log.FailureType = FailureClassification.CaseB_TrackerFailure;
                        log.FailureReason = $"ID-Switch: Track changed from Person #{previousObservedConfirmedId} to Person #{confirmedObs.MatchedTrackId} because IoU fell below gate ({log.LastMatchedIoU:F2} < 0.25) or normalized displacement ({log.LastNormalizedDisplacement:F2} > 0.35)";
                        summary.TrackerIdSwitches++;
                        summary.CaseBCount++;
                    }
                    previousObservedConfirmedId = confirmedObs.MatchedTrackId;
                }
                else if (previousObservedConfirmedId > 0 && observations.All(o => !o.IsConfirmed))
                {
                    // Track was demoted/lost into tentative state
                    log.FailureType = FailureClassification.CaseB_TrackerFailure;
                    log.FailureReason = "Detection re-acquired as tentative; active PersonId temporarily disappeared from display.";
                    summary.CaseBCount++;
                }
            }
            else if (groundTruthList.Count == 0 && detections.Count > 0)
            {
                log.FailureType = FailureClassification.CaseD_FalseDetection;
                log.FailureReason = "YOLO generated false positive detection on motion artifact";
                summary.FalsePersonDetections++;
                summary.CaseDCount++;
            }

            summary.Logs.Add(log);
            prevMat?.Dispose();
            prevMat = currMat.Clone();
        }

        prevMat?.Dispose();

        summary.NewFalseIdsCreated = Math.Max(0, knownConfirmedIds.Count - 1); // For single person scenarios
        summary.YoloMissRate = summary.GroundTruthPersonFrames > 0
            ? (double)summary.YoloMissesTotal / summary.GroundTruthPersonFrames
            : 0.0;

        var failedLogs = summary.Logs.Where(l => l.FailureType != FailureClassification.None).ToList();
        summary.AvgSharpnessOnFailures = failedLogs.Count > 0 ? failedLogs.Average(l => l.LaplacianVariance) : 0;
        summary.AvgCameraMotionOnFailures = failedLogs.Count > 0 ? failedLogs.Average(l => l.CameraMotionScore) : 0;

        return summary;
    }

    // =========================================================================
    // SCENARIO DEFINITIONS (TEST A to TEST F)
    // =========================================================================

    // TEST A: Camera completely stationary (Person stationary in frame)
    public static ScenarioSummary RunTestA()
    {
        return RunSimulation("TEST A: Camera Stationary", 20, frameIdx =>
        {
            var mat = new Mat(480, 640, MatType.CV_8UC3, new Scalar(180, 180, 180));
            // Draw clear synthetic person figure
            Cv2.Rectangle(mat, new Rect(250, 100, 120, 300), new Scalar(50, 50, 200), -1);
            Cv2.Circle(mat, new Point(310, 80), 30, new Scalar(180, 150, 120), -1);

            var gt = new List<BoundingBox> { new BoundingBox(250, 50, 120, 350) };
            // YOLO detects consistently with high confidence
            var dets = new List<DetectionResult> { new DetectionResult(new BoundingBox(250, 50, 120, 350), 0.92f, "person") };
            return (mat, gt, dets, true);
        });
    }

    // TEST B: Camera slowly pans left/right (displacement 6-12px per frame)
    public static ScenarioSummary RunTestB()
    {
        return RunSimulation("TEST B: Camera Slowly Pans", 25, frameIdx =>
        {
            var mat = new Mat(480, 640, MatType.CV_8UC3, new Scalar(180, 180, 180));
            // Person shifts slowly across frame due to gentle panning (e.g. 8px per frame)
            float x = 200f + (frameIdx * 8f);
            Cv2.Rectangle(mat, new Rect((int)x, 100, 120, 300), new Scalar(50, 50, 200), -1);
            Cv2.Circle(mat, new Point((int)x + 60, 80), 30, new Scalar(180, 150, 120), -1);

            var gt = new List<BoundingBox> { new BoundingBox(x, 50, 120, 350) };
            var dets = new List<DetectionResult> { new DetectionResult(new BoundingBox(x, 50, 120, 350), 0.89f, "person") };
            return (mat, gt, dets, true);
        });
    }

    // TEST C: Camera quickly pans left/right (fast swipe: 80-140px jump + motion blur during swipe)
    public static ScenarioSummary RunTestC()
    {
        return RunSimulation("TEST C: Camera Quickly Pans", 25, frameIdx =>
        {
            var mat = new Mat(480, 640, MatType.CV_8UC3, new Scalar(180, 180, 180));

            float x = 200f;
            bool isSwiping = frameIdx >= 8 && frameIdx <= 12;

            if (isSwiping)
            {
                // Fast swipe: frame jumps by 75px per frame, high blur
                x = 200f + ((frameIdx - 7) * 75f);
                // Heavy horizontal motion blur
                Cv2.Rectangle(mat, new Rect(Math.Max(0, (int)x), 100, 120, 300), new Scalar(100, 100, 160), -1);
                Cv2.GaussianBlur(mat, mat, new Size(25, 25), 15);
            }
            else if (frameIdx > 12)
            {
                x = 200f + (5 * 75f); // Settled at new position (575)
                Cv2.Rectangle(mat, new Rect((int)x, 100, 120, 300), new Scalar(50, 50, 200), -1);
            }
            else
            {
                Cv2.Rectangle(mat, new Rect((int)x, 100, 120, 300), new Scalar(50, 50, 200), -1);
            }

            var gt = new List<BoundingBox> { new BoundingBox(x, 50, 120, 350) };
            var dets = new List<DetectionResult>();

            if (isSwiping)
            {
                // Frame 9 & 10: motion blur causes YOLO to drop confidence or miss completely (Case A)
                if (frameIdx == 9 || frameIdx == 10)
                {
                    // YOLO missed (Detection count = 0)
                }
                else
                {
                    // YOLO detects with lower confidence but bbox displaced by 75px
                    dets.Add(new DetectionResult(new BoundingBox(x, 50, 120, 350), 0.62f, "person"));
                }
            }
            else
            {
                dets.Add(new DetectionResult(new BoundingBox(x, 50, 120, 350), 0.90f, "person"));
            }

            return (mat, gt, dets, !isSwiping);
        });
    }

    // TEST D: Camera quickly moves and then becomes stationary
    public static ScenarioSummary RunTestD()
    {
        return RunSimulation("TEST D: Fast Move Then Stationary", 25, frameIdx =>
        {
            var mat = new Mat(480, 640, MatType.CV_8UC3, new Scalar(180, 180, 180));
            float x;
            bool isJumping = frameIdx == 6 || frameIdx == 7;

            if (frameIdx < 6)
            {
                x = 150f;
            }
            else if (isJumping)
            {
                x = 150f + ((frameIdx - 5) * 120f); // Fast 120px step
                Cv2.GaussianBlur(mat, mat, new Size(17, 17), 9);
            }
            else
            {
                x = 390f; // New stable position
            }

            Cv2.Rectangle(mat, new Rect((int)x, 100, 120, 300), new Scalar(50, 50, 200), -1);

            var gt = new List<BoundingBox> { new BoundingBox(x, 50, 120, 350) };
            var dets = new List<DetectionResult>();

            // YOLO detects throughout, but position jumped abruptly by 120px
            dets.Add(new DetectionResult(new BoundingBox(x, 50, 120, 350), isJumping ? 0.70f : 0.91f, "person"));

            return (mat, gt, dets, true);
        });
    }

    // TEST E: Person moves while camera remains stationary (Normal walking)
    public static ScenarioSummary RunTestE()
    {
        return RunSimulation("TEST E: Person Moves, Camera Still", 25, frameIdx =>
        {
            var mat = new Mat(480, 640, MatType.CV_8UC3, new Scalar(180, 180, 180));
            float x = 120f + (frameIdx * 14f); // Walking at 14px/frame
            Cv2.Rectangle(mat, new Rect((int)x, 100, 120, 300), new Scalar(50, 50, 200), -1);

            var gt = new List<BoundingBox> { new BoundingBox(x, 50, 120, 350) };
            var dets = new List<DetectionResult> { new DetectionResult(new BoundingBox(x, 50, 120, 350), 0.88f, "person") };
            return (mat, gt, dets, true);
        });
    }

    // TEST F: Person and camera both move (Opposite directions -> relative speed 90px/frame)
    public static ScenarioSummary RunTestF()
    {
        return RunSimulation("TEST F: Both Move (High Relative Motion)", 25, frameIdx =>
        {
            var mat = new Mat(480, 640, MatType.CV_8UC3, new Scalar(180, 180, 180));
            float x;
            bool isIntense = frameIdx >= 7 && frameIdx <= 11;

            if (isIntense)
            {
                x = 100f + ((frameIdx - 6) * 90f); // 90px relative displacement per frame
                Cv2.GaussianBlur(mat, mat, new Size(19, 19), 11);
            }
            else if (frameIdx > 11)
            {
                x = 550f;
            }
            else
            {
                x = 100f + (frameIdx * 10f);
            }

            Cv2.Rectangle(mat, new Rect(Math.Min(520, (int)x), 100, 120, 300), new Scalar(50, 50, 200), -1);

            var gt = new List<BoundingBox> { new BoundingBox(x, 50, 120, 350) };
            var dets = new List<DetectionResult>();

            if (frameIdx == 8)
            {
                // YOLO temporary dropout under blur (Case A)
            }
            else
            {
                dets.Add(new DetectionResult(new BoundingBox(x, 50, 120, 350), isIntense ? 0.65f : 0.90f, "person"));
            }

            return (mat, gt, dets, !isIntense);
        });
    }

    public static void RunFullInvestigation()
    {
        var summaries = new List<ScenarioSummary>
        {
            RunTestA(),
            RunTestB(),
            RunTestC(),
            RunTestD(),
            RunTestE(),
            RunTestF()
        };

        Console.WriteLine("================================================================================");
        Console.WriteLine("        LIVE CAMERA MOTION & TRACKING EMPIRICAL INVESTIGATION REPORT           ");
        Console.WriteLine("================================================================================");

        foreach (var s in summaries)
        {
            Console.WriteLine($"\n--- SCENARIO: {s.ScenarioName} ---");
            Console.WriteLine($"Total Frames: {s.TotalFrames} | GT Person Observations: {s.GroundTruthPersonFrames}");
            Console.WriteLine($"YOLO Detections: {s.YoloDetectionsTotal} | YOLO Misses: {s.YoloMissesTotal} (Miss Rate: {s.YoloMissRate:P1})");
            Console.WriteLine($"Tracker ID Switches: {s.TrackerIdSwitches} | New False IDs: {s.NewFalseIdsCreated}");
            Console.WriteLine($"Case A (Detector Failures): {s.CaseACount}");
            Console.WriteLine($"Case B (Tracker Failures):  {s.CaseBCount}");
            Console.WriteLine($"Case C (Both / Compounded): {s.CaseCCount}");
            Console.WriteLine($"Case D (False Detections):  {s.CaseDCount}");
            if (s.AvgSharpnessOnFailures > 0)
                Console.WriteLine($"Avg Sharpness on Failures: {s.AvgSharpnessOnFailures:F1} | Avg Camera Motion Score: {s.AvgCameraMotionOnFailures:F2}");

            Console.WriteLine("\nFrame Breakdown (Sample Critical Frames):");
            Console.WriteLine("Frame | Time(s) | Sharpness | CamMotion | Quadrant       | YOLO Det | Tracker PersonIDs | IoU   | Disp(px) | Failure / Note");
            Console.WriteLine("-----------------------------------------------------------------------------------------------------------------------------");

            foreach (var l in s.Logs.Where((x, idx) => x.FailureType != FailureClassification.None || idx % 4 == 0))
            {
                string ids = l.ActivePersonIds.Count > 0 ? string.Join(",", l.ActivePersonIds) : "None";
                string note = l.FailureType != FailureClassification.None ? $"[{l.FailureType}] {l.FailureReason}" : "Normal";
                if (note.Length > 45) note = note.Substring(0, 42) + "...";

                Console.WriteLine($"{l.FrameNumber,5} | {l.TimestampSec,7:F1} | {l.LaplacianVariance,9:F1} | {l.CameraMotionScore,9:F2} | {l.Quadrant,-14} | {l.RawDetectionCount,8} | {ids,-17} | {l.LastMatchedIoU,5:F2} | {l.LastCenterDisplacement,8:F1} | {note}");
            }
        }
    }
}
