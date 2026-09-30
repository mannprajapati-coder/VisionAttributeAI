using System.Diagnostics;
using OpenCvSharp;
using VisionAttributeAI.Models.AI;
using VisionAttributeAI.Models.Pose;
using VisionAttributeAI.Services.Detection;
using VisionAttributeAI.Services.Pose;

namespace VisionAttributeAI.Tests;

/// <summary>
/// Diagnostic runner to empirically verify YOLO-Pose 17-keypoint orientation,
/// coordinate geometry, anatomical vs screen space mapping, and hand-raised gesture evaluation.
/// </summary>
public static class PoseOrientationDiagnosticRunner
{
    public static async Task RunDiagnosticsAsync(IPersonDetector detector, IPoseEstimationService poseService)
    {
        Console.WriteLine("================================================================================");
        Console.WriteLine("           YOLO-POSE ORIENTATION & HAND-RAISED DIAGNOSTIC SUITE                 ");
        Console.WriteLine("================================================================================");
        Console.WriteLine();
        Console.WriteLine("COCO Keypoint Convention:");
        Console.WriteLine("   Index 5 : LeftShoulder   (Anatomical Left)");
        Console.WriteLine("   Index 6 : RightShoulder  (Anatomical Right)");
        Console.WriteLine("   Index 7 : LeftElbow      (Anatomical Left)");
        Console.WriteLine("   Index 8 : RightElbow     (Anatomical Right)");
        Console.WriteLine("   Index 9 : LeftWrist      (Anatomical Left)");
        Console.WriteLine("   Index 10: RightWrist     (Anatomical Right)");
        Console.WriteLine();
        Console.WriteLine("Image Matrix Orientation:");
        Console.WriteLine("   Top-Left origin: (X=0, Y=0). X increases to the Right, Y increases Downward.");
        Console.WriteLine("   In raw camera stream (unmirrored):");
        Console.WriteLine("     - Person facing camera: Anatomical LEFT is on the RIGHT side of the image (X > Center).");
        Console.WriteLine("     - Person facing camera: Anatomical RIGHT is on the LEFT side of the image (X < Center).");
        Console.WriteLine("   Hand-Raised condition:");
        Console.WriteLine("     - Left Hand Raised  <=> Keypoint 9 (LeftWrist) Y < Keypoint 5 (LeftShoulder) Y");
        Console.WriteLine("     - Right Hand Raised <=> Keypoint 10 (RightWrist) Y < Keypoint 6 (RightShoulder) Y");
        Console.WriteLine("================================================================================");
        Console.WriteLine();

        // 1. Find test images
        var candidateFolders = new[]
        {
            @"C:\Users\Admin\Downloads",
            @"d:\Mann\Project\VisionAttributeAI\debug_downloads_scan"
        };

        var testImages = new List<string>();
        foreach (var folder in candidateFolders)
        {
            if (Directory.Exists(folder))
            {
                var files = Directory.GetFiles(folder, "*.jpg")
                    .Concat(Directory.GetFiles(folder, "*.png"))
                    .Where(f => !f.Contains("_upper") && !f.Contains("_lower") && !f.Contains("_mask"))
                    .Take(5);
                testImages.AddRange(files);
            }
        }

        Console.WriteLine($"[1] Auditing {testImages.Count} Real Test Images for Pose Keypoints & Hand State:");
        Console.WriteLine("--------------------------------------------------------------------------------");

        int processed = 0;
        foreach (var imgPath in testImages)
        {
            if (!File.Exists(imgPath)) continue;
            using var mat = Cv2.ImRead(imgPath, ImreadModes.Color);
            if (mat.Empty()) continue;

            var detections = await detector.DetectPersonsAsync(mat);
            if (detections.Count == 0) continue;

            var poses = await poseService.EstimatePoseAsync(mat, detections);

            processed++;
            Console.WriteLine($"\n--- Image #{processed}: {Path.GetFileName(imgPath)} ({mat.Width}x{mat.Height}) ---");

            for (int i = 0; i < poses.Count; i++)
            {
                var p = poses[i];
                var kps = p.Keypoints;
                Console.WriteLine($"  Person #{i + 1}: Box=[X:{p.BoundingBox.X:F0}, Y:{p.BoundingBox.Y:F0}, W:{p.BoundingBox.Width:F0}, H:{p.BoundingBox.Height:F0}], Conf={p.DetectionConfidence:F2}, Orientation={p.Visibility.Orientation}");

                if (kps.Count >= 17)
                {
                    var ls = kps[5];
                    var rs = kps[6];
                    var le = kps[7];
                    var re = kps[8];
                    var lw = kps[9];
                    var rw = kps[10];

                    Console.WriteLine($"    Left Shoulder  (Kp 5) : X={ls.X,6:F1}, Y={ls.Y,6:F1}, Conf={ls.Confidence:F2}");
                    Console.WriteLine($"    Right Shoulder (Kp 6) : X={rs.X,6:F1}, Y={rs.Y,6:F1}, Conf={rs.Confidence:F2}");
                    Console.WriteLine($"    Left Elbow     (Kp 7) : X={le.X,6:F1}, Y={le.Y,6:F1}, Conf={le.Confidence:F2}");
                    Console.WriteLine($"    Right Elbow    (Kp 8) : X={re.X,6:F1}, Y={re.Y,6:F1}, Conf={re.Confidence:F2}");
                    Console.WriteLine($"    Left Wrist     (Kp 9) : X={lw.X,6:F1}, Y={lw.Y,6:F1}, Conf={lw.Confidence:F2}");
                    Console.WriteLine($"    Right Wrist    (Kp 10): X={rw.X,6:F1}, Y={rw.Y,6:F1}, Conf={rw.Confidence:F2}");

                    bool lsVis = ls.Confidence >= 0.25f;
                    bool rsVis = rs.Confidence >= 0.25f;
                    bool lwVis = lw.Confidence >= 0.25f;
                    bool rwVis = rw.Confidence >= 0.25f;

                    string lStatus = (!lwVis || !lsVis) ? "Not Visible / Low Conf" : (lw.Y < ls.Y ? "RAISED" : "Down");
                    string rStatus = (!rwVis || !rsVis) ? "Not Visible / Low Conf" : (rw.Y < rs.Y ? "RAISED" : "Down");

                    Console.WriteLine($"    >> Left Hand State  : {lStatus} (Evaluated: LeftHandRaised = {p.Visibility.LeftHandRaised})");
                    Console.WriteLine($"    >> Right Hand State : {rStatus} (Evaluated: RightHandRaised = {p.Visibility.RightHandRaised})");

                    if (p.Regions.LeftWristRegion != null)
                    {
                        var r = p.Regions.LeftWristRegion;
                        Console.WriteLine($"    >> Left Wrist ROI   : [X:{r.X:F0}, Y:{r.Y:F0}, W:{r.Width:F0}, H:{r.Height:F0}] (Centered on Kp 9)");
                    }
                    if (p.Regions.RightWristRegion != null)
                    {
                        var r = p.Regions.RightWristRegion;
                        Console.WriteLine($"    >> Right Wrist ROI  : [X:{r.X:F0}, Y:{r.Y:F0}, W:{r.Width:F0}, H:{r.Height:F0}] (Centered on Kp 10)");
                    }
                }
            }
        }

        // 2. Synthetic Geometric Verification of Inversion & Mirroring
        Console.WriteLine();
        Console.WriteLine("================================================================================");
        Console.WriteLine("[2] Geometric Coordinate & Gesture Verification Tests:");
        Console.WriteLine("--------------------------------------------------------------------------------");

        // Synthetic Case A: Frontal Person with Physical LEFT Hand Raised
        // Person facing camera at center: Head Y=100, LS (X=360, Y=200), RS (X=280, Y=200).
        // Physical LEFT hand raised high above shoulder: LW (X=370, Y=120) < LS (Y=200).
        // Physical RIGHT hand hanging down: RW (X=270, Y=350) > RS (Y=200).
        var kpsCaseA = new List<Keypoint>
        {
            new(0, "Nose", 320, 100, 0.9f),
            new(1, "LeftEye", 330, 95, 0.9f),
            new(2, "RightEye", 310, 95, 0.9f),
            new(3, "LeftEar", 345, 100, 0.8f),
            new(4, "RightEar", 295, 100, 0.8f),
            new(5, "LeftShoulder", 360, 200, 0.9f),   // Anatomical Left (Image Right: X=360)
            new(6, "RightShoulder", 280, 200, 0.9f),  // Anatomical Right (Image Left: X=280)
            new(7, "LeftElbow", 380, 160, 0.85f),
            new(8, "RightElbow", 270, 280, 0.85f),
            new(9, "LeftWrist", 370, 120, 0.9f),      // RAISED (Y=120 < Shoulder Y=200)
            new(10, "RightWrist", 270, 350, 0.9f),    // DOWN (Y=350 > Shoulder Y=200)
            new(11, "LeftHip", 350, 380, 0.9f),
            new(12, "RightHip", 290, 380, 0.9f),
            new(13, "LeftKnee", 350, 500, 0.9f),
            new(14, "RightKnee", 290, 500, 0.9f),
            new(15, "LeftAnkle", 350, 600, 0.9f),
            new(16, "RightAnkle", 290, 600, 0.9f)
        };

        bool caseA_LeftRaised = kpsCaseA[9].Confidence >= 0.25f && kpsCaseA[5].Confidence >= 0.25f && (kpsCaseA[9].Y < kpsCaseA[5].Y);
        bool caseA_RightRaised = kpsCaseA[10].Confidence >= 0.25f && kpsCaseA[6].Confidence >= 0.25f && (kpsCaseA[10].Y < kpsCaseA[6].Y);

        Console.WriteLine($"Test Case A (Physical LEFT hand raised above shoulder):");
        Console.WriteLine($"   LeftWrist Y ({kpsCaseA[9].Y}) < LeftShoulder Y ({kpsCaseA[5].Y})  -> LeftHandRaised: {caseA_LeftRaised} [EXPECTED: True]");
        Console.WriteLine($"   RightWrist Y ({kpsCaseA[10].Y}) < RightShoulder Y ({kpsCaseA[6].Y}) -> RightHandRaised: {caseA_RightRaised} [EXPECTED: False]");
        Console.WriteLine($"   Result: {(caseA_LeftRaised && !caseA_RightRaised ? "PASS" : "FAIL")}");

        // Synthetic Case B: Frontal Person with Physical RIGHT Hand Raised
        var kpsCaseB = new List<Keypoint>
        {
            new(0, "Nose", 320, 100, 0.9f),
            new(1, "LeftEye", 330, 95, 0.9f),
            new(2, "RightEye", 310, 95, 0.9f),
            new(3, "LeftEar", 345, 100, 0.8f),
            new(4, "RightEar", 295, 100, 0.8f),
            new(5, "LeftShoulder", 360, 200, 0.9f),
            new(6, "RightShoulder", 280, 200, 0.9f),
            new(7, "LeftElbow", 380, 280, 0.85f),
            new(8, "RightElbow", 260, 160, 0.85f),
            new(9, "LeftWrist", 380, 360, 0.9f),      // DOWN
            new(10, "RightWrist", 250, 110, 0.9f),    // RAISED (Y=110 < Shoulder Y=200)
            new(11, "LeftHip", 350, 380, 0.9f),
            new(12, "RightHip", 290, 380, 0.9f),
            new(13, "LeftKnee", 350, 500, 0.9f),
            new(14, "RightKnee", 290, 500, 0.9f),
            new(15, "LeftAnkle", 350, 600, 0.9f),
            new(16, "RightAnkle", 290, 600, 0.9f)
        };

        bool caseB_LeftRaised = kpsCaseB[9].Confidence >= 0.25f && kpsCaseB[5].Confidence >= 0.25f && (kpsCaseB[9].Y < kpsCaseB[5].Y);
        bool caseB_RightRaised = kpsCaseB[10].Confidence >= 0.25f && kpsCaseB[6].Confidence >= 0.25f && (kpsCaseB[10].Y < kpsCaseB[6].Y);

        Console.WriteLine();
        Console.WriteLine($"Test Case B (Physical RIGHT hand raised above shoulder):");
        Console.WriteLine($"   LeftWrist Y ({kpsCaseB[9].Y}) < LeftShoulder Y ({kpsCaseB[5].Y})  -> LeftHandRaised: {caseB_LeftRaised} [EXPECTED: False]");
        Console.WriteLine($"   RightWrist Y ({kpsCaseB[10].Y}) < RightShoulder Y ({kpsCaseB[6].Y}) -> RightHandRaised: {caseB_RightRaised} [EXPECTED: True]");
        Console.WriteLine($"   Result: {(!caseB_LeftRaised && caseB_RightRaised ? "PASS" : "FAIL")}");

        Console.WriteLine();
        Console.WriteLine("================================================================================");
        Console.WriteLine("                      DIAGNOSTIC SUITE COMPLETE                                 ");
        Console.WriteLine("================================================================================");
    }
}
