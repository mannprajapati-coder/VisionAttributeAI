using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using OpenCvSharp;
using VisionAttributeAI.Models.AI;
using VisionAttributeAI.Models.Pose;
using VisionAttributeAI.Services.Attributes;
using VisionAttributeAI.Services.Detection;
using VisionAttributeAI.Services.PersonAnalysis;
using VisionAttributeAI.Services.Pose;

namespace VisionAttributeAI.Tests;

public static class DownloadsColorAudit
{
    public static async Task RunAuditAsync(
        IPersonDetector detector,
        IPoseEstimationService poseService,
        IUnifiedPersonAnalysisService unifiedService,
        IFashionClipService clipService,
        string outDir = "debug_downloads_scan")
    {
        Directory.CreateDirectory(outDir);
        Console.WriteLine("\n=========================================================================================");
        Console.WriteLine(" COMPLETE DOWNLOADS DIRECTORY & REAL IMAGE COLOR AUDIT");
        Console.WriteLine("=========================================================================================\n");

        string dlPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads");
        string userUpPath = @"C:\Users\Admin\.gemini\antigravity\brain\d1c30ac8-9746-4946-b173-1b32a3a64125\.user_uploaded";

        var files = new List<string>();
        if (Directory.Exists(dlPath))
        {
            files.AddRange(Directory.GetFiles(dlPath, "*.*")
                .Where(f => f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) ||
                            f.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase) ||
                            f.EndsWith(".jfif", StringComparison.OrdinalIgnoreCase) ||
                            f.EndsWith(".png", StringComparison.OrdinalIgnoreCase)));
        }

        if (Directory.Exists(userUpPath))
        {
            files.AddRange(Directory.GetFiles(userUpPath, "*.*")
                .Where(f => f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) ||
                            f.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase) ||
                            f.EndsWith(".jfif", StringComparison.OrdinalIgnoreCase) ||
                            f.EndsWith(".png", StringComparison.OrdinalIgnoreCase)));
        }

        Console.WriteLine($"Found {files.Count} image files to scan.\n");

        int processed = 0;
        int personDetectedCount = 0;

        foreach (var file in files)
        {
            string fileName = Path.GetFileName(file);
            if (fileName.Contains("Dashboard") || fileName.Contains("Cause Tree") || fileName.Contains("11mb") || fileName.Contains("empty"))
            {
                continue; // Skip dashboard screenshots and empty road images
            }

            try
            {
                using var img = Cv2.ImRead(file);
                if (img.Empty()) continue;

                processed++;
                var dets = await detector.DetectPersonsAsync(img);
                if (dets.Count == 0) continue;

                personDetectedCount++;
                var poses = await poseService.EstimatePoseAsync(img, dets);

                for (int i = 0; i < dets.Count; i++)
                {
                    var d = dets[i];
                    var pose = (i < poses.Count) ? poses[i] : null;

                    var uBox = pose?.Regions?.UpperTorsoRegion ?? d.Box;
                    var lBox = pose?.Regions?.LowerBodyRegion;

                    // Crop upper & lower
                    int ux = Math.Clamp((int)uBox.X, 0, img.Width - 1);
                    int uy = Math.Clamp((int)uBox.Y, 0, img.Height - 1);
                    int uw = Math.Clamp((int)uBox.Width, 1, img.Width - ux);
                    int uh = Math.Clamp((int)uBox.Height, 1, img.Height - uy);
                    using var uRoi = new Mat(img, new Rect(ux, uy, uw, uh));

                    // Evaluate Upper Color with both Broken and Simplified Robust
                    var brokenSvc = new ClothingColorService();
                    var brokenUpper = brokenSvc.ClassifyDetailedColor(uRoi, img, "Upper", pose, uBox);
                    string robustUpper = ColorRootCauseDiagnosticSuite.PredictSimplifiedRobust(uRoi, img, "Upper", pose, uBox);
                    string hsvUpper = ColorRootCauseDiagnosticSuite.PredictOriginalBaseline(uRoi);

                    string brokenLower = "None";
                    string robustLower = "None";
                    string hsvLower = "None";

                    if (lBox != null)
                    {
                        int lx = Math.Clamp((int)lBox.X, 0, img.Width - 1);
                        int ly = Math.Clamp((int)lBox.Y, 0, img.Height - 1);
                        int lw = Math.Clamp((int)lBox.Width, 1, img.Width - lx);
                        int lh = Math.Clamp((int)lBox.Height, 1, img.Height - ly);
                        using var lRoi = new Mat(img, new Rect(lx, ly, lw, lh));

                        brokenLower = brokenSvc.ClassifyDetailedColor(lRoi, img, "Lower", pose, lBox).PrimaryColor;
                        robustLower = ColorRootCauseDiagnosticSuite.PredictSimplifiedRobust(lRoi, img, "Lower", pose, lBox);
                        hsvLower = ColorRootCauseDiagnosticSuite.PredictOriginalBaseline(lRoi);
                    }

                    // Fashion-CLIP garment types
                    var uType = await clipService.ClassifyUpperClothingTypeAsync(uRoi);

                    Console.WriteLine($"[{fileName} | Person #{i + 1}]");
                    Console.WriteLine($"   Garment Type : Upper='{uType.TopCategory}' (Conf: {uType.TopConfidence * 100:F0}%)");
                    Console.WriteLine($"   Upper Color  : Broken='{brokenUpper.PrimaryColor}' | HSVBaseline='{hsvUpper}' | Robust='{robustUpper}'");
                    Console.WriteLine($"   Lower Color  : Broken='{brokenLower}' | HSVBaseline='{hsvLower}' | Robust='{robustLower}'");

                    // Save crops for visual inspection
                    string baseName = Path.GetFileNameWithoutExtension(fileName).Replace(" ", "_");
                    uRoi.SaveImage(Path.Combine(outDir, $"{baseName}_p{i + 1}_upper.jpg"));
                    if (lBox != null)
                    {
                        int lx = Math.Clamp((int)lBox.X, 0, img.Width - 1);
                        int ly = Math.Clamp((int)lBox.Y, 0, img.Height - 1);
                        int lw = Math.Clamp((int)lBox.Width, 1, img.Width - lx);
                        int lh = Math.Clamp((int)lBox.Height, 1, img.Height - ly);
                        using var lRoi = new Mat(img, new Rect(lx, ly, lw, lh));
                        lRoi.SaveImage(Path.Combine(outDir, $"{baseName}_p{i + 1}_lower.jpg"));
                    }
                    Console.WriteLine();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[ERR on {fileName}]: {ex.Message}");
            }
        }

        Console.WriteLine($"\nScanned {processed} images. Detected people in {personDetectedCount} images.");
    }
}
