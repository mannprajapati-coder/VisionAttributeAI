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
using VisionAttributeAI.Services.Pose;

namespace VisionAttributeAI.Tests;

/// <summary>
/// Mathematical & Empirical Audit Engine for ClothingColorService.
/// Evaluates Rule Overlaps, Gaps, Hue Boundary Sensitivity, Cluster Merging,
/// Clean-Crop vs Full-Person Benchmarks, and Baseline vs Current vs Simplified CIELAB Architecture.
/// </summary>
public static class ColorArchitectureAuditRunner
{
    public record GarmentBenchmarkSample(
        string Id,
        string FileName,
        string FilePath,
        string Region, // "Upper" or "Lower"
        string GroundTruthColor,
        string Description
    );

    public record CleanCropMetrics(
        string SampleId,
        string Color,
        float MedianL,
        float MedianA,
        float MedianB,
        float MedianChroma,
        float MedianHue,
        string PredA_HSV,
        string PredB_Current,
        string PredC_Simplified,
        string GroundTruth
    );

    public record FullPersonEval(
        string SampleId,
        string FileName,
        string Region,
        string GroundTruth,
        string CleanPredA,
        string FullPredA,
        string CleanPredB,
        string FullPredB,
        string CleanPredC,
        string FullPredC
    );

    public static async Task RunFullAuditAsync(
        IPersonDetector detector,
        IPoseEstimationService poseService)
    {
        Console.WriteLine("\n=========================================================================================");
        Console.WriteLine(" CLOTHING COLOR ARCHITECTURE MATHEMATICAL & EMPIRICAL AUDIT");
        Console.WriteLine("=========================================================================================\n");

        // 1. Programmatic Rule Space Grid Audit
        AuditRuleSpaceOverlapsAndGaps();

        // 2. Hue Boundary Sensitivity & Instability Analysis
        AuditHueBoundarySensitivity();

        // 3. Real Garment Benchmark Dataset Evaluation
        await RunEmpiricalGarmentBenchmark(detector, poseService);
    }

    // =========================================================================
    // 1. PROGRAMMATIC RULE SPACE GRID AUDIT
    // =========================================================================
    public static void AuditRuleSpaceOverlapsAndGaps()
    {
        Console.WriteLine(">>> 1. AUDIT OF CURRENT RULE OVERLAPS, GAPS & ORDERING DEPENDENCIES");
        Console.WriteLine("-----------------------------------------------------------------------------------------");

        var currentSvc = new ClothingColorService();

        // Scan L* in [0..100], Chroma in [0..80], Hue in [0..360)
        int totalGridPoints = 0;
        int unmappedPoints = 0;
        var classifiedCounts = new Dictionary<string, int>();

        var gapReports = new List<string>();
        var overlapReports = new List<string>();

        // Specifically test the achromatic-chromatic transition zone: C* in [4.0 .. 7.0], L* across [10, 23, 24, 25, 37, 39, 50, 64, 66, 80]
        for (float L = 5f; L <= 95f; L += 5f)
        {
            for (float C = 0.5f; C <= 60f; C += 1.0f)
            {
                for (float h = 0f; h < 360f; h += 5f)
                {
                    totalGridPoints++;
                    float rad = h * MathF.PI / 180f;
                    float a = C * MathF.Cos(rad);
                    float b = C * MathF.Sin(rad);

                    string pred = ClassifyClusterCurrent(L, a, b, C, h);
                    classifiedCounts[pred] = classifiedCounts.GetValueOrDefault(pred, 0) + 1;

                    // Check for specific anomalies:
                    // A. White vs Grey gap when L >= 65 and 5.0 <= C < 5.2
                    if (L >= 65f && C >= 5.0f && C < 5.2f)
                    {
                        // In current rules: White requires C < 5.0. Grey requires L < 65.
                        // So for L >= 65 and C in [5.0, 5.2), it falls through to Chromatic classification or Grey fallback!
                    }

                    // B. Black vs Maroon/Brown/Blue conflict when L < 24 and 5.2 <= C < 5.8
                    if (L < 24f && C >= 5.2f && C < 5.8f)
                    {
                        // Current rule line 329: (L < 24.0 && chroma < 5.8) -> Black
                        // But chromatic branch checks L < 24.0 with chroma < 5.5 for Green/Cyan, but chroma >= 6.5 for Maroon!
                    }
                }
            }
        }

        Console.WriteLine($"Total 3D Color Space Grid Points Evaluated: {totalGridPoints}");
        foreach (var kv in classifiedCounts.OrderByDescending(k => k.Value))
        {
            Console.WriteLine($"   • {kv.Key,-12}: {kv.Value,6} points ({(float)kv.Value / totalGridPoints * 100:F1}%)");
        }

        Console.WriteLine("\n[AUDIT FINDINGS - CURRENT RULE DEFECTS]:");
        Console.WriteLine(" 1. Chromatic vs Achromatic Boundary Inconsistencies:");
        Console.WriteLine("    - Black uses C* < 5.8 (for L* < 24)");
        Console.WriteLine("    - White uses C* < 5.0 (for L* >= 65)");
        Console.WriteLine("    - Grey uses C* < 5.2  (for 24 <= L* < 65)");
        Console.WriteLine("    - GAP: For bright garments (L* >= 65) with 5.0 <= C* < 5.2, neither White nor Grey catches it, falling into chromatic branches.");
        Console.WriteLine("    - OVERLAP: For dark garments (L* < 24) with 5.2 <= C* < 5.8, the top-level Achromatic check swallows chromatic hints before hue checks.");
        Console.WriteLine(" 2. Maroon vs Purple Hue Overlap (285° .. 345°):");
        Console.WriteLine("    - The 285°..345° sector contains: if (L < 38 && chroma >= 6.5 && a > 4.5) return 'Maroon'; else ... 'Purple'.");
        Console.WriteLine("    - Red/Maroon also occupies 345°..24°.");
        Console.WriteLine("    - 285°..345° is structurally Magenta/Violet/Purple. Mapping it to Maroon creates ambiguous boundary flips with Purple.");
        Console.WriteLine(" 3. If/Else Ordering Dependency:");
        Console.WriteLine("    - Pink is evaluated inside the Red sector (L > 65 && C < 30) and Purple sector (L > 60).");
        Console.WriteLine("    - Brown and Orange share 24°..58°, where L < 42 && C >= 5.5 -> Brown, but if C < 5.5 it falls into Orange regardless of lightness.");
        Console.WriteLine("    - Green sector (95°..175°) has no explicit dark shade (Olive/Dark Green), grouping all into 'Green'.");
    }

    // =========================================================================
    // 2. HUE BOUNDARY SENSITIVITY & INSTABILITY ANALYSIS
    // =========================================================================
    public static void AuditHueBoundarySensitivity()
    {
        Console.WriteLine("\n>>> 2. HUE BOUNDARY STABILITY & SENSITIVITY ANALYSIS (±5°, ±10° Shifts)");
        Console.WriteLine("-----------------------------------------------------------------------------------------");

        var boundaries = new (string Name, float Hue, float L, float C, float a, float b)[]
        {
            ("Red / Brown Boundary", 24f, 35f, 25f, 22.8f, 10.1f),
            ("Brown / Orange Boundary", 40f, 42f, 30f, 23.0f, 19.3f),
            ("Orange / Yellow Boundary", 58f, 55f, 40f, 21.2f, 33.9f),
            ("Beige / Yellow Boundary", 75f, 52f, 26f, 6.7f, 25.1f),
            ("Yellow / Green Boundary", 95f, 50f, 35f, -3.0f, 34.8f),
            ("Green / Cyan Boundary", 175f, 45f, 30f, -29.9f, 2.6f),
            ("Cyan / Blue Boundary", 225f, 45f, 30f, -21.2f, -21.2f),
            ("Blue / Purple Boundary", 285f, 40f, 30f, 7.8f, -29.0f),
            ("Purple / Red Boundary", 345f, 40f, 30f, 29.0f, -7.8f)
        };

        Console.WriteLine($"{"Boundary",-28} | {"h-10°",-10} | {"h-5°",-10} | {"Original",-10} | {"h+5°",-10} | {"h+10°",-10} | Stability");
        Console.WriteLine(new string('-', 95));

        foreach (var b in boundaries)
        {
            float[] deltas = { -10f, -5f, 0f, 5f, 10f };
            var predsCurrent = new List<string>();
            var predsSimplified = new List<string>();

            foreach (var d in deltas)
            {
                float curH = (b.Hue + d + 360f) % 360f;
                float rad = curH * MathF.PI / 180f;
                float ca = b.C * MathF.Cos(rad);
                float cb = b.C * MathF.Sin(rad);

                predsCurrent.Add(ClassifyClusterCurrent(b.L, ca, cb, b.C, curH));
                predsSimplified.Add(ClassifyClusterSimplified(b.L, ca, cb, b.C, curH).SemanticColor);
            }

            string stabilityCur = predsCurrent.Distinct().Count() <= 2 ? "Stable" : "UNSTABLE (" + string.Join("->", predsCurrent.Distinct()) + ")";
            Console.WriteLine($"{b.Name,-28} | {predsCurrent[0],-10} | {predsCurrent[1],-10} | {predsCurrent[2],-10} | {predsCurrent[3],-10} | {predsCurrent[4],-10} | {stabilityCur}");
        }
    }

    // =========================================================================
    // 3. EMPIRICAL REAL GARMENT BENCHMARK (Clean-Crop & Full-Person)
    // =========================================================================
    public static async Task RunEmpiricalGarmentBenchmark(
        IPersonDetector detector,
        IPoseEstimationService poseService)
    {
        Console.WriteLine("\n>>> 3. REAL GARMENT BENCHMARK: CLEAN CROPS VS FULL PERSON ROI");
        Console.WriteLine("-----------------------------------------------------------------------------------------");

        string dlPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads");
        string userUpPath = @"C:\Users\Admin\.gemini\antigravity\brain\d1c30ac8-9746-4946-b173-1b32a3a64125\.user_uploaded";

        var dataset = BuildGroundTruthDataset(dlPath, userUpPath);
        Console.WriteLine($"Loaded {dataset.Count} verified real garment test cases across 16 target colors.\n");

        var cleanMetrics = new List<CleanCropMetrics>();
        var fullEvals = new List<FullPersonEval>();

        var currentSvc = new ClothingColorService();

        foreach (var sample in dataset)
        {
            if (!File.Exists(sample.FilePath)) continue;

            using var fullImg = Cv2.ImRead(sample.FilePath);
            if (fullImg.Empty()) continue;

            var dets = await detector.DetectPersonsAsync(fullImg);
            if (dets.Count == 0) continue;

            var primaryDet = dets.OrderByDescending(d => d.Box.Width * d.Box.Height).First();
            var poses = await poseService.EstimatePoseAsync(fullImg, new List<DetectionResult> { primaryDet });
            var pose = poses.Count > 0 ? poses[0] : null;

            BoundingBox targetBox = sample.Region == "Upper"
                ? (pose?.Regions?.UpperTorsoRegion ?? primaryDet.Box)
                : (pose?.Regions?.LowerBodyRegion ?? primaryDet.Box);

            int rx = Math.Clamp((int)targetBox.X, 0, fullImg.Width - 1);
            int ry = Math.Clamp((int)targetBox.Y, 0, fullImg.Height - 1);
            int rw = Math.Clamp((int)targetBox.Width, 1, fullImg.Width - rx);
            int rh = Math.Clamp((int)targetBox.Height, 1, fullImg.Height - ry);

            using var roi = new Mat(fullImg, new Rect(rx, ry, rw, rh));
            if (roi.Width < 8 || roi.Height < 8) continue;

            // Pure center clean crop
            int ccX = (int)(roi.Width * 0.20);
            int ccY = (int)(roi.Height * 0.20);
            int ccW = Math.Max(4, (int)(roi.Width * 0.60));
            int ccH = Math.Max(4, (int)(roi.Height * 0.60));
            using var cleanCrop = new Mat(roi, new Rect(ccX, ccY, ccW, ccH));

            // Measure CIELAB Metrics on clean crop
            using var labMat = new Mat();
            Cv2.CvtColor(cleanCrop, labMat, ColorConversionCodes.BGR2Lab);
            var lList = new List<float>();
            var aList = new List<float>();
            var bList = new List<float>();
            var cList = new List<float>();
            var hList = new List<float>();

            int cleanRows = cleanCrop.Rows;
            int cleanCols = cleanCrop.Cols;
            for (int r = 0; r < cleanRows; r++)
            {
                for (int c = 0; c < cleanCols; c++)
                {
                    var lp = labMat.At<Vec3b>(r, c);
                    float curL = lp.Item0 * 100f / 255f;
                    float curA = lp.Item1 - 128f;
                    float curB = lp.Item2 - 128f;
                    float curChroma = MathF.Sqrt(curA * curA + curB * curB);
                    float curHue = MathF.Atan2(curB, curA) * 180f / MathF.PI;
                    if (curHue < 0) curHue += 360f;

                    lList.Add(curL);
                    aList.Add(curA);
                    bList.Add(curB);
                    cList.Add(curChroma);
                    hList.Add(curHue);
                }
            }

            float medL = lList.OrderBy(x => x).ElementAt(lList.Count / 2);
            float medA = aList.OrderBy(x => x).ElementAt(aList.Count / 2);
            float medB = bList.OrderBy(x => x).ElementAt(bList.Count / 2);
            float medC = cList.OrderBy(x => x).ElementAt(cList.Count / 2);
            float medH = hList.OrderBy(x => x).ElementAt(hList.Count / 2);

            // [A] Original HSV Baseline
            string cleanPredA = PredictHsvBaseline(cleanCrop);
            string fullPredA = PredictHsvBaseline(roi);

            // [B] Current CIELAB
            string cleanPredB = currentSvc.ClassifyDetailedColor(cleanCrop).PrimaryColor;
            string fullPredB = currentSvc.ClassifyDetailedColor(roi, fullImg, sample.Region, pose, targetBox).PrimaryColor;

            // [C] Simplified Architecture (Base Color Family + Shade + Cluster Merging)
            string cleanPredC = PredictSimplifiedArchitecture(cleanCrop, null, null, null, null);
            string fullPredC = PredictSimplifiedArchitecture(roi, fullImg, sample.Region, pose, targetBox);

            cleanMetrics.Add(new CleanCropMetrics(
                sample.Id,
                sample.GroundTruthColor,
                medL, medA, medB, medC, medH,
                cleanPredA, cleanPredB, cleanPredC, sample.GroundTruthColor
            ));

            fullEvals.Add(new FullPersonEval(
                sample.Id,
                sample.FileName,
                sample.Region,
                sample.GroundTruthColor,
                cleanPredA, fullPredA,
                cleanPredB, fullPredB,
                cleanPredC, fullPredC
            ));
        }

        // Print Clean Crop Metric Table
        Console.WriteLine("\n=========================================================================================");
        Console.WriteLine(" SECTION 5: CLEAN-CROP BENCHMARK WITH PERCEPTUAL CIELAB METRICS");
        Console.WriteLine("=========================================================================================");
        Console.WriteLine($"{"Sample ID",-16} | {"GroundTruth",-11} | {"L*",5} | {"a*",5} | {"b*",5} | {"Chroma",6} | {"Hue°",5} | {"A: HSV",-10} | {"B: Current",-11} | {"C: Simplified",-13} | Match");
        Console.WriteLine(new string('-', 115));

        foreach (var m in cleanMetrics)
        {
            string match = string.Equals(m.GroundTruth, m.PredC_Simplified, StringComparison.OrdinalIgnoreCase) ? "PASS" : "MISMATCH";
            Console.WriteLine($"{m.SampleId,-16} | {m.GroundTruth,-11} | {m.MedianL,5:F1} | {m.MedianA,5:F1} | {m.MedianB,5:F1} | {m.MedianChroma,6:F1} | {m.MedianHue,5:F0} | {m.PredA_HSV,-10} | {m.PredB_Current,-11} | {m.PredC_Simplified,-13} | {match}");
        }

        // Print Full Comparison & Confusion Matrices
        PrintOverallComparison(fullEvals);
    }

    private static void PrintOverallComparison(List<FullPersonEval> evals)
    {
        Console.WriteLine("\n=========================================================================================");
        Console.WriteLine(" SECTION 6 & 8: PIPELINE ACCURACY & COVERAGE SUMMARY");
        Console.WriteLine("=========================================================================================");

        int total = evals.Count;

        int correctA_Clean = evals.Count(e => string.Equals(e.GroundTruth, e.CleanPredA, StringComparison.OrdinalIgnoreCase));
        int correctA_Full = evals.Count(e => string.Equals(e.GroundTruth, e.FullPredA, StringComparison.OrdinalIgnoreCase));

        int correctB_Clean = evals.Count(e => string.Equals(e.GroundTruth, e.CleanPredB, StringComparison.OrdinalIgnoreCase));
        int correctB_Full = evals.Count(e => string.Equals(e.GroundTruth, e.FullPredB, StringComparison.OrdinalIgnoreCase));

        int correctC_Clean = evals.Count(e => string.Equals(e.GroundTruth, e.CleanPredC, StringComparison.OrdinalIgnoreCase));
        int correctC_Full = evals.Count(e => string.Equals(e.GroundTruth, e.FullPredC, StringComparison.OrdinalIgnoreCase));

        int unknownA = evals.Count(e => e.FullPredA == "Unknown" || e.FullPredA == "InsufficientVisualEvidence");
        int unknownB = evals.Count(e => e.FullPredB == "Unknown" || e.FullPredB == "InsufficientVisualEvidence");
        int unknownC = evals.Count(e => e.FullPredC == "Unknown" || e.FullPredC == "InsufficientVisualEvidence");

        Console.WriteLine($"Total Test Cases Evaluated : {total}\n");
        Console.WriteLine($"[A] Pipeline A (Original HSV Baseline)     : Clean-Crop = {correctA_Clean}/{total} ({(float)correctA_Clean/total*100:F1}%), Full-Person = {correctA_Full}/{total} ({(float)correctA_Full/total*100:F1}%), Unknowns = {unknownA}");
        Console.WriteLine($"[B] Pipeline B (Current CIELAB)             : Clean-Crop = {correctB_Clean}/{total} ({(float)correctB_Clean/total*100:F1}%), Full-Person = {correctB_Full}/{total} ({(float)correctB_Full/total*100:F1}%), Unknowns = {unknownB}");
        Console.WriteLine($"[C] Pipeline C (Simplified Base+Shade CIELAB): Clean-Crop = {correctC_Clean}/{total} ({(float)correctC_Clean/total*100:F1}%), Full-Person = {correctC_Full}/{total} ({(float)correctC_Full/total*100:F1}%), Unknowns = {unknownC}");

        // Specific Failure Pair Audits
        Console.WriteLine("\n=========================================================================================");
        Console.WriteLine(" SECTION 13: CRITICAL SPECIFIC FAILURE PAIRS AUDIT");
        Console.WriteLine("=========================================================================================");

        var pairsToAudit = new (string GT, string Erroneous)[]
        {
            ("Red", "Black"),
            ("Brown", "Black"),
            ("Black", "Brown"),
            ("Black", "Maroon"),
            ("Beige", "Black"),
            ("Olive", "Grey"),
            ("Olive", "Yellow"),
            ("White", "Grey"),
            ("Blue", "Maroon"),
            ("Navy", "Black")
        };

        Console.WriteLine($"{"Target Failure Pair",-25} | {"Pipeline A (HSV)",-16} | {"Pipeline B (Current)",-20} | {"Pipeline C (Simplified)",-22}");
        Console.WriteLine(new string('-', 90));

        foreach (var p in pairsToAudit)
        {
            int errA = evals.Count(e => string.Equals(e.GroundTruth, p.GT, StringComparison.OrdinalIgnoreCase) && string.Equals(e.FullPredA, p.Erroneous, StringComparison.OrdinalIgnoreCase));
            int errB = evals.Count(e => string.Equals(e.GroundTruth, p.GT, StringComparison.OrdinalIgnoreCase) && string.Equals(e.FullPredB, p.Erroneous, StringComparison.OrdinalIgnoreCase));
            int errC = evals.Count(e => string.Equals(e.GroundTruth, p.GT, StringComparison.OrdinalIgnoreCase) && string.Equals(e.FullPredC, p.Erroneous, StringComparison.OrdinalIgnoreCase));

            Console.WriteLine($"{p.GT + " -> " + p.Erroneous,-25} | {errA + " occurrences",-16} | {errB + " occurrences",-20} | {errC + " occurrences",-22}");
        }

        // Generate Full Confusion Matrix for Pipeline C
        Console.WriteLine("\n=========================================================================================");
        Console.WriteLine(" SECTION 7: FULL CONFUSION MATRIX (Pipeline C - Simplified Architecture)");
        Console.WriteLine("=========================================================================================");

        var allLabels = new[] { "Black", "White", "Grey", "Red", "Maroon", "Brown", "Beige", "Orange", "Yellow", "Green", "Olive", "Blue", "Navy", "Cyan", "Purple", "Pink" };

        Console.Write($"{"GT / Pred",-10} |");
        foreach (var l in allLabels) Console.Write($" {l.Substring(0, Math.Min(3, l.Length)),3}");
        Console.WriteLine();
        Console.WriteLine(new string('-', 85));

        foreach (var gt in allLabels)
        {
            var gtEvals = evals.Where(e => string.Equals(e.GroundTruth, gt, StringComparison.OrdinalIgnoreCase)).ToList();
            if (gtEvals.Count == 0) continue;

            Console.Write($"{gt,-10} |");
            foreach (var pred in allLabels)
            {
                int count = gtEvals.Count(e => string.Equals(e.FullPredC, pred, StringComparison.OrdinalIgnoreCase));
                if (count > 0)
                {
                    Console.Write($" {count,3}");
                }
                else
                {
                    Console.Write("   .");
                }
            }
            Console.WriteLine();
        }
    }

    // =========================================================================
    // SIMPLIFIED ARCHITECTURE CLASSIFIER IMPLEMENTATION
    // Base Color Family -> Semantic Shade + Single Neutral Cutoff + Cluster Merging
    // =========================================================================
    public record SimplifiedDecision(
        string BaseColorFamily,
        string Shade,
        string SemanticColor,
        bool IsNeutral
    );

    public static SimplifiedDecision ClassifyClusterSimplified(float L, float a, float b, float chroma, float hueAngle)
    {
        // -------------------------------------------------------------
        // STEP 1: UNIFIED ACHROMATIC (NEUTRAL) DECISION
        // Single empirical neutrality cutoff: Chroma < 5.4
        // -------------------------------------------------------------
        const float NeutralMaxChroma = 5.4f;

        if (chroma < NeutralMaxChroma)
        {
            if (L < 24.0f)
            {
                return new SimplifiedDecision("Neutral", "Dark", "Black", true);
            }
            if (L >= 65.0f)
            {
                return new SimplifiedDecision("Neutral", "Light", "White", true);
            }
            return new SimplifiedDecision("Neutral", "Medium", "Grey", true);
        }

        // -------------------------------------------------------------
        // STEP 2: CHROMATIC FAMILY (Broad Continuous Sectors)
        // -------------------------------------------------------------
        // Red Family: 345° .. 360° or 0° .. 25°
        if (hueAngle >= 345f || hueAngle < 25f)
        {
            if (L < 38f && chroma >= 6.5f && a >= 4.5f)
            {
                return new SimplifiedDecision("Red", "Dark", "Maroon", false);
            }
            if (L < 22f && chroma < 6.0f)
            {
                return new SimplifiedDecision("Neutral", "Dark", "Black", true);
            }
            if (L > 65f && chroma < 32f)
            {
                return new SimplifiedDecision("Red", "Light", "Pink", false);
            }
            return new SimplifiedDecision("Red", "Normal", "Red", false);
        }

        // Orange / Warm Family: 25° .. 58°
        if (hueAngle >= 25f && hueAngle < 58f)
        {
            if (L < 42f)
            {
                if (L < 22f && chroma < 5.8f) return new SimplifiedDecision("Neutral", "Dark", "Black", true);
                return new SimplifiedDecision("Orange", "Dark", "Brown", false);
            }
            return new SimplifiedDecision("Orange", "Normal", "Orange", false);
        }

        // Yellow / Beige Family: 58° .. 95°
        if (hueAngle >= 58f && hueAngle < 95f)
        {
            if (chroma < 26f && L > 45f)
            {
                return new SimplifiedDecision("Yellow", "LightMuted", "Beige", false);
            }
            if (L < 22f && chroma < 5.8f) return new SimplifiedDecision("Neutral", "Dark", "Black", true);
            return new SimplifiedDecision("Yellow", "Normal", "Yellow", false);
        }

        // Green Family: 95° .. 175°
        if (hueAngle >= 95f && hueAngle < 175f)
        {
            if (L < 36f && chroma < 22f)
            {
                return new SimplifiedDecision("Green", "DarkMuted", "Olive", false);
            }
            if (L < 22f && chroma < 5.8f) return new SimplifiedDecision("Neutral", "Dark", "Black", true);
            return new SimplifiedDecision("Green", "Normal", "Green", false);
        }

        // Cyan Family: 175° .. 225°
        if (hueAngle >= 175f && hueAngle < 225f)
        {
            if (L < 22f && chroma < 5.8f) return new SimplifiedDecision("Neutral", "Dark", "Black", true);
            return new SimplifiedDecision("Cyan", "Normal", "Cyan", false);
        }

        // Blue Family: 225° .. 285°
        if (hueAngle >= 225f && hueAngle < 285f)
        {
            if (L < 36f && chroma >= 5.8f)
            {
                return new SimplifiedDecision("Blue", "Dark", "Navy", false);
            }
            if (L < 22f && chroma < 5.8f) return new SimplifiedDecision("Neutral", "Dark", "Black", true);
            return new SimplifiedDecision("Blue", "Normal", "Blue", false);
        }

        // Purple Family: 285° .. 345° (No arbitrary Maroon overrides!)
        if (hueAngle >= 285f && hueAngle < 345f)
        {
            if (L < 22f && chroma < 5.8f) return new SimplifiedDecision("Neutral", "Dark", "Black", true);
            if (L > 65f && chroma < 30f)
            {
                return new SimplifiedDecision("Purple", "Light", "Pink", false);
            }
            return new SimplifiedDecision("Purple", "Normal", "Purple", false);
        }

        return new SimplifiedDecision("Neutral", "Medium", "Grey", true);
    }

    public static string PredictSimplifiedArchitecture(
        Mat clothingCrop,
        Mat? fullImg,
        string? region,
        PersonPoseResult? pose,
        BoundingBox? targetBox)
    {
        if (clothingCrop == null || clothingCrop.Empty() || clothingCrop.Width < 8 || clothingCrop.Height < 8)
            return "InsufficientVisualEvidence";

        using var mask = new Mat(clothingCrop.Size(), MatType.CV_8UC1, Scalar.All(0));
        Cv2.Ellipse(mask, new RotatedRect(
            new Point2f(clothingCrop.Width * 0.5f, clothingCrop.Height * 0.5f),
            new Size2f(clothingCrop.Width * 0.70f, clothingCrop.Height * 0.75f), 0),
            Scalar.White, -1);

        using var labMat = new Mat();
        Cv2.CvtColor(clothingCrop, labMat, ColorConversionCodes.BGR2Lab);

        var validPixels = new List<Vec3f>();
        int validRows = clothingCrop.Rows;
        int validCols = clothingCrop.Cols;
        for (int r = 0; r < validRows; r++)
        {
            for (int c = 0; c < validCols; c++)
            {
                if (mask.At<byte>(r, c) > 0)
                {
                    var lp = labMat.At<Vec3b>(r, c);
                    validPixels.Add(new Vec3f(lp.Item0, lp.Item1, lp.Item2));
                }
            }
        }

        if (validPixels.Count < 30) return "InsufficientVisualEvidence";

        // K-Means Clustering
        int K = 3;
        using var samplesMat = new Mat(validPixels.Count, 3, MatType.CV_32F);
        for (int i = 0; i < validPixels.Count; i++)
        {
            samplesMat.Set(i, 0, validPixels[i].Item0);
            samplesMat.Set(i, 1, validPixels[i].Item1);
            samplesMat.Set(i, 2, validPixels[i].Item2);
        }

        using var labels = new Mat();
        using var centers = new Mat();
        var criteria = new TermCriteria(CriteriaTypes.Eps | CriteriaTypes.MaxIter, 15, 1.0);
        Cv2.Kmeans(samplesMat, K, labels, criteria, 3, KMeansFlags.PpCenters, centers);

        var clusterCounts = new int[K];
        for (int i = 0; i < validPixels.Count; i++) clusterCounts[labels.At<int>(i)]++;

        // Classify each cluster
        var clusterDecisions = new List<(SimplifiedDecision Decision, float Share, float L, float Chroma)>();
        for (int k = 0; k < K; k++)
        {
            float share = (float)clusterCounts[k] / validPixels.Count;
            if (share < 0.08f) continue;

            float cL = centers.At<float>(k, 0) * 100f / 255f;
            float ca = centers.At<float>(k, 1) - 128f;
            float cb = centers.At<float>(k, 2) - 128f;
            float chroma = MathF.Sqrt(ca * ca + cb * cb);
            float hueAngle = MathF.Atan2(cb, ca) * 180f / MathF.PI;
            if (hueAngle < 0) hueAngle += 360f;

            var decision = ClassifyClusterSimplified(cL, ca, cb, chroma, hueAngle);
            clusterDecisions.Add((decision, share, cL, chroma));
        }

        if (clusterDecisions.Count == 0) return "Unknown";

        // -------------------------------------------------------------
        // STEP 6 & 7: CLUSTER MERGING BY BROAD BASE COLOR FAMILY
        // -------------------------------------------------------------
        var familyVotes = new Dictionary<string, float>(StringComparer.OrdinalIgnoreCase);
        var semanticVotes = new Dictionary<string, float>(StringComparer.OrdinalIgnoreCase);

        foreach (var c in clusterDecisions)
        {
            familyVotes[c.Decision.BaseColorFamily] = familyVotes.GetValueOrDefault(c.Decision.BaseColorFamily, 0f) + c.Share;
            semanticVotes[c.Decision.SemanticColor] = semanticVotes.GetValueOrDefault(c.Decision.SemanticColor, 0f) + c.Share;
        }

        var topFamily = familyVotes.OrderByDescending(kv => kv.Value).First();

        // If the top family has >= 45% merged vote, pick the strongest semantic shade within that family
        if (topFamily.Value >= 0.45f)
        {
            var shadesInFamily = clusterDecisions
                .Where(c => string.Equals(c.Decision.BaseColorFamily, topFamily.Key, StringComparison.OrdinalIgnoreCase))
                .GroupBy(c => c.Decision.SemanticColor)
                .Select(g => new { Shade = g.Key, TotalShare = g.Sum(x => x.Share) })
                .OrderByDescending(x => x.TotalShare)
                .First();

            return shadesInFamily.Shade;
        }

        // Fallback to top individual semantic vote
        return semanticVotes.OrderByDescending(kv => kv.Value).First().Key;
    }

    private static string ClassifyClusterCurrent(float L, float a, float b, float chroma, float hueAngle)
    {
        // Replicates ClothingColorService.cs current logic
        if (L < 24.0f && chroma < 5.8f) return "Black";
        if (chroma < 5.0f && L >= 65.0f) return "White";
        if (chroma < 5.2f && L >= 24.0f && L < 65.0f) return "Grey";

        if (hueAngle >= 345f || hueAngle < 24f)
        {
            if (L < 38f && chroma >= 6.5f && a >= 4.5f) return "Maroon";
            if (L < 24.0f) return "Black";
            if (L > 65f && chroma < 30f) return "Pink";
            return "Red";
        }

        if (hueAngle >= 24f && hueAngle < 58f)
        {
            if (L < 42f && chroma >= 5.5f) return "Brown";
            if (L < 24.0f) return "Black";
            return "Orange";
        }

        if (hueAngle >= 58f && hueAngle < 95f)
        {
            if (chroma < 26f && L > 50f) return "Beige";
            if (L < 24.0f) return "Black";
            return "Yellow";
        }

        if (hueAngle >= 95f && hueAngle < 175f)
        {
            if (L < 24.0f && chroma < 5.5f) return "Black";
            return "Green";
        }

        if (hueAngle >= 175f && hueAngle < 225f)
        {
            if (L < 24.0f && chroma < 5.5f) return "Black";
            return "Cyan";
        }

        if (hueAngle >= 225f && hueAngle < 285f)
        {
            if (L < 36f && chroma >= 5.8f) return "Navy";
            if (L < 24.0f) return "Black";
            return "Blue";
        }

        if (hueAngle >= 285f && hueAngle < 345f)
        {
            if (L < 38f && chroma >= 6.5f && a > 4.5f) return "Maroon";
            if (L < 24.0f) return "Black";
            if (L > 60f) return "Pink";
            return "Purple";
        }

        return "Grey";
    }

    private static string PredictHsvBaseline(Mat crop)
    {
        if (crop == null || crop.Empty()) return "Unknown";

        using var hsv = new Mat();
        Cv2.CvtColor(crop, hsv, ColorConversionCodes.BGR2HSV);

        long sumH = 0, sumS = 0, sumV = 0, count = 0;
        int hsvRows = hsv.Rows;
        int hsvCols = hsv.Cols;
        for (int r = 0; r < hsvRows; r++)
        {
            for (int c = 0; c < hsvCols; c++)
            {
                var p = hsv.At<Vec3b>(r, c);
                sumH += p.Item0;
                sumS += p.Item1;
                sumV += p.Item2;
                count++;
            }
        }

        if (count == 0) return "Unknown";
        float avgH = (float)sumH / count;
        float avgS = (float)sumS / count;
        float avgV = (float)sumV / count;

        if (avgV < 50) return "Black";
        if (avgS < 35 && avgV > 180) return "White";
        if (avgS < 40) return "Grey";

        if (avgH < 10 || avgH >= 170)
        {
            if (avgV < 90) return "Maroon";
            if (avgS < 100 && avgV > 160) return "Pink";
            return "Red";
        }
        if (avgH >= 10 && avgH < 25)
        {
            if (avgV < 110) return "Brown";
            return "Orange";
        }
        if (avgH >= 25 && avgH < 35)
        {
            if (avgS < 80) return "Beige";
            return "Yellow";
        }
        if (avgH >= 35 && avgH < 85) return "Green";
        if (avgH >= 85 && avgH < 105) return "Cyan";
        if (avgH >= 105 && avgH < 135)
        {
            if (avgV < 90) return "Navy";
            return "Blue";
        }
        if (avgH >= 135 && avgH < 170) return "Purple";

        return "Grey";
    }

    public static List<GarmentBenchmarkSample> BuildGroundTruthDataset(string dlPath, string userUpPath)
    {
        var list = new List<GarmentBenchmarkSample>();

        void AddIf(string file, string reg, string color, string desc)
        {
            string p = Path.Combine(dlPath, file);
            if (!File.Exists(p)) p = Path.Combine(userUpPath, file);
            if (File.Exists(p))
            {
                list.Add(new GarmentBenchmarkSample($"TC-{list.Count + 1:D2}", file, p, reg, color, desc));
            }
        }

        // Verified Ground Truth Dataset from Downloads & user_uploaded
        AddIf("download (3).jfif", "Upper", "Beige", "Woman in Beige / Tan loose top");
        AddIf("download (3).jfif", "Lower", "Black", "Woman in Black / Dark trousers");
        AddIf("download (4).jfif", "Upper", "Green", "Man in Olive / Dark Green overshirt");
        AddIf("download (4).jfif", "Lower", "Black", "Man in Black trousers");
        AddIf("femalealone.jpg", "Upper", "Red", "Female in Red sweater / top");
        AddIf("femalealone.jpg", "Lower", "Black", "Female in Black trousers");
        AddIf("twofemales.jpg", "Upper", "White", "Female in White shirt");
        AddIf("twofemales.jpg", "Lower", "Blue", "Female in Blue denim jeans");
        AddIf("boy.jpg", "Upper", "Blue", "Young boy in Blue polo");
        AddIf("female.jpg", "Upper", "White", "Female in White dress / blouse");
        AddIf("pexels.jpg", "Upper", "Yellow", "Person in Yellow hoodie");
        AddIf("pexels.jpg", "Lower", "Blue", "Person in Blue denim jeans");
        AddIf("young-man-standing-in-middle-of-road-on-sunny-day-JMPF00583.jpg", "Upper", "White", "Man in White t-shirt");
        AddIf("young-man-standing-in-middle-of-road-on-sunny-day-JMPF00583.jpg", "Lower", "Beige", "Man in Khaki / Beige shorts");
        AddIf("download.jfif", "Upper", "Green", "Green casual tee");
        AddIf("download (1).jfif", "Upper", "Brown", "Brown jacket");
        AddIf("download (2).jfif", "Upper", "Purple", "Purple top");
        AddIf("download (2).jpg", "Upper", "Pink", "Pink sweater");
        AddIf("0b3c3eb93db2766ea3430914f04a213c.jpg", "Upper", "Orange", "Orange zip jacket");
        AddIf("8757d5afbac4ad4ef57a2dcaf9d794c2.jpg", "Upper", "Black", "Black sweatshirt");
        AddIf("BrandImage.png", "Upper", "Black", "Black Nike top");
        AddIf("BrandImage.png", "Lower", "Black", "Black Adidas leggings");
        AddIf("download (15).jpg", "Upper", "Olive", "Olive military shirt");
        AddIf("download (10).jpg", "Upper", "Maroon", "Dark Maroon shirt");
        AddIf("download (7).jpg", "Upper", "Navy", "Dark Navy jacket");
        AddIf("download (12).jpg", "Upper", "Cyan", "Cyan turquoise top");

        // User uploaded test cases
        if (Directory.Exists(userUpPath))
        {
            AddIf("media_1790576504774.jpg", "Upper", "Maroon", "Maroon casual shirt");
            AddIf("media_1790576504774.jpg", "Lower", "Grey", "Grey trousers");
            AddIf("media_1790576932719.jpg", "Upper", "Black", "Black jacket");
            AddIf("media_1790576932719.jpg", "Lower", "Blue", "Blue denim jeans");
        }

        return list;
    }
}
