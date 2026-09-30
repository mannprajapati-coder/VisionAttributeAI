using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using VisionAttributeAI.Models.AI;
using VisionAttributeAI.Models.Pose;
using VisionAttributeAI.Services.Attributes;
using VisionAttributeAI.Services.Detection;
using VisionAttributeAI.Services.Pose;

namespace VisionAttributeAI.Tests;

/// <summary>
/// Empirical Color Quality Gating & Best-Frame Selection Benchmark Engine.
/// Measures color-independent information loss signals (clipping, dynamic range, blur, resolution),
/// separates good vs degraded observations, and evaluates Multi-Frame Top-K Consensus on video sequences.
/// Zero production modifications.
/// </summary>
public static class ColorQualityGatingBenchmarkRunner
{
    public record ColorQualityMetrics(
        float NearWhiteClipPct,
        float NearBlackClipPct,
        float P05_L,
        float P25_L,
        float P50_L,
        float P75_L,
        float P95_L,
        float DynamicRange_L,
        float StdDev_L,
        float P50_Chroma,
        float P90_Chroma,
        double SharpnessVariance,
        int EffectiveWidth,
        int EffectiveHeight,
        int TotalArea,
        int ValidPixelCount,
        float ValidPixelRatio,
        float QualityScore,
        bool IsAccepted,
        string RejectionReason
    );

    public record EvaluatedSampleMetric(
        string SampleId,
        string SetType,
        string FileName,
        string Region,
        string GroundTruth,
        string RawPredictionCIELAB,
        string RawPredictionCLIP,
        bool RawCorrectCIELAB,
        bool RawCorrectCLIP,
        ColorQualityMetrics Quality
    );

    public static async Task RunQualityBenchmarkAsync(
        IPersonDetector detector,
        IPoseEstimationService poseService)
    {
        Console.WriteLine("\n=========================================================================================");
        Console.WriteLine(" EMPIRICAL COLOR INFORMATION LOSS & BEST-FRAME QUALITY GATING BENCHMARK");
        Console.WriteLine("=========================================================================================\n");

        string dlPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads");
        string userUpPath = @"C:\Users\Admin\.gemini\antigravity\brain\d1c30ac8-9746-4946-b173-1b32a3a64125\.user_uploaded";

        using var visionSession = new InferenceSession("models/par/fashion_clip_vision.onnx");
        using var textSession = new InferenceSession("models/par/fashion_clip_text.onnx");
        var upperBank = FashionClipColorBenchmarkRunner.PrecomputeUpperRegionColorEmbeddings(textSession);
        var lowerBank = FashionClipColorBenchmarkRunner.PrecomputeLowerRegionColorEmbeddings(textSession);

        // 1. Process 32 Verified Single-Image Samples and Extract Color-Independent Quality Signals
        var dataset = RigorousColorBenchmarkRunner.BuildComprehensiveDataset(dlPath, userUpPath);
        var evaluatedSamples = new List<EvaluatedSampleMetric>();

        foreach (var sample in dataset)
        {
            if (!File.Exists(sample.ImagePath)) continue;
            using var fullImg = Cv2.ImRead(sample.ImagePath);
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
            if (roi.Width < 6 || roi.Height < 6) continue;

            // Compute Color-Independent Quality Signals
            var q = ComputeColorQualitySignals(roi, sample.Region, pose, targetBox);

            // Raw Predictions (Simplified CIELAB and Fashion-CLIP)
            var (predCIELAB, _, _, _, _, _) = RigorousColorBenchmarkRunner.EvaluateSimplified(roi, fullImg, sample.Region, pose, targetBox);
            var imgEmb = FashionClipColorBenchmarkRunner.ExtractImageEmbedding(roi, visionSession);
            var clipRes = FashionClipColorBenchmarkRunner.ClassifyEmbeddingTop5(imgEmb, sample.Region == "Upper" ? upperBank : lowerBank);

            bool okCIELAB = string.Equals(sample.GroundTruthColor, predCIELAB, StringComparison.OrdinalIgnoreCase);
            bool okCLIP = string.Equals(sample.GroundTruthColor, clipRes.Top1, StringComparison.OrdinalIgnoreCase);

            evaluatedSamples.Add(new EvaluatedSampleMetric(
                sample.SampleId,
                sample.SetType,
                sample.FileName,
                sample.Region,
                sample.GroundTruthColor,
                predCIELAB,
                clipRes.Top1,
                okCIELAB,
                okCLIP,
                q
            ));
        }

        // =========================================================================
        // SECTION 1 & 2: QUALITY METRICS FOR CORRECT VS INCORRECT OBSERVATIONS
        // =========================================================================
        Console.WriteLine("=========================================================================================");
        Console.WriteLine(" 1 & 2. QUALITY METRIC DISTRIBUTIONS: CORRECT VS INCORRECT OBSERVATIONS");
        Console.WriteLine("=========================================================================================");

        var correctGroup = evaluatedSamples.Where(s => s.RawCorrectCIELAB).Select(s => s.Quality).ToList();
        var incorrectGroup = evaluatedSamples.Where(s => !s.RawCorrectCIELAB).Select(s => s.Quality).ToList();

        PrintGroupQualityStats("CORRECT OBSERVATIONS", correctGroup);
        PrintGroupQualityStats("INCORRECT OBSERVATIONS", incorrectGroup);

        // =========================================================================
        // SECTION 3: EXTREME-CASE ANALYSIS (B-05, B-10, B-32, B-36)
        // =========================================================================
        Console.WriteLine("\n=========================================================================================");
        Console.WriteLine(" 3 & 4. AUDIT OF THE FOUR EXTREME CASES (Color-Independent Information Loss)");
        Console.WriteLine("=========================================================================================");

        var extremeIds = new[] { "B-05", "B-10", "B-32", "B-36" };
        Console.WriteLine($"{"ID",-5} | {"GT",-7} | {"CIELAB",-8} | {"NearWhite%",10} | {"NearBlack%",10} | {"P05_L",6} | {"P50_L",6} | {"P95_L",6} | {"DynRange",8} | {"Sharpness",9} | {"QualityScore",12} | {"Gate Decision",-13} | {"Reason"}");
        Console.WriteLine(new string('-', 140));

        foreach (var id in extremeIds)
        {
            var s = evaluatedSamples.FirstOrDefault(e => e.SampleId == id);
            if (s == null) continue;
            var q = s.Quality;
            string decision = q.IsAccepted ? "ACCEPTED" : "REJECTED";
            Console.WriteLine($"{s.SampleId,-5} | {s.GroundTruth,-7} | {s.RawPredictionCIELAB,-8} | {q.NearWhiteClipPct * 100,9:F1}% | {q.NearBlackClipPct * 100,9:F1}% | {q.P05_L,6:F1} | {q.P50_L,6:F1} | {q.P95_L,6:F1} | {q.DynamicRange_L,8:F1} | {q.SharpnessVariance,9:F1} | {q.QualityScore,12:F2} | {decision,-13} | {q.RejectionReason}");
        }

        // =========================================================================
        // SECTION 4: GOOD OBSERVATION AUDIT (Ensuring True Black/White/Navy Not Rejected)
        // =========================================================================
        Console.WriteLine("\n=========================================================================================");
        Console.WriteLine(" 5. AUDIT OF VALID OBSERVATIONS (Black, White, Navy, Brown, Red, Green)");
        Console.WriteLine("=========================================================================================");

        var goodIds = new[] { "D-01", "D-02", "D-03", "D-04", "D-05", "D-06", "D-15", "D-16", "D-23", "B-04", "B-07", "B-09", "B-33" };
        Console.WriteLine($"{"ID",-5} | {"GT",-7} | {"CIELAB",-8} | {"NearWhite%",10} | {"NearBlack%",10} | {"P05_L",6} | {"P50_L",6} | {"P95_L",6} | {"DynRange",8} | {"Sharpness",9} | {"QualityScore",12} | {"Gate Decision",-13} | {"Status"}");
        Console.WriteLine(new string('-', 135));

        foreach (var id in goodIds)
        {
            var s = evaluatedSamples.FirstOrDefault(e => e.SampleId == id);
            if (s == null) continue;
            var q = s.Quality;
            string decision = q.IsAccepted ? "ACCEPTED" : "REJECTED";
            string status = (s.RawCorrectCIELAB && q.IsAccepted) ? "PASS (Kept)" : (q.IsAccepted ? "Accepted Error" : "Accidentally Rejected");
            Console.WriteLine($"{s.SampleId,-5} | {s.GroundTruth,-7} | {s.RawPredictionCIELAB,-8} | {q.NearWhiteClipPct * 100,9:F1}% | {q.NearBlackClipPct * 100,9:F1}% | {q.P05_L,6:F1} | {q.P50_L,6:F1} | {q.P95_L,6:F1} | {q.DynamicRange_L,8:F1} | {q.SharpnessVariance,9:F1} | {q.QualityScore,12:F2} | {decision,-13} | {status}");
        }

        // =========================================================================
        // SECTION 5: SINGLE-IMAGE ACCURACY VS COVERAGE EVALUATION
        // =========================================================================
        Console.WriteLine("\n=========================================================================================");
        Console.WriteLine(" 6, 7, 8. SINGLE-IMAGE QUALITY-GATED EVALUATION (DEV VS FROZEN BLIND)");
        Console.WriteLine("=========================================================================================");

        EvaluateSingleImageGatePerformance("DEVELOPMENT SET", evaluatedSamples.Where(e => e.SetType == "DEV").ToList());
        EvaluateSingleImageGatePerformance("FROZEN BLIND SET", evaluatedSamples.Where(e => e.SetType == "BLIND").ToList());
        EvaluateSingleImageGatePerformance("COMBINED DATASET", evaluatedSamples);

        // =========================================================================
        // SECTION 6: VIDEO TEMPORAL MULTI-FRAME & BEST-FRAME POOL EXPERIMENT
        // =========================================================================
        Console.WriteLine("\n=========================================================================================");
        Console.WriteLine(" 10, 11, 12. VIDEO SEQUENCES MULTI-FRAME BEST-FRAME POOL EXPERIMENT");
        Console.WriteLine("=========================================================================================");

        await RunVideoSequenceAudit(detector, poseService, dlPath, "twopeople.mp4", "Two People Crossing Paths in Hallway", visionSession, upperBank, lowerBank);
        await RunVideoSequenceAudit(detector, poseService, dlPath, "AloneFemale.mp4", "Woman Walking with Red Top & Black Pants", visionSession, upperBank, lowerBank);
        await RunVideoSequenceAudit(detector, poseService, dlPath, "walkingTalking.mp4", "Two People Walking Outdoors", visionSession, upperBank, lowerBank);
    }

    private static ColorQualityMetrics ComputeColorQualitySignals(
        Mat crop,
        string region,
        PersonPoseResult? pose,
        BoundingBox? targetBox)
    {
        int totalPixels = crop.Rows * crop.Cols;
        using var mask = new Mat(crop.Size(), MatType.CV_8UC1, Scalar.All(0));

        if (pose != null && targetBox != null)
        {
            ApplyPosePolygon(mask, crop, region, pose, targetBox);
        }
        else
        {
            Cv2.Ellipse(mask, new RotatedRect(
                new Point2f(crop.Width * 0.5f, crop.Height * 0.5f),
                new Size2f(crop.Width * 0.70f, crop.Height * 0.75f), 0),
                Scalar.White, -1);
        }

        using var labMat = new Mat();
        Cv2.CvtColor(crop, labMat, ColorConversionCodes.BGR2Lab);

        var lList = new List<float>();
        var cList = new List<float>();

        int nearWhiteCount = 0;
        int nearBlackCount = 0;

        int cRows = crop.Rows;
        int cCols = crop.Cols;
        for (int r = 0; r < cRows; r++)
        {
            for (int c = 0; c < cCols; c++)
            {
                if (mask.At<byte>(r, c) > 0)
                {
                    var lp = labMat.At<Vec3b>(r, c);
                    var bp = crop.At<Vec3b>(r, c);

                    float l = lp.Item0 * 100f / 255f;
                    float a = lp.Item1 - 128f;
                    float b = lp.Item2 - 128f;
                    float chroma = MathF.Sqrt(a * a + b * b);

                    lList.Add(l);
                    cList.Add(chroma);

                    // Check highlight clipping (near sensor max)
                    if (l >= 94.0f || (bp.Item0 >= 248 && bp.Item1 >= 248 && bp.Item2 >= 248)) nearWhiteCount++;

                    // Check shadow clipping (near sensor zero)
                    if (l <= 6.0f || (bp.Item0 <= 12 && bp.Item1 <= 12 && bp.Item2 <= 12)) nearBlackCount++;
                }
            }
        }

        int validCount = lList.Count;
        float validRatio = (float)validCount / Math.Max(1, totalPixels);

        if (validCount < 20)
        {
            return new ColorQualityMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, crop.Width, crop.Height, totalPixels, validCount, validRatio, 0.0f, false, "InsufficientValidPixels (< 20)");
        }

        lList.Sort();
        cList.Sort();

        float p05_L = lList[(int)(validCount * 0.05)];
        float p25_L = lList[(int)(validCount * 0.25)];
        float p50_L = lList[(int)(validCount * 0.50)];
        float p75_L = lList[(int)(validCount * 0.75)];
        float p95_L = lList[(int)(validCount * 0.95)];

        float dynRange_L = p95_L - p05_L;

        double meanL = lList.Average();
        double sumSqL = lList.Sum(x => (x - meanL) * (x - meanL));
        float stdDevL = (float)Math.Sqrt(sumSqL / validCount);

        float p50_Chroma = cList[(int)(validCount * 0.50)];
        float p90_Chroma = cList[(int)(validCount * 0.90)];

        float nearWhitePct = (float)nearWhiteCount / validCount;
        float nearBlackPct = (float)nearBlackCount / validCount;

        // Sharpness via Laplacian variance
        double sharpness = CalculateLaplacianVariance(crop);

        // Compute Color Quality Score Q in [0.0 .. 1.0]
        float sizeFactor = Math.Clamp((float)totalPixels / 15000f, 0.2f, 1.0f);
        float sharpFactor = Math.Clamp((float)(sharpness / 120.0), 0.2f, 1.0f);
        float dynFactor = Math.Clamp(dynRange_L / 35.0f, 0.2f, 1.0f);
        float ratioFactor = Math.Clamp(validRatio / 0.50f, 0.3f, 1.0f);

        // Penalties for extreme clipping
        float clipPenalty = 1.0f;
        if (nearWhitePct > 0.40f) clipPenalty *= Math.Max(0.2f, 1.0f - (nearWhitePct - 0.40f) * 2.0f);
        if (nearBlackPct > 0.65f) clipPenalty *= Math.Max(0.3f, 1.0f - (nearBlackPct - 0.65f) * 1.8f);

        float qualityScore = ((sizeFactor * 0.25f) + (sharpFactor * 0.30f) + (dynFactor * 0.25f) + (ratioFactor * 0.20f)) * clipPenalty;
        qualityScore = Math.Clamp(qualityScore, 0.05f, 1.0f);

        bool isAccepted = true;
        string rejectReason = string.Empty;

        // Color-Independent Rejection Rules:
        if (nearWhitePct >= 0.55f && p50_L >= 88.0f)
        {
            isAccepted = false;
            rejectReason = $"OverexposureHighlightClipping ({nearWhitePct * 100:F0}% near-white pixels, Median L*={p50_L:F0})";
        }
        else if (nearBlackPct >= 0.75f && p50_L <= 7.0f && dynRange_L < 8.0f)
        {
            isAccepted = false;
            rejectReason = $"UnderexposureShadowClipping ({nearBlackPct * 100:F0}% near-black pixels, DynRange={dynRange_L:F0})";
        }
        else if (sharpness < 18.0 && totalPixels < 3000)
        {
            isAccepted = false;
            rejectReason = $"SevereMotionBlurOrLowResolution (Sharpness={sharpness:F1}, Area={totalPixels})";
        }
        else if (validRatio < 0.20f)
        {
            isAccepted = false;
            rejectReason = $"InsufficientGarmentMaskCoverage (ValidRatio={validRatio * 100:F0}% < 20%)";
        }
        else if (qualityScore < 0.35f)
        {
            isAccepted = false;
            rejectReason = $"LowCompositeColorQuality (Score={qualityScore:F2} < 0.35)";
        }

        return new ColorQualityMetrics(
            nearWhitePct, nearBlackPct,
            p05_L, p25_L, p50_L, p75_L, p95_L,
            dynRange_L, stdDevL,
            p50_Chroma, p90_Chroma,
            sharpness,
            crop.Width, crop.Height, totalPixels,
            validCount, validRatio,
            qualityScore, isAccepted, rejectReason
        );
    }

    private static void PrintGroupQualityStats(string groupName, List<ColorQualityMetrics> metrics)
    {
        if (metrics.Count == 0) return;
        Console.WriteLine($"--- {groupName} (N = {metrics.Count}) ---");
        Console.WriteLine($"   • Quality Score (Mean ± Std) : {metrics.Average(m => m.QualityScore):F2} ± {StdDev(metrics.Select(m => (double)m.QualityScore)):F2} (Min: {metrics.Min(m => m.QualityScore):F2}, Max: {metrics.Max(m => m.QualityScore):F2})");
        Console.WriteLine($"   • Near-White Clipping %      : Mean = {metrics.Average(m => m.NearWhiteClipPct) * 100:F1}%, Max = {metrics.Max(m => m.NearWhiteClipPct) * 100:F1}%");
        Console.WriteLine($"   • Near-Black Clipping %      : Mean = {metrics.Average(m => m.NearBlackClipPct) * 100:F1}%, Max = {metrics.Max(m => m.NearBlackClipPct) * 100:F1}%");
        Console.WriteLine($"   • Dynamic Range (P95-P05 L*) : Mean = {metrics.Average(m => m.DynamicRange_L):F1}, Median = {Median(metrics.Select(m => m.DynamicRange_L)):F1}");
        Console.WriteLine($"   • Sharpness Variance         : Mean = {metrics.Average(m => m.SharpnessVariance):F1}, Median = {Median(metrics.Select(m => (float)m.SharpnessVariance)):F1}");
        Console.WriteLine($"   • Valid Garment Pixel %      : Mean = {metrics.Average(m => m.ValidPixelRatio) * 100:F1}%\n");
    }

    private static void EvaluateSingleImageGatePerformance(string setName, List<EvaluatedSampleMetric> samples)
    {
        int total = samples.Count;
        if (total == 0) return;

        int accepted = samples.Count(s => s.Quality.IsAccepted);
        int rejected = total - accepted;

        int rawCorrect = samples.Count(s => s.RawCorrectCIELAB);
        int acceptedCorrect = samples.Count(s => s.Quality.IsAccepted && s.RawCorrectCIELAB);
        int acceptedWrong = samples.Count(s => s.Quality.IsAccepted && !s.RawCorrectCIELAB);

        int rejectedWrong = samples.Count(s => !s.Quality.IsAccepted && !s.RawCorrectCIELAB);
        int rejectedCorrect = samples.Count(s => !s.Quality.IsAccepted && s.RawCorrectCIELAB);

        float rawAcc = (float)rawCorrect / total * 100f;
        float acceptedAcc = accepted > 0 ? (float)acceptedCorrect / accepted * 100f : 0f;
        float coverage = (float)accepted / total * 100f;
        float overallAcc = (float)acceptedCorrect / total * 100f;

        Console.WriteLine($"--- {setName} (Total N = {total}) ---");
        Console.WriteLine($"   • Total Observations       : {total}");
        Console.WriteLine($"   • Accepted Observations    : {accepted} (Coverage = {coverage:F1}%)");
        Console.WriteLine($"   • Rejected Observations    : {rejected} (Rejection Rate = {100f - coverage:F1}%)");
        Console.WriteLine($"   • Raw CIELAB Accuracy      : {rawCorrect}/{total} ({rawAcc:F1}%)");
        Console.WriteLine($"   • Quality-Gated Accepted Acc: {acceptedCorrect}/{accepted} ({acceptedAcc:F1}%)");
        Console.WriteLine($"   • Quality-Gated Overall Acc : {acceptedCorrect}/{total} ({overallAcc:F1}%)");
        Console.WriteLine($"   • Bad Observations Rejected : {rejectedWrong}/{total - rawCorrect} ({(float)rejectedWrong / Math.Max(1, total - rawCorrect) * 100:F1}% of all errors successfully filtered)");
        Console.WriteLine($"   • Good Obs Accidentally Lost: {rejectedCorrect}/{rawCorrect} ({(float)rejectedCorrect / Math.Max(1, rawCorrect) * 100:F1}% false rejection rate)\n");
    }

    private static async Task RunVideoSequenceAudit(
        IPersonDetector detector,
        IPoseEstimationService poseService,
        string dlPath,
        string videoFile,
        string description,
        InferenceSession visionSession,
        Dictionary<string, float[]> upperBank,
        Dictionary<string, float[]> lowerBank)
    {
        string fullPath = Path.Combine(dlPath, videoFile);
        if (!File.Exists(fullPath)) return;

        Console.WriteLine($"\n-----------------------------------------------------------------------------------------");
        Console.WriteLine($" VIDEO SEQUENCE: '{videoFile}' - {description}");
        Console.WriteLine($"-----------------------------------------------------------------------------------------");

        using var capture = new VideoCapture(fullPath);
        if (!capture.IsOpened()) return;

        int frameIdx = 0;
        using var frameMat = new Mat();

        // Independent Best-Frame Pool for Upper and Lower
        var upperObsList = new List<(int Frame, ColorQualityMetrics Quality, string PredCIELAB, string PredCLIP)>();
        var lowerObsList = new List<(int Frame, ColorQualityMetrics Quality, string PredCIELAB, string PredCLIP)>();

        while (capture.Read(frameMat) && frameIdx < 60)
        {
            frameIdx++;
            if (frameIdx % 3 != 0) continue; // Sample at ~8 FPS

            var dets = await detector.DetectPersonsAsync(frameMat);
            if (dets.Count == 0) continue;

            var primaryDet = dets.OrderByDescending(d => d.Box.Width * d.Box.Height).First();
            var poses = await poseService.EstimatePoseAsync(frameMat, new List<DetectionResult> { primaryDet });
            var pose = poses.Count > 0 ? poses[0] : null;

            // 1. Upper Body Candidate
            var uBox = pose?.Regions?.UpperTorsoRegion ?? primaryDet.Box;
            int ux = Math.Clamp((int)uBox.X, 0, frameMat.Width - 1);
            int uy = Math.Clamp((int)uBox.Y, 0, frameMat.Height - 1);
            int uw = Math.Clamp((int)uBox.Width, 1, frameMat.Width - ux);
            int uh = Math.Clamp((int)uBox.Height, 1, frameMat.Height - uy);

            using var uRoi = new Mat(frameMat, new Rect(ux, uy, uw, uh));
            if (uRoi.Width >= 6 && uRoi.Height >= 6)
            {
                var uQ = ComputeColorQualitySignals(uRoi, "Upper", pose, uBox);
                var (uPredCIELAB, _, _, _, _, _) = RigorousColorBenchmarkRunner.EvaluateSimplified(uRoi, frameMat, "Upper", pose, uBox);
                var uImgEmb = FashionClipColorBenchmarkRunner.ExtractImageEmbedding(uRoi, visionSession);
                var uClip = FashionClipColorBenchmarkRunner.ClassifyEmbeddingTop5(uImgEmb, upperBank);

                upperObsList.Add((frameIdx, uQ, uPredCIELAB, uClip.Top1));
            }

            // 2. Lower Body Candidate (Completely Independent History)
            var lBox = pose?.Regions?.LowerBodyRegion;
            if (lBox != null)
            {
                int lx = Math.Clamp((int)lBox.X, 0, frameMat.Width - 1);
                int ly = Math.Clamp((int)lBox.Y, 0, frameMat.Height - 1);
                int lw = Math.Clamp((int)lBox.Width, 1, frameMat.Width - lx);
                int lh = Math.Clamp((int)lBox.Height, 1, frameMat.Height - ly);

                using var lRoi = new Mat(frameMat, new Rect(lx, ly, lw, lh));
                if (lRoi.Width >= 6 && lRoi.Height >= 6)
                {
                    var lQ = ComputeColorQualitySignals(lRoi, "Lower", pose, lBox);
                    var (lPredCIELAB, _, _, _, _, _) = RigorousColorBenchmarkRunner.EvaluateSimplified(lRoi, frameMat, "Lower", pose, lBox);
                    var lImgEmb = FashionClipColorBenchmarkRunner.ExtractImageEmbedding(lRoi, visionSession);
                    var lClip = FashionClipColorBenchmarkRunner.ClassifyEmbeddingTop5(lImgEmb, lowerBank);

                    lowerObsList.Add((frameIdx, lQ, lPredCIELAB, lClip.Top1));
                }
            }
        }

        // Print Temporal Sequence Trace for Upper Garment
        Console.WriteLine($"Temporal Upper Garment Observations Log (Total Frames Sampled: {upperObsList.Count}):");
        Console.WriteLine($"{"Frame",-7} | {"QualityScore",12} | {"Gate",-8} | {"CIELAB Pred",-12} | {"CLIP Pred",-12} | {"Rejection Reason"}");
        Console.WriteLine(new string('-', 85));

        foreach (var obs in upperObsList.Take(12))
        {
            string gate = obs.Quality.IsAccepted ? "ACCEPT" : "REJECT";
            Console.WriteLine($"F{obs.Frame:D3}   | {obs.Quality.QualityScore,12:F2} | {gate,-8} | {obs.PredCIELAB,-12} | {obs.PredCLIP,-12} | {obs.Quality.RejectionReason}");
        }

        // Compute Consensus across the 4 Strategies:
        // Strategy A: Every-frame CIELAB
        string consA = ComputeConsensus(upperObsList.Select(o => o.PredCIELAB).ToList());

        // Strategy B: Every-frame Fashion-CLIP
        string consB = ComputeConsensus(upperObsList.Select(o => o.PredCLIP).ToList());

        // Strategy C: Quality-Gated CIELAB
        var acceptedUpper = upperObsList.Where(o => o.Quality.IsAccepted).ToList();
        string consC = acceptedUpper.Count > 0 ? ComputeConsensus(acceptedUpper.Select(o => o.PredCIELAB).ToList()) : "InsufficientVisualEvidence";

        // Strategy D: Top-K (K=5) Quality-Weighted Best-Frame Pool CIELAB
        var topKUpper = upperObsList.OrderByDescending(o => o.Quality.QualityScore).Take(5).ToList();
        string consD = topKUpper.Count > 0 && topKUpper[0].Quality.QualityScore >= 0.35f
            ? ComputeWeightedConsensus(topKUpper.Select(o => (o.PredCIELAB, o.Quality.QualityScore)).ToList())
            : "InsufficientVisualEvidence";

        // Strategy E: Quality-Gated Fashion-CLIP
        string consE = acceptedUpper.Count > 0 ? ComputeConsensus(acceptedUpper.Select(o => o.PredCLIP).ToList()) : "InsufficientVisualEvidence";

        Console.WriteLine($"\nMulti-Frame Video Consensus Strategy Results (Upper):");
        Console.WriteLine($"   • Strategy A (Every-Frame CIELAB)         : [{consA}] (from {upperObsList.Count} raw frames)");
        Console.WriteLine($"   • Strategy B (Every-Frame Fashion-CLIP)   : [{consB}] (from {upperObsList.Count} raw frames)");
        Console.WriteLine($"   • Strategy C (Quality-Gated CIELAB)       : [{consC}] (from {acceptedUpper.Count} accepted frames, {upperObsList.Count - acceptedUpper.Count} noisy frames rejected)");
        Console.WriteLine($"   • Strategy D (Top-5 Best-Frame Pool CIELAB): [{consD}] (weighted top-5 quality frames, avg Q = {(topKUpper.Count > 0 ? topKUpper.Average(x => x.Quality.QualityScore) : 0):F2})");
        Console.WriteLine($"   • Strategy E (Quality-Gated Fashion-CLIP) : [{consE}] (from {acceptedUpper.Count} accepted frames)\n");
    }

    private static string ComputeConsensus(List<string> preds)
    {
        if (preds.Count == 0) return "InsufficientVisualEvidence";
        return preds.GroupBy(p => p).OrderByDescending(g => g.Count()).First().Key;
    }

    private static string ComputeWeightedConsensus(List<(string Pred, float Weight)> weightedPreds)
    {
        var scores = new Dictionary<string, float>(StringComparer.OrdinalIgnoreCase);
        foreach (var (p, w) in weightedPreds)
        {
            scores[p] = scores.GetValueOrDefault(p, 0f) + w;
        }
        return scores.OrderByDescending(kv => kv.Value).First().Key;
    }

    private static double CalculateLaplacianVariance(Mat image)
    {
        try
        {
            using var gray = new Mat();
            if (image.Channels() == 1) image.CopyTo(gray);
            else Cv2.CvtColor(image, gray, ColorConversionCodes.BGR2GRAY);

            using var lap = new Mat();
            Cv2.Laplacian(gray, lap, MatType.CV_64F);
            Cv2.MeanStdDev(lap, out _, out Scalar std);
            return std.Val0 * std.Val0;
        }
        catch { return 0.0; }
    }

    private static void ApplyPosePolygon(Mat mask, Mat crop, string region, PersonPoseResult pose, BoundingBox targetBox)
    {
        int ox = (int)targetBox.X;
        int oy = (int)targetBox.Y;
        var kps = pose.Keypoints;

        if (region.Equals("Upper", StringComparison.OrdinalIgnoreCase))
        {
            if (kps.Count >= 17 && kps[5].IsVisible(0.2f) && kps[6].IsVisible(0.2f) && (kps[11].IsVisible(0.2f) || kps[12].IsVisible(0.2f)))
            {
                float span = kps[6].X - kps[5].X;
                var pts = new[]
                {
                    new Point(Math.Clamp((int)(kps[5].X - ox + span * 0.10f), 0, crop.Width - 1), Math.Clamp((int)(kps[5].Y - oy + 5), 0, crop.Height - 1)),
                    new Point(Math.Clamp((int)(kps[6].X - ox - span * 0.10f), 0, crop.Width - 1), Math.Clamp((int)(kps[6].Y - oy + 5), 0, crop.Height - 1)),
                    new Point(Math.Clamp((int)(kps[12].X - ox - 5), 0, crop.Width - 1), Math.Clamp((int)(kps[12].Y - oy - 5), 0, crop.Height - 1)),
                    new Point(Math.Clamp((int)(kps[11].X - ox + 5), 0, crop.Width - 1), Math.Clamp((int)(kps[11].Y - oy - 5), 0, crop.Height - 1))
                };
                Cv2.FillConvexPoly(mask, pts, Scalar.White);
            }
            else
            {
                Cv2.Ellipse(mask, new RotatedRect(
                    new Point2f(crop.Width * 0.5f, crop.Height * 0.5f),
                    new Size2f(crop.Width * 0.70f, crop.Height * 0.75f), 0),
                    Scalar.White, -1);
            }
        }
        else
        {
            Cv2.Ellipse(mask, new RotatedRect(
                new Point2f(crop.Width * 0.5f, crop.Height * 0.5f),
                new Size2f(crop.Width * 0.60f, crop.Height * 0.70f), 0),
                Scalar.White, -1);
        }
    }

    private static double StdDev(IEnumerable<double> values)
    {
        var list = values.ToList();
        if (list.Count <= 1) return 0;
        double avg = list.Average();
        double sumSq = list.Sum(d => (d - avg) * (d - avg));
        return Math.Sqrt(sumSq / (list.Count - 1));
    }

    private static float Median(IEnumerable<float> values)
    {
        var list = values.OrderBy(x => x).ToList();
        if (list.Count == 0) return 0;
        return list[list.Count / 2];
    }
}
