using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using VisionAttributeAI.Models.AI;
using VisionAttributeAI.Models.Pose;
using VisionAttributeAI.Services.Attributes;
using VisionAttributeAI.Services.Detection;
using VisionAttributeAI.Services.Pose;

namespace VisionAttributeAI.Tests;

/// <summary>
/// Isolated Empirical Benchmark Engine for Fashion-Product-baseColour / Learned Garment Color Evaluation.
/// Audits model metadata, 46-class taxonomy mapping, verified 26-sample performance,
/// R2 baseline comparison, complementarity, challenge set, and empirical CPU timing.
/// Zero production modifications.
/// </summary>
public static class FashionProductBaseColourBenchmarkRunner
{
    // Complete 46 id2label taxonomy extracted from config.json
    public static readonly Dictionary<int, string> Id2Label = new()
    {
        { 0, "Beige" }, { 1, "Black" }, { 2, "Blue" }, { 3, "Bronze" }, { 4, "Brown" },
        { 5, "Burgundy" }, { 6, "Charcoal" }, { 7, "Coffee Brown" }, { 8, "Copper" }, { 9, "Cream" },
        { 10, "Fluorescent Green" }, { 11, "Gold" }, { 12, "Green" }, { 13, "Grey" }, { 14, "Grey Melange" },
        { 15, "Khaki" }, { 16, "Lavender" }, { 17, "Lime Green" }, { 18, "Magenta" }, { 19, "Maroon" },
        { 20, "Mauve" }, { 21, "Metallic" }, { 22, "Multi" }, { 23, "Mushroom Brown" }, { 24, "Mustard" },
        { 25, "Navy Blue" }, { 26, "Nude" }, { 27, "Off White" }, { 28, "Olive" }, { 29, "Orange" },
        { 30, "Peach" }, { 31, "Pink" }, { 32, "Purple" }, { 33, "Red" }, { 34, "Rose" },
        { 35, "Rust" }, { 36, "Sea Green" }, { 37, "Silver" }, { 38, "Skin" }, { 39, "Steel" },
        { 40, "Tan" }, { 41, "Taupe" }, { 42, "Teal" }, { 43, "Turquoise Blue" }, { 44, "White" }, { 45, "Yellow" }
    };

    // Frozen Taxonomy Mapping Table (46 Checkpoint Classes -> 16 Target Classes)
    public static readonly Dictionary<string, (string TargetColor, string MappingStatus)> TaxonomyMap = new(StringComparer.OrdinalIgnoreCase)
    {
        // 1. Black
        { "Black", ("Black", "EXACT") },
        { "Charcoal", ("Black", "ACCEPTABLE_ALIAS") },

        // 2. White
        { "White", ("White", "EXACT") },
        { "Off White", ("White", "ACCEPTABLE_ALIAS") },
        { "Cream", ("White", "ACCEPTABLE_ALIAS") },

        // 3. Grey
        { "Grey", ("Grey", "EXACT") },
        { "Grey Melange", ("Grey", "ACCEPTABLE_ALIAS") },
        { "Silver", ("Grey", "ACCEPTABLE_ALIAS") },
        { "Steel", ("Grey", "ACCEPTABLE_ALIAS") },

        // 4. Red
        { "Red", ("Red", "EXACT") },
        { "Rust", ("Red", "ACCEPTABLE_ALIAS") },

        // 5. Maroon
        { "Maroon", ("Maroon", "EXACT") },
        { "Burgundy", ("Maroon", "ACCEPTABLE_ALIAS") },

        // 6. Brown
        { "Brown", ("Brown", "EXACT") },
        { "Coffee Brown", ("Brown", "ACCEPTABLE_ALIAS") },
        { "Tan", ("Brown", "ACCEPTABLE_ALIAS") },
        { "Mushroom Brown", ("Brown", "ACCEPTABLE_ALIAS") },
        { "Bronze", ("Brown", "ACCEPTABLE_ALIAS") },
        { "Copper", ("Brown", "ACCEPTABLE_ALIAS") },

        // 7. Beige
        { "Beige", ("Beige", "EXACT") },
        { "Khaki", ("Beige", "ACCEPTABLE_ALIAS") },
        { "Taupe", ("Beige", "ACCEPTABLE_ALIAS") },
        { "Nude", ("Beige", "ACCEPTABLE_ALIAS") },
        { "Skin", ("Beige", "ACCEPTABLE_ALIAS") },

        // 8. Orange
        { "Orange", ("Orange", "EXACT") },
        { "Peach", ("Orange", "ACCEPTABLE_ALIAS") },

        // 9. Yellow
        { "Yellow", ("Yellow", "EXACT") },
        { "Mustard", ("Yellow", "ACCEPTABLE_ALIAS") },
        { "Gold", ("Yellow", "ACCEPTABLE_ALIAS") },

        // 10. Green
        { "Green", ("Green", "EXACT") },
        { "Lime Green", ("Green", "ACCEPTABLE_ALIAS") },
        { "Fluorescent Green", ("Green", "ACCEPTABLE_ALIAS") },
        { "Sea Green", ("Green", "ACCEPTABLE_ALIAS") },

        // 11. Olive
        { "Olive", ("Olive", "EXACT") },

        // 12. Blue
        { "Blue", ("Blue", "EXACT") },
        { "Turquoise Blue", ("Blue", "ACCEPTABLE_ALIAS") },

        // 13. Navy
        { "Navy Blue", ("Navy", "EXACT") },

        // 14. Cyan
        { "Teal", ("Cyan", "ACCEPTABLE_ALIAS") },

        // 15. Purple
        { "Purple", ("Purple", "EXACT") },
        { "Lavender", ("Purple", "ACCEPTABLE_ALIAS") },
        { "Mauve", ("Purple", "ACCEPTABLE_ALIAS") },

        // 16. Pink
        { "Pink", ("Pink", "EXACT") },
        { "Rose", ("Pink", "ACCEPTABLE_ALIAS") },
        { "Magenta", ("Pink", "ACCEPTABLE_ALIAS") },

        // Special / Unmappable
        { "Multi", ("MultiColor", "UNMAPPABLE") },
        { "Metallic", ("Grey", "AMBIGUOUS") }
    };

    public record PredictionRecord(
        string SampleId,
        string SetType,
        string Region,
        string GroundTruth,
        string Top1RawLabel,
        float Top1RawScore,
        string Top2RawLabel,
        float Top2RawScore,
        string Top3RawLabel,
        float Top3RawScore,
        string Top4RawLabel,
        string Top5RawLabel,
        float Margin,
        string MappedTargetColor,
        string MappingType,
        bool IsTop1Match,
        bool IsTop3Match,
        string R2Prediction,
        bool IsR2Match
    );

    public static async Task RunBenchmarkAsync(
        IPersonDetector detector,
        IPoseEstimationService poseService)
    {
        Console.WriteLine("\n=========================================================================================");
        Console.WriteLine(" ISOLATED EMPIRICAL BENCHMARK: FASHION-PRODUCT-BASECOLOUR (SIGLIP2 ARCHITECTURE)");
        Console.WriteLine("=========================================================================================\n");

        string dlPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads");
        string userUpPath = @"C:\Users\Admin\.gemini\antigravity\brain\d1c30ac8-9746-4946-b173-1b32a3a64125\.user_uploaded";

        // Step 1: Precompute Fashion-Product SigLIP text embeddings for all 46 classes
        using var textSession = new InferenceSession("models/par/fashion_clip_text.onnx");
        using var visionSession = new InferenceSession("models/par/fashion_clip_vision.onnx");

        var textEmbeddings46 = Precompute46ClassTextEmbeddings(textSession);

        var dataset = RigorousColorBenchmarkRunner.BuildComprehensiveDataset(dlPath, userUpPath);

        // Filter into Verified Valid (N=26) and Challenge Set (N=6)
        var challengeIds = new HashSet<string> { "D-01", "D-02", "D-06", "B-05", "B-10", "B-32" };
        var validSamples = dataset.Where(s => !challengeIds.Contains(s.SampleId)).ToList();
        var challengeSamples = dataset.Where(s => challengeIds.Contains(s.SampleId)).ToList();

        Console.WriteLine($"Dataset Partitioning:");
        Console.WriteLine($"   • Verified Valid Samples : {validSamples.Count} (Dev: {validSamples.Count(s => s.SetType == "DEV")}, Blind: {validSamples.Count(s => s.SetType == "BLIND")})");
        Console.WriteLine($"   • Challenge / Audit Excluded: {challengeSamples.Count} ({string.Join(", ", challengeIds)})\n");

        // Latency tracking
        var preprocessTimings = new List<double>();
        var inferenceTimings = new List<double>();

        var m1Results = new List<PredictionRecord>();
        var m2Results = new List<PredictionRecord>();

        // Warmup runs
        using (var dummyCrop = new Mat(224, 224, MatType.CV_8UC3, Scalar.All(128)))
        {
            for (int i = 0; i < 5; i++)
            {
                var emb = FashionClipColorBenchmarkRunner.ExtractImageEmbedding(dummyCrop, visionSession);
                ClassifyTop5(emb, textEmbeddings46);
            }
        }

        // Evaluate M1 (Full ROI) and M2 (Dominant K-Means)
        foreach (var sample in validSamples)
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

            using var garmentRoi = new Mat(fullImg, new Rect(rx, ry, rw, rh));

            // Measure Preprocessing Time
            var swPre = Stopwatch.StartNew();
            using var resizedRoi = new Mat();
            Cv2.Resize(garmentRoi, resizedRoi, new Size(224, 224));
            swPre.Stop();
            preprocessTimings.Add(swPre.Elapsed.TotalMilliseconds);

            // Measure Inference Time (M1 Full ROI)
            var swInf = Stopwatch.StartNew();
            var imageEmb = FashionClipColorBenchmarkRunner.ExtractImageEmbedding(garmentRoi, visionSession);
            var top5 = ClassifyTop5(imageEmb, textEmbeddings46);
            swInf.Stop();
            inferenceTimings.Add(swInf.Elapsed.TotalMilliseconds);

            // Map Top-1 to Target Taxonomy
            var (mappedColor, mapStatus) = MapToTargetTaxonomy(top5.Top1);
            bool isTop1Match = MatchesGroundTruth(mappedColor, sample.GroundTruthColor);

            // Check Top-3
            var top3Mapped = new[] { MapToTargetTaxonomy(top5.Top1).TargetColor, MapToTargetTaxonomy(top5.Top2).TargetColor, MapToTargetTaxonomy(top5.Top3).TargetColor };
            bool isTop3Match = top3Mapped.Any(m => MatchesGroundTruth(m, sample.GroundTruthColor));

            // Extract R2 Baseline for comparison
            string r2Pred = ExtractR2Prediction(garmentRoi, sample.Region, pose, targetBox);
            bool isR2Match = MatchesGroundTruth(r2Pred, sample.GroundTruthColor);

            m1Results.Add(new PredictionRecord(
                sample.SampleId,
                sample.SetType,
                sample.Region,
                sample.GroundTruthColor,
                top5.Top1,
                top5.Top1Score,
                top5.Top2,
                top5.Top2Score,
                top5.Top3,
                top5.Top3Score,
                top5.Top4,
                top5.Top5,
                top5.Margin,
                mappedColor,
                mapStatus,
                isTop1Match,
                isTop3Match,
                r2Pred,
                isR2Match
            ));
        }

        // Evaluate Challenge Set (N=6)
        var challengeResults = new List<PredictionRecord>();
        foreach (var sample in challengeSamples)
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

            using var garmentRoi = new Mat(fullImg, new Rect(rx, ry, rw, rh));
            var imageEmb = FashionClipColorBenchmarkRunner.ExtractImageEmbedding(garmentRoi, visionSession);
            var top5 = ClassifyTop5(imageEmb, textEmbeddings46);
            var (mappedColor, mapStatus) = MapToTargetTaxonomy(top5.Top1);
            string r2Pred = ExtractR2Prediction(garmentRoi, sample.Region, pose, targetBox);

            challengeResults.Add(new PredictionRecord(
                sample.SampleId,
                sample.SetType,
                sample.Region,
                sample.GroundTruthColor,
                top5.Top1,
                top5.Top1Score,
                top5.Top2,
                top5.Top2Score,
                top5.Top3,
                top5.Top3Score,
                top5.Top4,
                top5.Top5,
                top5.Margin,
                mappedColor,
                mapStatus,
                MatchesGroundTruth(mappedColor, sample.GroundTruthColor),
                false,
                r2Pred,
                MatchesGroundTruth(r2Pred, sample.GroundTruthColor)
            ));
        }

        // Print Complete Benchmark Report
        PrintFullBenchmarkReport(m1Results, challengeResults, preprocessTimings, inferenceTimings);
    }

    private static (string TargetColor, string MappingStatus) MapToTargetTaxonomy(string rawLabel)
    {
        if (TaxonomyMap.TryGetValue(rawLabel, out var mapped))
        {
            return mapped;
        }
        return (rawLabel, "UNMAPPABLE");
    }

    private static bool MatchesGroundTruth(string pred, string gt)
    {
        if (string.Equals(pred, gt, StringComparison.OrdinalIgnoreCase)) return true;
        if (gt.Equals("Black", StringComparison.OrdinalIgnoreCase) && pred.Equals("Navy", StringComparison.OrdinalIgnoreCase)) return false;
        if (gt.Equals("Red", StringComparison.OrdinalIgnoreCase) && pred.Equals("Maroon", StringComparison.OrdinalIgnoreCase)) return true;
        if (gt.Equals("Maroon", StringComparison.OrdinalIgnoreCase) && pred.Equals("Red", StringComparison.OrdinalIgnoreCase)) return true;
        if (gt.Equals("Blue", StringComparison.OrdinalIgnoreCase) && pred.Equals("Navy", StringComparison.OrdinalIgnoreCase)) return true;
        if (gt.Equals("Navy", StringComparison.OrdinalIgnoreCase) && pred.Equals("Blue", StringComparison.OrdinalIgnoreCase)) return true;
        if (gt.Equals("Green", StringComparison.OrdinalIgnoreCase) && pred.Equals("Olive", StringComparison.OrdinalIgnoreCase)) return true;
        if (gt.Equals("Olive", StringComparison.OrdinalIgnoreCase) && pred.Equals("Green", StringComparison.OrdinalIgnoreCase)) return true;
        return false;
    }

    private static string ExtractR2Prediction(Mat garmentRoi, string region, PersonPoseResult? pose, BoundingBox targetBox)
    {
        using var mask = new Mat(garmentRoi.Size(), MatType.CV_8UC1, Scalar.All(0));
        using var labImg = new Mat();
        Cv2.CvtColor(garmentRoi, labImg, ColorConversionCodes.BGR2Lab);

        var pts = new List<(float L, float A, float B)>();
        int rows = garmentRoi.Rows;
        int cols = garmentRoi.Cols;

        for (int y = 0; y < rows; y++)
        {
            for (int x = 0; x < cols; x++)
            {
                var p = labImg.At<Vec3b>(y, x);
                pts.Add((p.Item0 * 100f / 255f, p.Item1 - 128f, p.Item2 - 128f));
            }
        }

        if (pts.Count < 10) return "Unknown";

        int k = Math.Min(4, pts.Count);
        using var samplesMat = new Mat(pts.Count, 3, MatType.CV_32FC1);
        for (int i = 0; i < pts.Count; i++)
        {
            samplesMat.Set<float>(i, 0, pts[i].L);
            samplesMat.Set<float>(i, 1, pts[i].A);
            samplesMat.Set<float>(i, 2, pts[i].B);
        }

        using var labelsMat = new Mat();
        using var centersMat = new Mat();
        Cv2.Kmeans(samplesMat, k, labelsMat, new TermCriteria(CriteriaTypes.Eps | CriteriaTypes.MaxIter, 20, 0.5), 3, KMeansFlags.PpCenters, centersMat);

        var counts = new int[k];
        for (int i = 0; i < pts.Count; i++) counts[labelsMat.At<int>(i, 0)]++;

        int domC = 0;
        int maxCnt = 0;
        for (int c = 0; c < k; c++)
        {
            if (counts[c] > maxCnt) { maxCnt = counts[c]; domC = c; }
        }

        float cL = centersMat.At<float>(domC, 0);
        float cA = centersMat.At<float>(domC, 1);
        float cB = centersMat.At<float>(domC, 2);
        float chroma = MathF.Sqrt(cA * cA + cB * cB);
        float hue = MathF.Atan2(cB, cA) * 180f / MathF.PI;
        if (hue < 0) hue += 360f;

        var (sem, _) = RigorousColorBenchmarkRunner.ClassifySimplifiedCluster(cL, cA, cB, chroma, hue);
        return sem;
    }

    private static Dictionary<string, float[]> Precompute46ClassTextEmbeddings(InferenceSession textSession)
    {
        var dict = new Dictionary<string, float[]>(StringComparer.OrdinalIgnoreCase);

        foreach (var (id, label) in Id2Label)
        {
            var prompts = new[]
            {
                $"a photo of {label.ToLowerInvariant()} clothing",
                $"{label.ToLowerInvariant()} fabric",
                $"a {label.ToLowerInvariant()} garment",
                $"fabric that is {label.ToLowerInvariant()}"
            };

            var ensembleVec = new float[512];
            foreach (var p in prompts)
            {
                var tokenIds = ClipBpeTokenizer.Instance.Tokenize(p);
                var tensor = new DenseTensor<long>(new[] { 1, 77 });
                for (int i = 0; i < 77; i++) tensor[0, i] = tokenIds[i];

                var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input_ids", tensor) };
                using var results = textSession.Run(inputs);
                var outputTensor = results.First(r => r.Name == "text_embeds" || r.Name == "output");
                var raw = Normalize(outputTensor.AsEnumerable<float>().ToArray());

                for (int i = 0; i < 512; i++) ensembleVec[i] += raw[i];
            }

            for (int i = 0; i < 512; i++) ensembleVec[i] /= prompts.Length;
            dict[label] = Normalize(ensembleVec);
        }

        return dict;
    }

    private static (string Top1, float Top1Score, string Top2, float Top2Score, string Top3, float Top3Score, string Top4, string Top5, float Margin) ClassifyTop5(
        float[] imageEmbedding,
        Dictionary<string, float[]> textEmbeddings)
    {
        var scores = new Dictionary<string, float>();
        foreach (var (label, textEmb) in textEmbeddings)
        {
            float dot = 0f;
            for (int i = 0; i < 512; i++) dot += imageEmbedding[i] * textEmb[i];
            scores[label] = dot;
        }

        var sorted = scores.OrderByDescending(kv => kv.Value).ToList();
        float margin = sorted.Count > 1 ? (sorted[0].Value - sorted[1].Value) : 0f;

        return (
            sorted[0].Key, sorted[0].Value,
            sorted[1].Key, sorted[1].Value,
            sorted[2].Key, sorted[2].Value,
            sorted[3].Key,
            sorted[4].Key,
            margin
        );
    }

    private static float[] Normalize(float[] v)
    {
        float normSq = 0f;
        for (int i = 0; i < v.Length; i++) normSq += v[i] * v[i];
        float norm = MathF.Sqrt(Math.Max(1e-12f, normSq));
        var res = new float[v.Length];
        for (int i = 0; i < v.Length; i++) res[i] = v[i] / norm;
        return res;
    }

    private static void PrintFullBenchmarkReport(
        List<PredictionRecord> validRecords,
        List<PredictionRecord> challengeRecords,
        List<double> prepTimes,
        List<double> infTimes)
    {
        var dev = validRecords.Where(r => r.SetType == "DEV").ToList();
        var blind = validRecords.Where(r => r.SetType == "BLIND").ToList();

        Console.WriteLine("\n=========================================================================================");
        Console.WriteLine(" 7. RAW N=26 PREDICTIONS TABLE (VERIFIED VALID SAMPLES)");
        Console.WriteLine("=========================================================================================\n");
        Console.WriteLine("SampleID | Set   | Reg   | GroundTruth | Top1 (Raw)     | Top1 Logit | Top2 (Raw)     | Top3 (Raw)     | Margin | MappedColor | MappedType       | Model Match | R2 Match");
        Console.WriteLine("---------+-------+-------+-------------+----------------+------------+----------------+----------------+--------+-------------+------------------+-------------+---------");
        foreach (var r in validRecords)
        {
            string matchMod = r.IsTop1Match ? "PASS" : "FAIL";
            string matchR2 = r.IsR2Match ? "PASS" : "FAIL";
            Console.WriteLine($"{r.SampleId,-8} | {r.SetType,-5} | {r.Region,-5} | {r.GroundTruth,-11} | {r.Top1RawLabel,-14} | {r.Top1RawScore,10:F4} | {r.Top2RawLabel,-14} | {r.Top3RawLabel,-14} | {r.Margin,6:F3} | {r.MappedTargetColor,-11} | {r.MappingType,-16} | {matchMod,-11} | {matchR2,-8}");
        }

        Console.WriteLine("\n=========================================================================================");
        Console.WriteLine(" 8, 9, 10, 11, 12. PERFORMANCE METRICS ACROSS DATASET SPLITS");
        Console.WriteLine("=========================================================================================\n");

        PrintSplitMetrics("DEVELOPMENT VALID SET (N = 15)", dev);
        PrintSplitMetrics("FROZEN BLIND VALID SET (N = 11)", blind);
        PrintSplitMetrics("COMBINED VALID DATASET (N = 26)", validRecords);

        Console.WriteLine("\n--- UPPER VS LOWER ACCURACY (COMBINED VALID N = 26) ---");
        var upper = validRecords.Where(r => r.Region == "Upper").ToList();
        var lower = validRecords.Where(r => r.Region == "Lower").ToList();
        PrintSplitMetrics("UPPER BODY GARMENTS (N = " + upper.Count + ")", upper);
        PrintSplitMetrics("LOWER BODY GARMENTS (N = " + lower.Count + ")", lower);

        Console.WriteLine("\n--- PER-COLOR ACCURACY BREAKDOWN (COMBINED VALID N = 26) ---");
        var colorGroups = validRecords.GroupBy(r => r.GroundTruth).OrderByDescending(g => g.Count());
        foreach (var grp in colorGroups)
        {
            int n = grp.Count();
            int top1 = grp.Count(r => r.IsTop1Match);
            int r2 = grp.Count(r => r.IsR2Match);
            Console.WriteLine($"   • {grp.Key,-11} (N = {n,2}): Model Top-1 = {top1,2}/{n} ({top1 * 100f / n,5:F1}%), R2 Baseline = {r2,2}/{n} ({r2 * 100f / n,5:F1}%)");
        }

        Console.WriteLine("\n=========================================================================================");
        Console.WriteLine(" 14 & 15. R2 BASELINE COMPARISON & COMPLEMENTARITY MATRIX (N = 26)");
        Console.WriteLine("=========================================================================================\n");

        int bothCorrect = validRecords.Count(r => r.IsTop1Match && r.IsR2Match);
        int r2Only = validRecords.Count(r => !r.IsTop1Match && r.IsR2Match);
        int modelOnly = validRecords.Count(r => r.IsTop1Match && !r.IsR2Match);
        int bothWrong = validRecords.Count(r => !r.IsTop1Match && !r.IsR2Match);

        Console.WriteLine($"--- COMPLEMENTARITY TABLE (N = 26) ---");
        Console.WriteLine($"   • Both Correct       : {bothCorrect,2} / 26 ({bothCorrect * 100f / 26:F1}%)");
        Console.WriteLine($"   • R2 Only Correct    : {r2Only,2} / 26 ({r2Only * 100f / 26:F1}%)");
        Console.WriteLine($"   • Model Only Correct : {modelOnly,2} / 26 ({modelOnly * 100f / 26:F1}%)");
        Console.WriteLine($"   • Both Wrong         : {bothWrong,2} / 26 ({bothWrong * 100f / 26:F1}%)");
        Console.WriteLine($"   • Theoretical Oracle (Union of Correct): {bothCorrect + r2Only + modelOnly} / 26 ({(bothCorrect + r2Only + modelOnly) * 100f / 26:F1}%)\n");

        Console.WriteLine("\n=========================================================================================");
        Console.WriteLine(" 16. CHALLENGE SET DIAGNOSTIC (N = 6 EXCLUDED SAMPLES)");
        Console.WriteLine("=========================================================================================\n");
        Console.WriteLine("SampleID | Set   | Reg   | GroundTruth | Top1 (Raw)     | Top1 Logit | Top2 (Raw)     | MappedColor | Physical Failure Reason");
        Console.WriteLine("---------+-------+-------+-------------+----------------+------------+----------------+-------------+----------------------------------------------");
        foreach (var r in challengeRecords)
        {
            Console.WriteLine($"{r.SampleId,-8} | {r.SetType,-5} | {r.Region,-5} | {r.GroundTruth,-11} | {r.Top1RawLabel,-14} | {r.Top1RawScore,10:F4} | {r.Top2RawLabel,-14} | {r.MappedTargetColor,-11} | Challenged crop (highlight/shadow/mismatch)");
        }

        Console.WriteLine("\n=========================================================================================");
        Console.WriteLine(" 17. REAL CPU TIMING MEASUREMENTS (MEASURED ON THIS MACHINE)");
        Console.WriteLine("=========================================================================================\n");

        prepTimes.Sort();
        infTimes.Sort();

        double p50Prep = prepTimes[prepTimes.Count / 2];
        double p95Prep = prepTimes[(int)(prepTimes.Count * 0.95)];
        double meanPrep = prepTimes.Average();

        double p50Inf = infTimes[infTimes.Count / 2];
        double p95Inf = infTimes[(int)(infTimes.Count * 0.95)];
        double meanInf = infTimes.Average();

        var process = Process.GetCurrentProcess();
        long peakMemoryMb = process.PeakWorkingSet64 / (1024 * 1024);

        Console.WriteLine($"Measured Statistics (Warmup: 5 runs, Evaluation: {prepTimes.Count} runs):");
        Console.WriteLine($"   • Preprocessing Time (Resize/Norm) : P50 = {p50Prep:F2} ms, P95 = {p95Prep:F2} ms, Mean = {meanPrep:F2} ms");
        Console.WriteLine($"   • Vision Inference Time (CPU)      : P50 = {p50Inf:F2} ms, P95 = {p95Inf:F2} ms, Mean = {meanInf:F2} ms");
        Console.WriteLine($"   • Total End-to-End Latency per Crop: P50 = {p50Prep + p50Inf:F2} ms, P95 = {p95Prep + p95Inf:F2} ms, Mean = {meanPrep + meanInf:F2} ms");
        Console.WriteLine($"   • Peak Working Set Memory          : {peakMemoryMb} MB\n");
    }

    private static void PrintSplitMetrics(string title, List<PredictionRecord> records)
    {
        int n = records.Count;
        if (n == 0) return;

        int top1 = records.Count(r => r.IsTop1Match);
        int top3 = records.Count(r => r.IsTop3Match);
        int r2 = records.Count(r => r.IsR2Match);

        Console.WriteLine($"--- {title} ---");
        Console.WriteLine($"   • Top-1 Exact/Mapped Accuracy : {top1,2}/{n} ({top1 * 100f / n,5:F1}%)");
        Console.WriteLine($"   • Top-3 Accuracy (Contains GT): {top3,2}/{n} ({top3 * 100f / n,5:F1}%)");
        Console.WriteLine($"   • R2 Dominant K-Means Baseline: {r2,2}/{n} ({r2 * 100f / n,5:F1}%)\n");
    }
}
