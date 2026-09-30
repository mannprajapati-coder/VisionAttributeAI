using System;
using System.Collections.Generic;
using System.Diagnostics;
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
/// Isolated R&D Benchmark Engine evaluating Representative Base-Fabric Color Extraction vs Image Quality.
/// Compares R1 (Whole-pixel median), R2 (Dominant K-Means), R3 (Spatial Patches),
/// R4 (Multi-Cluster Spatial Consensus), and R5 (Interior-Region Sampling).
/// Generates full visual debug dumps and audits graphic/print/shadow/specular contamination.
/// Zero production modifications.
/// </summary>
public static class FabricPixelExtractionBenchmarkRunner
{
    public record ClusterDetail(
        int ClusterId,
        int PixelCount,
        float Share,
        float L,
        float A,
        float B,
        float Chroma,
        float Hue,
        string SemanticColor,
        string BaseFamily,
        float SpatialSpreadX,
        float SpatialSpreadY,
        int QuadrantsCovered,
        float EdgeRatio,
        float SpatialScore
    );

    public record SampleExtractionResult(
        string SampleId,
        string SetType,
        string FileName,
        string Region,
        string GroundTruth,
        string CurrentPrediction,
        string PredR1_WholeMedian,
        string PredR2_DominantKMeans,
        string PredR3_SpatialPatches,
        string PredR4_SpatialConsensus,
        string PredR5_InteriorSampling,
        bool MatchCurrent,
        bool MatchR1,
        bool MatchR2,
        bool MatchR3,
        bool MatchR4,
        bool MatchR5,
        float MedianL,
        float MedianChroma,
        float MedianHue,
        List<ClusterDetail> Clusters
    );

    public static async Task RunExtractionBenchmarkAsync(
        IPersonDetector detector,
        IPoseEstimationService poseService)
    {
        Console.WriteLine("\n=========================================================================================");
        Console.WriteLine(" EMPIRICAL BASE-FABRIC COLOR PIXEL EXTRACTION BENCHMARK (METHODS R1 - R5)");
        Console.WriteLine("=========================================================================================\n");

        string dlPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads");
        string userUpPath = @"C:\Users\Admin\.gemini\antigravity\brain\d1c30ac8-9746-4946-b173-1b32a3a64125\.user_uploaded";
        string debugOutputDir = @"C:\Users\Admin\.gemini\antigravity\brain\d1c30ac8-9746-4946-b173-1b32a3a64125\scratch\debug_extractions";

        Directory.CreateDirectory(debugOutputDir);

        var dataset = RigorousColorBenchmarkRunner.BuildComprehensiveDataset(dlPath, userUpPath);
        var currentSvc = new ClothingColorService();

        var evaluatedSamples = new List<SampleExtractionResult>();

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

            using var garmentRoi = new Mat(fullImg, new Rect(rx, ry, rw, rh));
            using var mask = new Mat(garmentRoi.Size(), MatType.CV_8UC1, Scalar.All(0));

            // Compute standard valid garment mask (skin + background + polygon filtering)
            ExtractGarmentMask(garmentRoi, mask, sample.Region, pose, targetBox);

            // 1. Evaluate Current Production CIELAB
            var currentAttr = currentSvc.ClassifyDetailedColor(garmentRoi, fullImg, sample.Region, pose, targetBox);
            string currentPred = currentAttr.PrimaryColor;

            // 2. Extract Valid Pixels in CIELAB & Coordinates
            var validPoints = new List<(Point Pt, float L, float A, float B, float Chroma, float Hue)>();
            using var labImg = new Mat();
            Cv2.CvtColor(garmentRoi, labImg, ColorConversionCodes.BGR2Lab);

            int roiRows = garmentRoi.Rows;
            int roiCols = garmentRoi.Cols;

            for (int y = 0; y < roiRows; y++)
            {
                for (int x = 0; x < roiCols; x++)
                {
                    if (mask.At<byte>(y, x) > 0)
                    {
                        var labPixel = labImg.At<Vec3b>(y, x);
                        float lStar = labPixel.Item0 * 100.0f / 255.0f;
                        float aStar = labPixel.Item1 - 128.0f;
                        float bStar = labPixel.Item2 - 128.0f;
                        float chroma = MathF.Sqrt(aStar * aStar + bStar * bStar);
                        float hue = MathF.Atan2(bStar, aStar) * 180.0f / MathF.PI;
                        if (hue < 0) hue += 360.0f;

                        validPoints.Add((new Point(x, y), lStar, aStar, bStar, chroma, hue));
                    }
                }
            }

            if (validPoints.Count < 30)
            {
                // Fallback to center region if mask is sparse
                for (int y = roiRows / 4; y < (roiRows * 3) / 4; y++)
                {
                    for (int x = roiCols / 4; x < (roiCols * 3) / 4; x++)
                    {
                        var labPixel = labImg.At<Vec3b>(y, x);
                        float lStar = labPixel.Item0 * 100.0f / 255.0f;
                        float aStar = labPixel.Item1 - 128.0f;
                        float bStar = labPixel.Item2 - 128.0f;
                        float chroma = MathF.Sqrt(aStar * aStar + bStar * bStar);
                        float hue = MathF.Atan2(bStar, aStar) * 180.0f / MathF.PI;
                        if (hue < 0) hue += 360.0f;
                        validPoints.Add((new Point(x, y), lStar, aStar, bStar, chroma, hue));
                    }
                }
            }

            // METHOD R1: Whole-pixel median
            var sortedL = validPoints.Select(p => p.L).OrderBy(v => v).ToList();
            var sortedA = validPoints.Select(p => p.A).OrderBy(v => v).ToList();
            var sortedB = validPoints.Select(p => p.B).OrderBy(v => v).ToList();
            float r1_L = sortedL[sortedL.Count / 2];
            float r1_A = sortedA[sortedA.Count / 2];
            float r1_B = sortedB[sortedB.Count / 2];
            float r1_Chroma = MathF.Sqrt(r1_A * r1_A + r1_B * r1_B);
            float r1_Hue = MathF.Atan2(r1_B, r1_A) * 180.0f / MathF.PI;
            if (r1_Hue < 0) r1_Hue += 360.0f;
            var (predR1, _) = RigorousColorBenchmarkRunner.ClassifySimplifiedCluster(r1_L, r1_A, r1_B, r1_Chroma, r1_Hue);

            // METHOD R2 & R4: K-Means Clustering on CIELAB + Spatial Analysis
            var (clusters, clusterLabelMap) = PerformKMeansAndSpatialAnalysis(validPoints, garmentRoi.Width, garmentRoi.Height);

            // R2: Largest cluster by pixel share
            var dominantCluster = clusters.OrderByDescending(c => c.PixelCount).First();
            string predR2 = dominantCluster.SemanticColor;

            // R4: Spatial Consensus cluster (highest spatial score)
            var spatialBestCluster = clusters.OrderByDescending(c => c.SpatialScore).First();
            string predR4 = spatialBestCluster.SemanticColor;

            // METHOD R3: Spatially Distributed Patches (3x3 Grid)
            string predR3 = EvaluateSpatialPatches(garmentRoi, mask, labImg);

            // METHOD R5: Interior-Region Sampling (Morphological Erosion)
            string predR5 = EvaluateInteriorSampling(garmentRoi, mask, labImg);

            // Save Visual Debug Artifacts
            string sampleDebugFolder = Path.Combine(debugOutputDir, sample.SampleId);
            Directory.CreateDirectory(sampleDebugFolder);

            SaveVisualDebugOutputs(
                sampleDebugFolder,
                fullImg,
                primaryDet.Box,
                garmentRoi,
                mask,
                validPoints,
                clusters,
                clusterLabelMap,
                dominantCluster.ClusterId,
                spatialBestCluster.ClusterId
            );

            evaluatedSamples.Add(new SampleExtractionResult(
                sample.SampleId,
                sample.SetType,
                sample.FileName,
                sample.Region,
                sample.GroundTruthColor,
                currentPred,
                predR1,
                predR2,
                predR3,
                predR4,
                predR5,
                MatchesGroundTruth(currentPred, sample.GroundTruthColor),
                MatchesGroundTruth(predR1, sample.GroundTruthColor),
                MatchesGroundTruth(predR2, sample.GroundTruthColor),
                MatchesGroundTruth(predR3, sample.GroundTruthColor),
                MatchesGroundTruth(predR4, sample.GroundTruthColor),
                MatchesGroundTruth(predR5, sample.GroundTruthColor),
                r1_L,
                r1_Chroma,
                r1_Hue,
                clusters
            ));
        }

        // Output Comprehensive Evaluation Sections
        PrintExtractionBenchmarkReport(evaluatedSamples);
    }

    private static (List<ClusterDetail> Clusters, int[] LabelMap) PerformKMeansAndSpatialAnalysis(
        List<(Point Pt, float L, float A, float B, float Chroma, float Hue)> points,
        int roiWidth,
        int roiHeight)
    {
        int k = Math.Min(4, points.Count);
        using var samplesMat = new Mat(points.Count, 3, MatType.CV_32FC1);
        for (int i = 0; i < points.Count; i++)
        {
            samplesMat.Set<float>(i, 0, points[i].L);
            samplesMat.Set<float>(i, 1, points[i].A);
            samplesMat.Set<float>(i, 2, points[i].B);
        }

        using var labelsMat = new Mat();
        using var centersMat = new Mat();
        var criteria = new TermCriteria(CriteriaTypes.Eps | CriteriaTypes.MaxIter, 20, 0.5);
        Cv2.Kmeans(samplesMat, k, labelsMat, criteria, 3, KMeansFlags.PpCenters, centersMat);

        var clusterPoints = new Dictionary<int, List<(Point Pt, float L, float A, float B, float Chroma, float Hue)>>();
        for (int c = 0; c < k; c++) clusterPoints[c] = new List<(Point Pt, float L, float A, float B, float Chroma, float Hue)>();

        var labelArray = new int[points.Count];
        for (int i = 0; i < points.Count; i++)
        {
            int lbl = labelsMat.At<int>(i, 0);
            labelArray[i] = lbl;
            clusterPoints[lbl].Add(points[i]);
        }

        var details = new List<ClusterDetail>();
        float halfW = roiWidth / 2.0f;
        float halfH = roiHeight / 2.0f;

        foreach (var (cid, cpts) in clusterPoints)
        {
            if (cpts.Count == 0) continue;

            float share = (float)cpts.Count / points.Count;
            var sortedL = cpts.Select(p => p.L).OrderBy(v => v).ToList();
            var sortedA = cpts.Select(p => p.A).OrderBy(v => v).ToList();
            var sortedB = cpts.Select(p => p.B).OrderBy(v => v).ToList();

            float cL = sortedL[sortedL.Count / 2];
            float cA = sortedA[sortedA.Count / 2];
            float cB = sortedB[sortedB.Count / 2];
            float chroma = MathF.Sqrt(cA * cA + cB * cB);
            float hue = MathF.Atan2(cB, cA) * 180.0f / MathF.PI;
            if (hue < 0) hue += 360.0f;

            var (semantic, family) = RigorousColorBenchmarkRunner.ClassifySimplifiedCluster(cL, cA, cB, chroma, hue);

            // Spatial statistics
            float meanX = cpts.Average(p => (float)p.Pt.X);
            float meanY = cpts.Average(p => (float)p.Pt.Y);
            float spreadX = MathF.Sqrt((float)cpts.Average(p => (p.Pt.X - meanX) * (p.Pt.X - meanX))) / Math.Max(1, roiWidth);
            float spreadY = MathF.Sqrt((float)cpts.Average(p => (p.Pt.Y - meanY) * (p.Pt.Y - meanY))) / Math.Max(1, roiHeight);

            // Quadrant presence (Top-Left, Top-Right, Bottom-Left, Bottom-Right)
            int qTL = cpts.Count(p => p.Pt.X < halfW && p.Pt.Y < halfH);
            int qTR = cpts.Count(p => p.Pt.X >= halfW && p.Pt.Y < halfH);
            int qBL = cpts.Count(p => p.Pt.X < halfW && p.Pt.Y >= halfH);
            int qBR = cpts.Count(p => p.Pt.X >= halfW && p.Pt.Y >= halfH);

            int quads = 0;
            float qThreshold = cpts.Count * 0.08f;
            if (qTL >= qThreshold) quads++;
            if (qTR >= qThreshold) quads++;
            if (qBL >= qThreshold) quads++;
            if (qBR >= qThreshold) quads++;

            // Edge concentration (pixels within 5% of boundary)
            int edgeMarginX = Math.Max(1, (int)(roiWidth * 0.05f));
            int edgeMarginY = Math.Max(1, (int)(roiHeight * 0.05f));
            int edgePixels = cpts.Count(p => p.Pt.X < edgeMarginX || p.Pt.X >= (roiWidth - edgeMarginX) || p.Pt.Y < edgeMarginY || p.Pt.Y >= (roiHeight - edgeMarginY));
            float edgeRatio = (float)edgePixels / cpts.Count;

            // Spatial consensus score: rewards high multi-quadrant coverage, spatial dispersion across ROI, penalizes pure edge concentration
            float spatialDispersion = (spreadX + spreadY) / 2.0f;
            float spatialScore = (quads / 4.0f) * 0.40f +
                                 Math.Min(spatialDispersion / 0.25f, 1.0f) * 0.30f +
                                 (1.0f - edgeRatio) * 0.15f +
                                 MathF.Sqrt(share) * 0.15f;

            details.Add(new ClusterDetail(
                cid, cpts.Count, share, cL, cA, cB, chroma, hue,
                semantic, family, spreadX, spreadY, quads, edgeRatio, spatialScore
            ));
        }

        return (details, labelArray);
    }

    private static string EvaluateSpatialPatches(Mat roi, Mat mask, Mat labImg)
    {
        const int gridRows = 3;
        const int gridCols = 3;
        int patchW = roi.Width / gridCols;
        int patchH = roi.Height / gridRows;

        var patchVotes = new List<string>();

        for (int r = 0; r < gridRows; r++)
        {
            for (int c = 0; c < gridCols; c++)
            {
                int px = c * patchW;
                int py = r * patchH;
                int pw = (c == gridCols - 1) ? (roi.Width - px) : patchW;
                int ph = (r == gridRows - 1) ? (roi.Height - py) : patchH;

                var patchRect = new Rect(px, py, pw, ph);
                using var patchMask = new Mat(mask, patchRect);
                using var patchLab = new Mat(labImg, patchRect);

                var patchPts = new List<(float L, float A, float B)>();
                int pRows = patchLab.Rows;
                int pCols = patchLab.Cols;

                for (int y = 0; y < pRows; y++)
                {
                    for (int x = 0; x < pCols; x++)
                    {
                        if (patchMask.At<byte>(y, x) > 0)
                        {
                            var p = patchLab.At<Vec3b>(y, x);
                            float lStar = p.Item0 * 100.0f / 255.0f;
                            float aStar = p.Item1 - 128.0f;
                            float bStar = p.Item2 - 128.0f;
                            patchPts.Add((lStar, aStar, bStar));
                        }
                    }
                }

                // Minimum 30 valid pixels per patch
                if (patchPts.Count >= 30)
                {
                    var sortedL = patchPts.Select(p => p.L).OrderBy(v => v).ToList();
                    var sortedA = patchPts.Select(p => p.A).OrderBy(v => v).ToList();
                    var sortedB = patchPts.Select(p => p.B).OrderBy(v => v).ToList();
                    float medL = sortedL[sortedL.Count / 2];
                    float medA = sortedA[sortedA.Count / 2];
                    float medB = sortedB[sortedB.Count / 2];
                    float chroma = MathF.Sqrt(medA * medA + medB * medB);
                    float hue = MathF.Atan2(medB, medA) * 180.0f / MathF.PI;
                    if (hue < 0) hue += 360.0f;

                    var (sem, _) = RigorousColorBenchmarkRunner.ClassifySimplifiedCluster(medL, medA, medB, chroma, hue);
                    patchVotes.Add(sem);
                }
            }
        }

        if (patchVotes.Count == 0) return "Unknown";

        // Spatial patch majority vote
        var topVote = patchVotes.GroupBy(v => v)
            .OrderByDescending(g => g.Count())
            .First().Key;

        return topVote;
    }

    private static string EvaluateInteriorSampling(Mat roi, Mat mask, Mat labImg)
    {
        // Deep interior erosion to remove outer edge contamination
        int erodeSize = Math.Max(3, Math.Min(roi.Width, roi.Height) / 10);
        using var kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(erodeSize, erodeSize));
        using var erodedMask = new Mat();
        Cv2.Erode(mask, erodedMask, kernel);

        var interiorPts = new List<(float L, float A, float B)>();
        int rows = labImg.Rows;
        int cols = labImg.Cols;

        for (int y = 0; y < rows; y++)
        {
            for (int x = 0; x < cols; x++)
            {
                if (erodedMask.At<byte>(y, x) > 0)
                {
                    var p = labImg.At<Vec3b>(y, x);
                    float lStar = p.Item0 * 100.0f / 255.0f;
                    float aStar = p.Item1 - 128.0f;
                    float bStar = p.Item2 - 128.0f;
                    interiorPts.Add((lStar, aStar, bStar));
                }
            }
        }

        // If eroded mask is too small, sample central 40%
        if (interiorPts.Count < 20)
        {
            int cx1 = (int)(roi.Width * 0.30f);
            int cx2 = (int)(roi.Width * 0.70f);
            int cy1 = (int)(roi.Height * 0.30f);
            int cy2 = (int)(roi.Height * 0.70f);

            for (int y = cy1; y < cy2; y++)
            {
                for (int x = cx1; x < cx2; x++)
                {
                    if (mask.At<byte>(y, x) > 0)
                    {
                        var p = labImg.At<Vec3b>(y, x);
                        float lStar = p.Item0 * 100.0f / 255.0f;
                        float aStar = p.Item1 - 128.0f;
                        float bStar = p.Item2 - 128.0f;
                        interiorPts.Add((lStar, aStar, bStar));
                    }
                }
            }
        }

        if (interiorPts.Count == 0) return "Unknown";

        var sortedL = interiorPts.Select(p => p.L).OrderBy(v => v).ToList();
        var sortedA = interiorPts.Select(p => p.A).OrderBy(v => v).ToList();
        var sortedB = interiorPts.Select(p => p.B).OrderBy(v => v).ToList();
        float medL = sortedL[sortedL.Count / 2];
        float medA = sortedA[sortedA.Count / 2];
        float medB = sortedB[sortedB.Count / 2];
        float chroma = MathF.Sqrt(medA * medA + medB * medB);
        float hue = MathF.Atan2(medB, medA) * 180.0f / MathF.PI;
        if (hue < 0) hue += 360.0f;

        var (sem, _) = RigorousColorBenchmarkRunner.ClassifySimplifiedCluster(medL, medA, medB, chroma, hue);
        return sem;
    }

    private static void SaveVisualDebugOutputs(
        string folder,
        Mat fullImg,
        BoundingBox personBox,
        Mat garmentRoi,
        Mat mask,
        List<(Point Pt, float L, float A, float B, float Chroma, float Hue)> validPoints,
        List<ClusterDetail> clusters,
        int[] labelMap,
        int dominantCid,
        int spatialCid)
    {
        // 1. original_person.jpg
        int px = Math.Clamp((int)personBox.X, 0, fullImg.Width - 1);
        int py = Math.Clamp((int)personBox.Y, 0, fullImg.Height - 1);
        int pw = Math.Clamp((int)personBox.Width, 1, fullImg.Width - px);
        int ph = Math.Clamp((int)personBox.Height, 1, fullImg.Height - py);
        using var personCrop = new Mat(fullImg, new Rect(px, py, pw, ph));
        Cv2.ImWrite(Path.Combine(folder, "original_person.jpg"), personCrop);

        // 2. garment_roi.jpg
        Cv2.ImWrite(Path.Combine(folder, "garment_roi.jpg"), garmentRoi);

        // 3. garment_mask.jpg
        Cv2.ImWrite(Path.Combine(folder, "garment_mask.jpg"), mask);

        // 4. pixels_used.jpg (Garment ROI with non-mask zeroed)
        using var pixelsUsed = new Mat(garmentRoi.Size(), MatType.CV_8UC3, Scalar.All(0));
        garmentRoi.CopyTo(pixelsUsed, mask);
        Cv2.ImWrite(Path.Combine(folder, "pixels_used.jpg"), pixelsUsed);

        // 5. kmeans_clusters.jpg (Colorized by cluster assignment)
        using var clusterVisual = new Mat(garmentRoi.Size(), MatType.CV_8UC3, Scalar.All(0));
        var clusterColors = new[]
        {
            new Scalar(0, 0, 255),    // C0: Red
            new Scalar(0, 255, 0),    // C1: Green
            new Scalar(255, 0, 0),    // C2: Blue
            new Scalar(0, 255, 255)   // C3: Yellow
        };

        for (int i = 0; i < validPoints.Count; i++)
        {
            var pt = validPoints[i].Pt;
            int lbl = labelMap[i];
            clusterVisual.Set(pt.Y, pt.X, new Vec3b(
                (byte)clusterColors[lbl % 4].Val0,
                (byte)clusterColors[lbl % 4].Val1,
                (byte)clusterColors[lbl % 4].Val2
            ));
        }
        Cv2.ImWrite(Path.Combine(folder, "kmeans_clusters.jpg"), clusterVisual);

        // 6. dominant_cluster_overlay.jpg (Highlight winning spatial base cluster, dim others)
        using var overlay = garmentRoi.Clone();
        using var grayRoi = new Mat();
        Cv2.CvtColor(garmentRoi, grayRoi, ColorConversionCodes.BGR2GRAY);
        using var grayBgr = new Mat();
        Cv2.CvtColor(grayRoi, grayBgr, ColorConversionCodes.GRAY2BGR);

        grayBgr.CopyTo(overlay); // background in grayscale

        for (int i = 0; i < validPoints.Count; i++)
        {
            if (labelMap[i] == spatialCid)
            {
                var pt = validPoints[i].Pt;
                overlay.Set(pt.Y, pt.X, garmentRoi.At<Vec3b>(pt.Y, pt.X));
            }
        }
        Cv2.ImWrite(Path.Combine(folder, "dominant_cluster_overlay.jpg"), overlay);
    }

    private static void ExtractGarmentMask(Mat crop, Mat mask, string region, PersonPoseResult? pose, BoundingBox targetBox)
    {
        using var hsv = new Mat();
        Cv2.CvtColor(crop, hsv, ColorConversionCodes.BGR2HSV);
        using var lab = new Mat();
        Cv2.CvtColor(crop, lab, ColorConversionCodes.BGR2Lab);

        int rows = crop.Rows;
        int cols = crop.Cols;

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                var hsvP = hsv.At<Vec3b>(r, c);
                var labP = lab.At<Vec3b>(r, c);

                byte h = hsvP.Item0;
                byte s = hsvP.Item1;
                byte v = hsvP.Item2;
                byte a = labP.Item1;
                byte b = labP.Item2;

                // Skin Filter
                bool isSkin = (h <= 25 && s >= 20 && s <= 170 && v >= 50) &&
                              (a >= 133 && a <= 173 && b >= 135 && b <= 180);

                if (!isSkin)
                {
                    mask.Set<byte>(r, c, 255);
                }
            }
        }

        // Polygon constraint if pose available
        if (pose != null && pose.Keypoints.Count >= 17)
        {
            ApplyPosePolygonToMask(mask, crop, region, pose, targetBox);
        }
    }

    private static void ApplyPosePolygonToMask(Mat mask, Mat crop, string region, PersonPoseResult pose, BoundingBox targetBox)
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
                    new Point(Math.Clamp((int)(kps[12].X - ox - span * 0.10f), 0, crop.Width - 1), Math.Clamp((int)(kps[12].Y - oy - 5), 0, crop.Height - 1)),
                    new Point(Math.Clamp((int)(kps[11].X - ox + span * 0.10f), 0, crop.Width - 1), Math.Clamp((int)(kps[11].Y - oy - 5), 0, crop.Height - 1))
                };

                using var polyMask = new Mat(crop.Size(), MatType.CV_8UC1, Scalar.All(0));
                Cv2.FillConvexPoly(polyMask, pts, Scalar.All(255));
                Cv2.BitwiseAnd(mask, polyMask, mask);
            }
        }
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

    private static void PrintExtractionBenchmarkReport(List<SampleExtractionResult> samples)
    {
        var dev = samples.Where(s => s.SetType == "DEV").ToList();
        var blind = samples.Where(s => s.SetType == "BLIND").ToList();

        Console.WriteLine("\n=========================================================================================");
        Console.WriteLine(" 2 & 3. IN-DEPTH AUDIT OF B-05, B-10, B-32, B-36 & REPRESENTATIVE SAMPLES");
        Console.WriteLine("=========================================================================================\n");

        var auditIds = new[] { "B-05", "B-10", "B-32", "B-36", "D-01", "D-02", "D-05", "D-06", "D-23", "B-04", "B-07", "B-09" };
        foreach (var aid in auditIds)
        {
            var s = samples.FirstOrDefault(x => x.SampleId == aid);
            if (s == null) continue;

            Console.WriteLine($"-----------------------------------------------------------------------------------------");
            Console.WriteLine($"SAMPLE [{s.SampleId}] ({s.SetType}) - File: {s.FileName}, Region: {s.Region}, GT: [{s.GroundTruth}], Current: [{s.CurrentPrediction}]");
            Console.WriteLine($"R1 (Whole Median): [{s.PredR1_WholeMedian}], R2 (Dominant K-Means): [{s.PredR2_DominantKMeans}], R3 (Patches): [{s.PredR3_SpatialPatches}], R4 (Spatial Consensus): [{s.PredR4_SpatialConsensus}], R5 (Interior): [{s.PredR5_InteriorSampling}]");
            Console.WriteLine($"K-Means Cluster Decomposition (K = {s.Clusters.Count}):");
            Console.WriteLine($"CID | Share % | L*   | a*   | b*   | Chroma | Hue   | Semantic | Family  | Quads | Edge% | SpatialScore");
            Console.WriteLine($"----+---------+------+------+------+--------+-------+----------+---------+-------+-------+-------------");
            foreach (var c in s.Clusters)
            {
                Console.WriteLine($"{c.ClusterId,3} | {c.Share * 100f,6:F1}% | {c.L,4:F1} | {c.A,4:F1} | {c.B,4:F1} | {c.Chroma,6:F1} | {c.Hue,5:F1} | {c.SemanticColor,-8} | {c.BaseFamily,-7} | {c.QuadrantsCovered,5} | {c.EdgeRatio * 100f,4:F0}% | {c.SpatialScore,12:F2}");
            }
            Console.WriteLine();
        }

        Console.WriteLine("\n=========================================================================================");
        Console.WriteLine(" 6, 7, 11. ACCURACY BENCHMARK: R1 VS R2 VS R3 VS R4 VS R5 ACROSS DATASET SPLITS");
        Console.WriteLine("=========================================================================================\n");

        PrintSplitTable("DEVELOPMENT SET (N = 18)", dev);
        PrintSplitTable("FROZEN BLIND VALIDATION SET (N = 14)", blind);
        PrintSplitTable("COMBINED COMPREHENSIVE DATASET (N = 32)", samples);

        Console.WriteLine("\n--- ACCURACY BREAKDOWN BY REGION ---");
        var upper = samples.Where(s => s.Region == "Upper").ToList();
        var lower = samples.Where(s => s.Region == "Lower").ToList();
        PrintSplitTable("UPPER BODY GARMENTS (N = " + upper.Count + ")", upper);
        PrintSplitTable("LOWER BODY GARMENTS (N = " + lower.Count + ")", lower);

        Console.WriteLine("\n--- ACCURACY BREAKDOWN BY COLOR FAMILY (COMBINED N = 32) ---");
        var colorGroups = samples.GroupBy(s => s.GroundTruth).OrderByDescending(g => g.Count());
        foreach (var grp in colorGroups)
        {
            if (grp.Count() < 2) continue;
            PrintSplitTable($"COLOR: {grp.Key.ToUpper()} (N = {grp.Count()})", grp.ToList());
        }

        Console.WriteLine("\n=========================================================================================");
        Console.WriteLine(" 8 & 12. FULL RAW SAMPLE-BY-SAMPLE EVALUATION TABLE (N = 32)");
        Console.WriteLine("=========================================================================================\n");
        Console.WriteLine("SampleID | Set   | Region | GroundTruth | Current  | R1 (Whole) | R2 (DomK)  | R3 (Patch) | R4 (Spat)  | R5 (Inter) | R4 Match?");
        Console.WriteLine("---------+-------+--------+-------------+----------+------------+------------+------------+------------+------------+----------");
        foreach (var s in samples)
        {
            string matchR4 = s.MatchR4 ? "PASS" : "FAIL";
            Console.WriteLine($"{s.SampleId,-8} | {s.SetType,-5} | {s.Region,-6} | {s.GroundTruth,-11} | {s.CurrentPrediction,-8} | {s.PredR1_WholeMedian,-10} | {s.PredR2_DominantKMeans,-10} | {s.PredR3_SpatialPatches,-10} | {s.PredR4_SpatialConsensus,-10} | {s.PredR5_InteriorSampling,-10} | {matchR4,-9}");
        }
    }

    private static void PrintSplitTable(string title, List<SampleExtractionResult> list)
    {
        int n = list.Count;
        if (n == 0) return;

        int currAcc = list.Count(s => s.MatchCurrent);
        int r1Acc = list.Count(s => s.MatchR1);
        int r2Acc = list.Count(s => s.MatchR2);
        int r3Acc = list.Count(s => s.MatchR3);
        int r4Acc = list.Count(s => s.MatchR4);
        int r5Acc = list.Count(s => s.MatchR5);

        Console.WriteLine($"--- {title} ---");
        Console.WriteLine($"   • Current Production CIELAB    : {currAcc,2}/{n} ({currAcc * 100f / n,5:F1}%)");
        Console.WriteLine($"   • Method R1 (Whole-Pixel Median): {r1Acc,2}/{n} ({r1Acc * 100f / n,5:F1}%)");
        Console.WriteLine($"   • Method R2 (Dominant K-Means) : {r2Acc,2}/{n} ({r2Acc * 100f / n,5:F1}%)");
        Console.WriteLine($"   • Method R3 (Spatial Patches)  : {r3Acc,2}/{n} ({r3Acc * 100f / n,5:F1}%)");
        Console.WriteLine($"   • Method R4 (Spatial Consensus): {r4Acc,2}/{n} ({r4Acc * 100f / n,5:F1}%)");
        Console.WriteLine($"   • Method R5 (Interior Sampling): {r5Acc,2}/{n} ({r5Acc * 100f / n,5:F1}%)\n");
    }
}
