using Microsoft.Extensions.Options;
using OpenCvSharp;
using VisionAttributeAI.Models.AI;
using VisionAttributeAI.Models.Pose;

namespace VisionAttributeAI.Services.Attributes;

/// <summary>
/// Robust, deterministic Clothing Color Recognition Service.
/// Implements Pose-Guided Polygon Masking, Skin/Hair Tone Filtering,
/// CIELAB + HSV Hybrid Perceptual Feature Space, K-Means Clustering,
/// and Semantic Multi-Color Decomposition.
/// </summary>
public interface IClothingColorService
{
    ClipClassificationResult ClassifyClothingColor(Mat clothingCrop);
    ColorClassificationResult ClassifyDetailedColor(Mat clothingCrop);
    ColorClassificationResult ClassifyDetailedColor(
        Mat clothingCrop,
        Mat? fullImage = null,
        string? region = null,
        PersonPoseResult? pose = null,
        BoundingBox? targetBox = null);
}

public class ClothingColorService : IClothingColorService
{
    private readonly ColorAnalysisOptions _options;

    public ClothingColorService(IOptions<ColorAnalysisOptions>? options = null)
    {
        _options = options?.Value ?? new ColorAnalysisOptions();
    }

    public ClipClassificationResult ClassifyClothingColor(Mat clothingCrop)
    {
        var detailed = ClassifyDetailedColor(clothingCrop);
        var scores = detailed.AllColorPercentages;

        return new ClipClassificationResult
        {
            TopCategory = detailed.PrimaryColor,
            TopConfidence = detailed.ColorConfidence,
            CategoryScores = scores,
            Margin = detailed.PrimaryPercentage - detailed.SecondaryPercentage
        };
    }

    public ColorClassificationResult ClassifyDetailedColor(Mat clothingCrop)
    {
        return ClassifyDetailedColor(clothingCrop, null, null, null, null);
    }

    public ColorClassificationResult ClassifyDetailedColor(
        Mat clothingCrop,
        Mat? fullImage = null,
        string? region = null,
        PersonPoseResult? pose = null,
        BoundingBox? targetBox = null)
    {
        if (clothingCrop == null || clothingCrop.Empty() || clothingCrop.Width < 8 || clothingCrop.Height < 8)
        {
            return new ColorClassificationResult
            {
                PrimaryColor = "InsufficientVisualEvidence",
                ColorConfidence = 0.0f
            };
        }

        // 1. Build Pose-Guided Foreground Clothing Mask
        using var mask = new Mat(clothingCrop.Size(), MatType.CV_8UC1, Scalar.All(0));

        if (_options.EnablePoseGuidedMasking && pose != null && targetBox != null)
        {
            ApplyPoseGuidedMask(mask, clothingCrop, region ?? "Upper", pose, targetBox);
        }
        else
        {
            // Default Central 75% Ellipse to avoid outer borders/background
            Cv2.Ellipse(mask, new RotatedRect(
                new Point2f(clothingCrop.Width * 0.5f, clothingCrop.Height * 0.5f),
                new Size2f(clothingCrop.Width * 0.72f, clothingCrop.Height * 0.78f), 0),
                Scalar.White, -1);
        }

        // 2. Filter Skin & Hair (YCrCb Skin Space)
        if (_options.EnableSkinExclusion)
        {
            using var ycrcb = new Mat();
            Cv2.CvtColor(clothingCrop, ycrcb, ColorConversionCodes.BGR2YCrCb);
            using var skinMask = new Mat();
            Cv2.InRange(ycrcb, new Scalar(0, 133, 77), new Scalar(255, 173, 127), skinMask);
            Cv2.BitwiseNot(skinMask, skinMask);
            Cv2.BitwiseAnd(mask, skinMask, mask);
        }

        // 3. Extract Valid Clothing Pixels in CIELAB and HSV
        using var labMat = new Mat();
        using var hsvMat = new Mat();
        Cv2.CvtColor(clothingCrop, labMat, ColorConversionCodes.BGR2Lab);
        Cv2.CvtColor(clothingCrop, hsvMat, ColorConversionCodes.BGR2HSV);

        var validPixels = new List<Vec3f>();

        int rows = clothingCrop.Rows;
        int cols = clothingCrop.Cols;

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                if (mask.At<byte>(r, c) > 0)
                {
                    var lp = labMat.At<Vec3b>(r, c);
                    validPixels.Add(new Vec3f(lp.Item0, lp.Item1, lp.Item2));
                }
            }
        }

        if (validPixels.Count < _options.MinValidPixelCount)
        {
            return new ColorClassificationResult
            {
                PrimaryColor = "InsufficientVisualEvidence",
                ColorConfidence = 0.0f
            };
        }

        // 4. K-Means Clustering on CIELAB Space
        int K = Math.Clamp(_options.KMeansClusters, 2, Math.Max(2, validPixels.Count / 25));
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
        for (int i = 0; i < validPixels.Count; i++)
        {
            clusterCounts[labels.At<int>(i)]++;
        }

        // 5. Extract Cluster Descriptors & Apply Shadow/Highlight Cluster Merging
        var clusters = new List<(float L, float a, float b, float Chroma, float Hue, float Share, string Color)>();
        for (int k = 0; k < K; k++)
        {
            float clusterShare = (float)clusterCounts[k] / validPixels.Count;
            if (clusterShare < _options.MinClusterShare) continue;

            float cL = centers.At<float>(k, 0) * 100f / 255f;
            float ca = centers.At<float>(k, 1) - 128f;
            float cb = centers.At<float>(k, 2) - 128f;
            float chroma = MathF.Sqrt(ca * ca + cb * cb);

            float hueAngle = MathF.Atan2(cb, ca) * 180f / MathF.PI;
            if (hueAngle < 0) hueAngle += 360f;

            string clusterColor = ClassifyCluster(cL, ca, cb, chroma, hueAngle);
            clusters.Add((cL, ca, cb, chroma, hueAngle, clusterShare, clusterColor));
        }

        var colorVotes = new Dictionary<string, float>(StringComparer.OrdinalIgnoreCase);
        foreach (var cName in _options.SupportedColors) colorVotes[cName] = 0f;

        foreach (var c in clusters)
        {
            colorVotes[c.Color] = colorVotes.GetValueOrDefault(c.Color, 0f) + c.Share;
        }

        // Normalize Votes
        float totalVote = colorVotes.Values.Sum();
        if (totalVote > 0)
        {
            foreach (var key in colorVotes.Keys.ToList())
            {
                colorVotes[key] /= totalVote;
            }
        }

        var sortedVotes = colorVotes.OrderByDescending(kv => kv.Value).ToList();
        var top = sortedVotes[0];
        var second = sortedVotes.Count > 1 ? sortedVotes[1] : new KeyValuePair<string, float>("None", 0f);

        if (top.Value < _options.MinPrimaryVoteShare)
        {
            return new ColorClassificationResult
            {
                PrimaryColor = "Unknown",
                ColorConfidence = top.Value,
                AllColorPercentages = colorVotes
            };
        }

        // Multi-Color Evaluation
        bool isMulti = top.Value < _options.MultiColorPrimaryThreshold &&
                       second.Value >= _options.SecondaryColorMinShare &&
                       !string.Equals(second.Key, "None", StringComparison.OrdinalIgnoreCase);

        float confidence = isMulti
            ? Math.Min(1.0f, top.Value + second.Value * 0.5f)
            : top.Value;

        return new ColorClassificationResult
        {
            PrimaryColor = top.Key,
            PrimaryPercentage = top.Value,
            SecondaryColor = second.Value >= 0.15f ? second.Key : "None",
            SecondaryPercentage = second.Value,
            ColorConfidence = confidence,
            IsMultiColor = isMulti,
            AllColorPercentages = colorVotes
        };
    }

    private static void ApplyPoseGuidedMask(
        Mat mask,
        Mat crop,
        string region,
        PersonPoseResult pose,
        BoundingBox targetBox)
    {
        int ox = (int)targetBox.X;
        int oy = (int)targetBox.Y;
        var kps = pose.Keypoints;

        if (region.Equals("Upper", StringComparison.OrdinalIgnoreCase))
        {
            // Torso Polygon: Shoulders -> Hips with 10% horizontal safety inset to eliminate arm/background
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
        else // Lower Body
        {
            bool hasLHip = kps.Count >= 17 && kps[11].IsVisible(0.2f);
            bool hasRHip = kps.Count >= 17 && kps[12].IsVisible(0.2f);
            bool hasLKnee = kps.Count >= 17 && kps[13].IsVisible(0.2f);
            bool hasRKnee = kps.Count >= 17 && kps[14].IsVisible(0.2f);

            float hSpan = (hasLHip && hasRHip) ? Math.Abs(kps[12].X - kps[11].X) : 0f;

            if (hasLHip && hasRHip && hSpan > 25f && (hasLKnee || hasRKnee))
            {
                float legW = Math.Max(20f, hSpan * 0.45f);

                // Left Leg Corridor
                if (hasLKnee)
                {
                    var leftLegPts = new[]
                    {
                        new Point((int)Math.Clamp(kps[11].X - ox - legW * 0.5f, 0, crop.Width - 1), (int)Math.Clamp(kps[11].Y - oy + 5, 0, crop.Height - 1)),
                        new Point((int)Math.Clamp(kps[11].X - ox + legW * 0.5f, 0, crop.Width - 1), (int)Math.Clamp(kps[11].Y - oy + 5, 0, crop.Height - 1)),
                        new Point((int)Math.Clamp(kps[13].X - ox + legW * 0.4f, 0, crop.Width - 1), (int)Math.Clamp(kps[13].Y - oy - 5, 0, crop.Height - 1)),
                        new Point((int)Math.Clamp(kps[13].X - ox - legW * 0.4f, 0, crop.Width - 1), (int)Math.Clamp(kps[13].Y - oy - 5, 0, crop.Height - 1))
                    };
                    Cv2.FillConvexPoly(mask, leftLegPts, Scalar.White);
                }

                // Right Leg Corridor
                if (hasRKnee)
                {
                    var rightLegPts = new[]
                    {
                        new Point((int)Math.Clamp(kps[12].X - ox - legW * 0.5f, 0, crop.Width - 1), (int)Math.Clamp(kps[12].Y - oy + 5, 0, crop.Height - 1)),
                        new Point((int)Math.Clamp(kps[12].X - ox + legW * 0.5f, 0, crop.Width - 1), (int)Math.Clamp(kps[12].Y - oy + 5, 0, crop.Height - 1)),
                        new Point((int)Math.Clamp(kps[14].X - ox + legW * 0.4f, 0, crop.Width - 1), (int)Math.Clamp(kps[14].Y - oy - 5, 0, crop.Height - 1)),
                        new Point((int)Math.Clamp(kps[14].X - ox - legW * 0.4f, 0, crop.Width - 1), (int)Math.Clamp(kps[14].Y - oy - 5, 0, crop.Height - 1))
                    };
                    Cv2.FillConvexPoly(mask, rightLegPts, Scalar.White);
                }
            }
            else if (hasLHip || hasRHip)
            {
                // Side / Profile / Single visible leg View: Corridor centered on visible hip and visible knee
                float midHipX = (hasLHip && hasRHip) ? (kps[11].X + kps[12].X) * 0.5f : (hasLHip ? kps[11].X : kps[12].X);
                float midHipY = (hasLHip && hasRHip) ? Math.Max(kps[11].Y, kps[12].Y) : (hasLHip ? kps[11].Y : kps[12].Y);

                float kneeX = (hasLKnee && hasRKnee) ? (kps[13].X + kps[14].X) * 0.5f : (hasLKnee ? kps[13].X : (hasRKnee ? kps[14].X : midHipX));
                float kneeY = (hasLKnee && hasRKnee) ? (kps[13].Y + kps[14].Y) * 0.5f : (hasLKnee ? kps[13].Y : (hasRKnee ? kps[14].Y : targetBox.Y + targetBox.Height));

                float legRadius = Math.Clamp(crop.Width * 0.35f, 15f, 60f);

                var profileLegPts = new[]
                {
                    new Point((int)Math.Clamp(midHipX - ox - legRadius, 0, crop.Width - 1), (int)Math.Clamp(midHipY - oy + 5, 0, crop.Height - 1)),
                    new Point((int)Math.Clamp(midHipX - ox + legRadius, 0, crop.Width - 1), (int)Math.Clamp(midHipY - oy + 5, 0, crop.Height - 1)),
                    new Point((int)Math.Clamp(kneeX - ox + legRadius * 0.8f, 0, crop.Width - 1), (int)Math.Clamp(kneeY - oy - 5, 0, crop.Height - 1)),
                    new Point((int)Math.Clamp(kneeX - ox - legRadius * 0.8f, 0, crop.Width - 1), (int)Math.Clamp(kneeY - oy - 5, 0, crop.Height - 1))
                };
                Cv2.FillConvexPoly(mask, profileLegPts, Scalar.White);
            }
            else
            {
                Cv2.Ellipse(mask, new RotatedRect(
                    new Point2f(crop.Width * 0.5f, crop.Height * 0.5f),
                    new Size2f(crop.Width * 0.60f, crop.Height * 0.70f), 0),
                    Scalar.White, -1);
            }
        }
    }

    private string ClassifyCluster(float L, float a, float b, float chroma, float hueAngle)
    {
        // 1. Achromatic Classification:
        // Black allows realistic sensor/lighting noise (Chroma < 5.8) for dark garments (L < 24.0).
        if (L < _options.BlackMaxLightness && chroma < 5.8f)
        {
            return "Black";
        }
        if (chroma < 5.0f && L >= _options.WhiteMinLightness)
        {
            return "White";
        }
        if (chroma < _options.AchromaticMaxChroma && L >= _options.BlackMaxLightness && L < _options.WhiteMinLightness)
        {
            return "Grey";
        }

        // 2. Chromatic Mapping via CIELAB Hue Angle & Lightness
        // Red / Maroon sector: True Maroon requires genuine red saturation (Chroma >= 6.5 and a* >= 4.5)
        if (hueAngle >= 345f || hueAngle < 24f)
        {
            if (L < 38f && chroma >= 6.5f && a >= 4.5f) return "Maroon";
            if (L < _options.BlackMaxLightness) return "Black"; // Low-saturation dark fabric defaults to Black
            if (L > 65f && chroma < 30f) return "Pink";
            return "Red";
        }

        if (hueAngle >= 24f && hueAngle < 58f)
        {
            if (L < 42f && chroma >= 5.5f) return "Brown";
            if (L < _options.BlackMaxLightness) return "Black";
            return "Orange";
        }

        if (hueAngle >= 58f && hueAngle < 95f)
        {
            if (chroma < 26f && L > 50f) return "Beige";
            if (L < _options.BlackMaxLightness) return "Black";
            return "Yellow";
        }

        if (hueAngle >= 95f && hueAngle < 175f)
        {
            if (L < _options.BlackMaxLightness && chroma < 5.5f) return "Black";
            return "Green"; // Includes Olive / Dark Green
        }

        if (hueAngle >= 175f && hueAngle < 225f)
        {
            if (L < _options.BlackMaxLightness && chroma < 5.5f) return "Black";
            return "Cyan";
        }

        if (hueAngle >= 225f && hueAngle < 285f)
        {
            if (L < 36f && chroma >= 5.8f) return "Navy";
            if (L < _options.BlackMaxLightness) return "Black";
            return "Blue";
        }

        if (hueAngle >= 285f && hueAngle < 345f)
        {
            if (L < 38f && chroma >= 6.5f && a > 4.5f) return "Maroon";
            if (L < _options.BlackMaxLightness) return "Black";
            if (L > 60f) return "Pink";
            return "Purple";
        }

        return "Grey";
    }
}
