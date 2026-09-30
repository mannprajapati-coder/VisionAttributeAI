using Microsoft.Extensions.Options;
using OpenCvSharp;
using VisionAttributeAI.Models.AI;
using VisionAttributeAI.Models.Tracking;

namespace VisionAttributeAI.Services.Quality;

/// <summary>
/// Evaluates frame quality using OpenCV Laplacian variance (sharpness/blur), bounding box dimensions, and YOLO confidence.
/// Rejects low-confidence, small, invalid, or blurry person crops to prevent noisy aggregation.
/// </summary>
public class FrameQualityFilter : IFrameQualityFilter
{
    private readonly LiveTrackingOptions _options;
    private readonly ILogger<FrameQualityFilter> _logger;

    public FrameQualityFilter(
        IOptions<LiveTrackingOptions> options,
        ILogger<FrameQualityFilter> logger)
    {
        _options = options.Value;
        _logger = logger;
    }

    public TrackQualityResult EvaluateCropQuality(Mat cropMat, BoundingBox box, float detectionConfidence)
    {
        if (cropMat == null || cropMat.Empty() || cropMat.Width <= 0 || cropMat.Height <= 0)
        {
            return TrackQualityResult.Rejected("Invalid or empty crop image buffer.");
        }

        // 1. Check YOLO Detection Confidence
        if (detectionConfidence < _options.MinDetectionConfidence)
        {
            return TrackQualityResult.Rejected(
                $"YOLO detection confidence too low ({detectionConfidence:P1} < {_options.MinDetectionConfidence:P1}).");
        }

        // 2. Check Dimensions & Area
        int width = cropMat.Width;
        int height = cropMat.Height;
        long area = (long)width * height;

        if (width < _options.MinCropWidth)
        {
            return TrackQualityResult.Rejected(
                $"Crop width too small ({width}px < {_options.MinCropWidth}px).");
        }

        if (height < _options.MinCropHeight)
        {
            return TrackQualityResult.Rejected(
                $"Crop height too small ({height}px < {_options.MinCropHeight}px).");
        }

        if (area < _options.MinCropArea)
        {
            return TrackQualityResult.Rejected(
                $"Crop area too small ({area}px² < {_options.MinCropArea}px²).");
        }

        // 3. Check Image Blur / Sharpness via OpenCV Laplacian Variance
        double sharpnessVariance = CalculateLaplacianVariance(cropMat);
        if (sharpnessVariance < _options.MinSharpnessVariance)
        {
            return TrackQualityResult.Rejected(
                $"Crop is blurry (Sharpness variance {sharpnessVariance:F1} < {_options.MinSharpnessVariance:F1}).",
                sharpnessVariance);
        }

        // 4. Calculate Composite Quality Score [0.1 .. 1.0] and Observation Weight
        // Higher confidence, larger resolution, and crisper focus yield higher weight
        float confFactor = Math.Clamp(detectionConfidence, 0.4f, 1.0f);
        float sizeFactor = Math.Clamp((float)area / 40000f, 0.2f, 1.0f);
        float sharpnessFactor = Math.Clamp((float)(sharpnessVariance / 150.0), 0.2f, 1.0f);

        float qualityScore = (confFactor * 0.45f) + (sizeFactor * 0.30f) + (sharpnessFactor * 0.25f);
        float computedWeight = Math.Clamp(qualityScore * confFactor, 0.1f, 1.0f);

        return TrackQualityResult.Accepted(sharpnessVariance, qualityScore, computedWeight);
    }

    private static double CalculateLaplacianVariance(Mat image)
    {
        try
        {
            using var gray = new Mat();
            if (image.Channels() == 1)
            {
                image.CopyTo(gray);
            }
            else
            {
                Cv2.CvtColor(image, gray, ColorConversionCodes.BGR2GRAY);
            }

            using var laplacian = new Mat();
            Cv2.Laplacian(gray, laplacian, MatType.CV_64F);

            Cv2.MeanStdDev(laplacian, out _, out Scalar stddev);
            double variance = stddev.Val0 * stddev.Val0;

            return variance;
        }
        catch
        {
            return 0.0;
        }
    }
}
