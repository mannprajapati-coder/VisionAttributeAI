using OpenCvSharp;
using VisionAttributeAI.Models.AI;
using VisionAttributeAI.Models.Tracking;

namespace VisionAttributeAI.Services.Quality;

/// <summary>
/// Contract for validating image crop quality before sending to visual attribute recognizers.
/// </summary>
public interface IFrameQualityFilter
{
    /// <summary>
    /// Evaluates the visual quality, dimensions, confidence, and blur level of a detected person crop.
    /// </summary>
    TrackQualityResult EvaluateCropQuality(Mat cropMat, BoundingBox box, float detectionConfidence);
}
