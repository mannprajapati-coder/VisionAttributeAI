using OpenCvSharp;
using VisionAttributeAI.Models.Tracking;

namespace VisionAttributeAI.Services.Tracking;

/// <summary>
/// Service contract for estimating inter-frame global camera ego-motion (pan/tilt/zoom/translation).
/// </summary>
public interface ICameraMotionEstimator
{
    /// <summary>
    /// Estimates global camera motion translation vector between two consecutive full frames.
    /// </summary>
    /// <param name="prevFrame">Previous frame Mat (BGR or Grayscale)</param>
    /// <param name="currFrame">Current frame Mat (BGR or Grayscale)</param>
    /// <returns>CameraMotionResult containing DeltaX, DeltaY, MotionScore, and Reliability</returns>
    CameraMotionResult EstimateMotion(Mat prevFrame, Mat currFrame);

    /// <summary>
    /// Resets any temporal smoothing state or cached keyframes.
    /// </summary>
    void Reset();
}
