namespace VisionAttributeAI.Models.Tracking;

/// <summary>
/// Categorized intensity state of global camera ego-motion.
/// </summary>
public enum CameraMotionState
{
    Stable,
    ModerateMotion,
    HighMotion,
    Unknown
}

/// <summary>
/// Encapsulates estimated inter-frame camera translation, velocity, and reliability.
/// </summary>
public class CameraMotionResult
{
    /// <summary>
    /// Whether the estimated camera motion vector is mathematically sound and safe to apply.
    /// </summary>
    public bool IsReliable { get; set; }

    /// <summary>
    /// Estimated horizontal camera translation (in full frame pixels) from previous frame to current frame.
    /// A positive DeltaX means scene objects shifted right in pixel space (camera panned left).
    /// </summary>
    public float DeltaX { get; set; }

    /// <summary>
    /// Estimated vertical camera translation (in full frame pixels) from previous frame to current frame.
    /// A positive DeltaY means scene objects shifted down in pixel space (camera tilted up).
    /// </summary>
    public float DeltaY { get; set; }

    /// <summary>
    /// Normalized global camera motion magnitude score [0.0 .. 1.0].
    /// </summary>
    public double MotionScore { get; set; }

    /// <summary>
    /// Discrete categorization of camera motion intensity.
    /// </summary>
    public CameraMotionState MotionState { get; set; }

    /// <summary>
    /// Mathematical confidence of the motion estimation [0.0 .. 1.0].
    /// </summary>
    public float Confidence { get; set; }

    /// <summary>
    /// Informational diagnostics or fallback reasoning.
    /// </summary>
    public string Description { get; set; } = string.Empty;

    public static CameraMotionResult Stable(double motionScore = 0.0, float confidence = 1.0f) => new()
    {
        IsReliable = true,
        DeltaX = 0f,
        DeltaY = 0f,
        MotionScore = motionScore,
        MotionState = CameraMotionState.Stable,
        Confidence = confidence,
        Description = "Stationary / Stable camera position"
    };

    public static CameraMotionResult Motion(float dx, float dy, double motionScore, CameraMotionState state, float confidence, string description = "") => new()
    {
        IsReliable = confidence >= 0.35f,
        DeltaX = dx,
        DeltaY = dy,
        MotionScore = motionScore,
        MotionState = state,
        Confidence = confidence,
        Description = string.IsNullOrEmpty(description) ? $"Camera motion estimated: ΔX={dx:F1}px, ΔY={dy:F1}px ({state})" : description
    };

    public static CameraMotionResult Unreliable(string reason, double motionScore = 0.0) => new()
    {
        IsReliable = false,
        DeltaX = 0f,
        DeltaY = 0f,
        MotionScore = motionScore,
        MotionState = CameraMotionState.Unknown,
        Confidence = 0f,
        Description = $"Unreliable motion estimation: {reason}"
    };
}
