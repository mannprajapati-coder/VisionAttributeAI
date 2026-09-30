namespace VisionAttributeAI.Models.Tracking;

/// <summary>
/// Configuration options for real-time live camera tracking, quality filtering, and temporal aggregation.
/// </summary>
public class LiveTrackingOptions
{
    public const string SectionName = "VisionAi:LiveTracking";

    /// <summary>
    /// Desired camera frame processing rate in frames per second (default: 2.0).
    /// </summary>
    public double TargetFps { get; set; } = 2.0;

    /// <summary>
    /// Maximum number of consecutive missed frames before a track is considered lost in live camera (default: 10 frames / ~5s at 2fps).
    /// </summary>
    public int MaxMissedFrames { get; set; } = 10;

    /// <summary>
    /// Maximum number of consecutive missed frames before a track is retired in offline video analysis (default: 4 frames / ~1.3s at 3fps).
    /// Prevents ghost tracks from remaining alive long after a person exits.
    /// </summary>
    public int VideoMaxMissedFrames { get; set; } = 4;

    /// <summary>
    /// Track expiration timeout in seconds after which all observations for the track are cleared (default: 5.0 seconds).
    /// </summary>
    public double TrackTimeoutSeconds { get; set; } = 5.0;

    /// <summary>
    /// Minimum IoU overlap threshold to associate a new detection with an existing track (default: 0.25).
    /// </summary>
    public float MatchIouThreshold { get; set; } = 0.25f;

    /// <summary>
    /// Maximum allowed center displacement normalized by previous bounding box height (default: 0.35).
    /// Prevents distant detections from being matched to a previous track.
    /// </summary>
    public float MaxAllowedCenterDisplacementRatio { get; set; } = 0.35f;

    /// <summary>
    /// Maximum allowed ratio between larger and smaller bounding box area (default: 2.2).
    /// Prevents drastic scale jumps from corrupting tracks.
    /// </summary>
    public float MaxAllowedAreaChangeRatio { get; set; } = 2.2f;

    /// <summary>
    /// Maximum missed frame age that permits proximity fallback matching (default: 2).
    /// Tracks missing for more than 2 frames require direct IoU overlap.
    /// </summary>
    public int MaxMissedFramesForProximity { get; set; } = 2;

    /// <summary>
    /// Minimum YOLO detection confidence required for a bounding box to be tracked (default: 0.40).
    /// </summary>
    public float MinDetectionConfidence { get; set; } = 0.40f;

    /// <summary>
    /// Minimum crop width in pixels to pass quality filtering (default: 40 px).
    /// </summary>
    public int MinCropWidth { get; set; } = 40;

    /// <summary>
    /// Minimum crop height in pixels to pass quality filtering (default: 80 px).
    /// </summary>
    public int MinCropHeight { get; set; } = 80;

    /// <summary>
    /// Minimum crop bounding box area in pixels to pass quality filtering (default: 3200 px²).
    /// </summary>
    public int MinCropArea { get; set; } = 3200;

    /// <summary>
    /// Minimum OpenCV Laplacian variance threshold for sharpness / blur detection (default: 35.0).
    /// Values below this are flagged as blurry and rejected.
    /// </summary>
    public double MinSharpnessVariance { get; set; } = 35.0;

    /// <summary>
    /// Maximum rolling observation history size per PersonId (default: 10).
    /// </summary>
    public int HistoryWindowSize { get; set; } = 10;

    /// <summary>
    /// Minimum valid observations required before declaring a stable prediction (default: 5).
    /// </summary>
    public int MinValidObservations { get; set; } = 5;

    /// <summary>
    /// Minimum confidence margin between Male and Female aggregated score to prevent ambiguous classification (default: 0.15).
    /// If |Male - Female| is below this threshold, status is marked as "Unknown".
    /// </summary>
    public float AmbiguityMarginThreshold { get; set; } = 0.15f;
}
