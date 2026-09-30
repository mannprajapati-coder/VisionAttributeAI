namespace VisionAttributeAI.Models.AI;

/// <summary>
/// Centralized configuration options for the Person Analysis, Live Camera Quality Gating, and Multi-Frame Aggregation Pipeline.
/// </summary>
public class PersonAnalysisOptions
{
    public const string SectionName = "VisionAi:PersonAnalysis";

    // Camera Frame Ingestion
    public double CameraProcessingFps { get; set; } = 2.0;

    // Detection & Global Quality Gates
    public float MinPersonConfidence { get; set; } = 0.40f;
    public int MinCropWidth { get; set; } = 40;
    public int MinCropHeight { get; set; } = 80;
    public int MinCropArea { get; set; } = 3200;
    public double MinSharpness { get; set; } = 35.0;
    public float MinKeypointConfidence { get; set; } = 0.35f;

    // Attribute-Specific Minimum Resolution & Quality Gates (0 = auto)
    public int MinHeadWidth { get; set; } = 40;
    public int MinHeadHeight { get; set; } = 40;
    public int MinTorsoWidth { get; set; } = 50;
    public int MinTorsoHeight { get; set; } = 55;
    public int MinLowerWidth { get; set; } = 45;
    public int MinLowerHeight { get; set; } = 55;
    
    // Shoe / Feet Quality Gate
    public int MinFeetWidth { get; set; } = 40;
    public int MinFeetHeight { get; set; } = 25;
    public double MinFeetSharpness { get; set; } = 15.0;
    public float MinFeetPoseConfidence { get; set; } = 0.30f;

    // Watch / Wrist Quality Gate & Object Detection
    public int WatchMinRoiWidth { get; set; } = 35;
    public int WatchMinRoiHeight { get; set; } = 35;
    public double WatchMinSharpness { get; set; } = 20.0;
    public float WatchMinPoseConfidence { get; set; } = 0.35f;
    public float WatchDetectionConfidence { get; set; } = 0.50f;
    public int WatchRequiredStableObservations { get; set; } = 3;
    public int WatchNegativeStableObservations { get; set; } = 5;

    // Temporal Aggregation & Stability
    public int MinimumValidFrames { get; set; } = 5;
    public int ObservationWindowSize { get; set; } = 10;
    public float StableConfidenceThreshold { get; set; } = 0.60f;
    public float UnknownMarginThreshold { get; set; } = 0.12f;
    public float MinimumConsensusRatio { get; set; } = 0.65f; // Requires >= 65% agreement for Stable state
    public int HysteresisContradictionFrames { get; set; } = 3; // Requires >= 3 consecutive contradictory observations to flip a stable state
    public int TopKBestFrames { get; set; } = 5;

    // Color Confidence & Multi-Color Thresholds
    public float PrimaryColorDominanceThreshold { get; set; } = 0.45f;
    public float SecondaryColorDominanceThreshold { get; set; } = 0.20f;

    // Tracking Diagnostics
    public double TrackExpirationSeconds { get; set; } = 5.0;
    public int MaxMissedFrames { get; set; } = 10;
    public float MatchIouThreshold { get; set; } = 0.30f;
    public float PossibleIdSwitchIouThreshold { get; set; } = 0.20f;
    public float MaxCenterDisplacementRatio { get; set; } = 0.45f;

    // Model Paths
    public string YoloModelPath { get; set; } = "models/yolo/yolov5n.onnx";
    public string PoseModelPath { get; set; } = "models/pose/yolov8n-pose.onnx";
    public string ClipVisionModelPath { get; set; } = "models/par/fashion_clip_vision.onnx";
    public string ClipTextModelPath { get; set; } = "models/par/fashion_clip_text.onnx";
    public string PromptCachePath { get; set; } = "models/par/prompt_embeddings.bin";
    public string WatchModelPath { get; set; } = "models/watch/yolov8n-watch.onnx";
}
