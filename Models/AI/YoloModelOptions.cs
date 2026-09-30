namespace VisionAttributeAI.Models.AI;

/// <summary>
/// Configuration options for YOLO detection ONNX model.
/// </summary>
public class YoloModelOptions
{
    public const string SectionName = "VisionAi:Yolo";

    /// <summary>
    /// Path to the YOLO ONNX model file.
    /// </summary>
    public string ModelPath { get; set; } = "models/yolo/yolov8n.onnx";

    /// <summary>
    /// Input width expected by YOLO (default: 640).
    /// </summary>
    public int InputWidth { get; set; } = 640;

    /// <summary>
    /// Input height expected by YOLO (default: 640).
    /// </summary>
    public int InputHeight { get; set; } = 640;

    /// <summary>
    /// Minimum confidence threshold to consider a detection valid.
    /// </summary>
    public float ConfidenceThreshold { get; set; } = 0.45f;

    /// <summary>
    /// Non-Maximum Suppression (IoU) threshold.
    /// </summary>
    public float NmsThreshold { get; set; } = 0.45f;

    /// <summary>
    /// Target class ID for person (COCO person class is 0).
    /// </summary>
    public int PersonClassId { get; set; } = 0;

    /// <summary>
    /// GPU Device ID if using DirectML/CUDA, or -1 for CPU.
    /// </summary>
    public int GpuDeviceId { get; set; } = -1;
}
