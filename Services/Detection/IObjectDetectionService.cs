using OpenCvSharp;
using VisionAttributeAI.Models.AI;

namespace VisionAttributeAI.Services.Detection;

/// <summary>
/// Service contract for object and person detection in images.
/// </summary>
public interface IObjectDetectionService : IDisposable
{
    /// <summary>
    /// Gets whether the underlying ONNX model session is loaded and ready.
    /// </summary>
    bool IsModelLoaded { get; }

    /// <summary>
    /// Model input tensor name inferred from ONNX metadata.
    /// </summary>
    string InputName { get; }

    /// <summary>
    /// Model output tensor name inferred from ONNX metadata.
    /// </summary>
    string OutputName { get; }

    /// <summary>
    /// Input dimensions expected by the ONNX model.
    /// </summary>
    int[] InputDimensions { get; }

    /// <summary>
    /// Detects persons within the given OpenCvSharp Mat image.
    /// Handles letterbox resizing, normalization, ONNX inference, output parsing, coordinate mapping, and NMS.
    /// </summary>
    Task<IReadOnlyList<DetectionResult>> DetectPersonsAsync(Mat image, CancellationToken cancellationToken = default);
}
