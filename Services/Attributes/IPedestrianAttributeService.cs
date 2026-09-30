using OpenCvSharp;
using VisionAttributeAI.DTOs;

namespace VisionAttributeAI.Services.Attributes;

/// <summary>
/// Service contract for Pedestrian Attribute Recognition (PAR) inference.
/// Processes a single cropped person image through a pretrained PAR ONNX model.
/// </summary>
public interface IPedestrianAttributeService : IDisposable
{
    /// <summary>
    /// Indicates whether the ONNX inference session is initialized and ready.
    /// </summary>
    bool IsModelLoaded { get; }

    /// <summary>
    /// Discovered ONNX input tensor name.
    /// </summary>
    string InputName { get; }

    /// <summary>
    /// Discovered ONNX output tensor name.
    /// </summary>
    string OutputName { get; }

    /// <summary>
    /// Input dimensions expected by the ONNX model [Batch, Channels, Height, Width].
    /// </summary>
    int[] InputDimensions { get; }

    /// <summary>
    /// Data type of the input tensor elements.
    /// </summary>
    Type InputElementType { get; }

    /// <summary>
    /// Data type of the output tensor elements.
    /// </summary>
    Type OutputElementType { get; }

    /// <summary>
    /// Recognizes pedestrian attributes from an OpenCV Mat image containing a cropped person.
    /// </summary>
    Task<PersonAttributes> RecognizeAttributesAsync(Mat personCrop, CancellationToken cancellationToken = default);

    /// <summary>
    /// Recognizes pedestrian attributes from raw encoded image bytes of a cropped person.
    /// </summary>
    Task<PersonAttributes> RecognizeAttributesAsync(byte[] imageBytes, CancellationToken cancellationToken = default);
}
