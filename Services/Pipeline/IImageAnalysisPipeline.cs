using OpenCvSharp;
using VisionAttributeAI.Models.AI;

namespace VisionAttributeAI.Services.Pipeline;

/// <summary>
/// Common image-analysis pipeline for Person Attribute Recognition.
/// Shared by both uploaded image endpoints and future live camera stream processing.
/// </summary>
public interface IImageAnalysisPipeline
{
    /// <summary>
    /// Analyzes an image already loaded as an OpenCvSharp Mat (ideal for camera frames).
    /// </summary>
    Task<ImageAnalysisResult> AnalyzeAsync(Mat image, CancellationToken cancellationToken = default);

    /// <summary>
    /// Decodes an image from byte array and runs the analysis pipeline.
    /// </summary>
    Task<ImageAnalysisResult> AnalyzeAsync(byte[] imageBytes, CancellationToken cancellationToken = default);

    /// <summary>
    /// Decodes an image from stream and runs the analysis pipeline (ideal for uploaded files).
    /// </summary>
    Task<ImageAnalysisResult> AnalyzeAsync(Stream imageStream, CancellationToken cancellationToken = default);
}
