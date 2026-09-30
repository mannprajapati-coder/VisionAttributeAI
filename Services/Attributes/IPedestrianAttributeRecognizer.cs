using OpenCvSharp;
using VisionAttributeAI.Models.AI;

namespace VisionAttributeAI.Services.Attributes;

/// <summary>
/// Service contract for Pedestrian Attribute Recognition (PAR) on cropped person images.
/// </summary>
public interface IPedestrianAttributeRecognizer : IDisposable
{
    /// <summary>
    /// Checks if the underlying PAR model session is loaded and ready.
    /// </summary>
    bool IsModelLoaded { get; }

    /// <summary>
    /// Recognizes pedestrian visual attributes (e.g., gender, age group, upper/lower clothing, accessories)
    /// from a cropped image of a detected person.
    /// </summary>
    Task<PersonAttributeResult> RecognizeAttributesAsync(Mat personCrop, CancellationToken cancellationToken = default);
}
