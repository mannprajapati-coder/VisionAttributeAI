using OpenCvSharp;

namespace VisionAttributeAI.Services.Attributes;

/// <summary>
/// Output of zero-shot classification over a controlled dictionary of candidate categories.
/// </summary>
public record ClipClassificationResult
{
    public string TopCategory { get; init; } = "Unknown";
    public float TopConfidence { get; init; }
    public Dictionary<string, float> CategoryScores { get; init; } = new();
    public float Margin { get; init; }
}

/// <summary>
/// Contract for multi-attribute zero-shot visual classification using Fashion-CLIP.
/// </summary>
public interface IFashionClipService : IDisposable
{
    bool IsModelLoaded { get; }

    /// <summary>
    /// Classifies appearance sex ("Male", "Female") on head/upper crop.
    /// </summary>
    Task<ClipClassificationResult> ClassifySexAsync(Mat crop, CancellationToken cancellationToken = default);

    /// <summary>
    /// Classifies upper clothing type on upper torso crop.
    /// </summary>
    Task<ClipClassificationResult> ClassifyUpperClothingTypeAsync(Mat torsoCrop, CancellationToken cancellationToken = default);

    /// <summary>
    /// Classifies upper clothing color on upper torso crop.
    /// </summary>
    Task<ClipClassificationResult> ClassifyUpperClothingColorAsync(Mat torsoCrop, CancellationToken cancellationToken = default);

    /// <summary>
    /// Classifies lower clothing type on lower body crop.
    /// </summary>
    Task<ClipClassificationResult> ClassifyLowerClothingTypeAsync(Mat lowerCrop, CancellationToken cancellationToken = default);

    /// <summary>
    /// Classifies lower clothing color on lower body crop.
    /// </summary>
    Task<ClipClassificationResult> ClassifyLowerClothingColorAsync(Mat lowerCrop, CancellationToken cancellationToken = default);

    /// <summary>
    /// Classifies footwear type on feet crop.
    /// </summary>
    Task<ClipClassificationResult> ClassifyShoesTypeAsync(Mat feetCrop, CancellationToken cancellationToken = default);
}
