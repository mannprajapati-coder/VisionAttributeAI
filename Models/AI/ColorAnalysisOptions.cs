namespace VisionAttributeAI.Models.AI;

/// <summary>
/// Centralized, configurable parameters for Clothing Color Recognition,
/// CIELAB & HSV thresholds, K-Means clustering, and evidence gating.
/// </summary>
public class ColorAnalysisOptions
{
    public const string SectionName = "ColorAnalysis";

    /// <summary>
    /// Comprehensive Clothing Color Taxonomy supported by the classifier.
    /// </summary>
    public string[] SupportedColors { get; set; } =
    [
        "Black",
        "White",
        "Grey",
        "Red",
        "Maroon",
        "Orange",
        "Yellow",
        "Green",
        "Blue",
        "Navy",
        "Cyan",
        "Purple",
        "Pink",
        "Brown",
        "Beige"
    ];

    /// <summary>
    /// Minimum count of valid clothing pixels after pose masking and skin removal.
    /// Below this threshold, returns InsufficientVisualEvidence.
    /// </summary>
    public int MinValidPixelCount { get; set; } = 35;

    /// <summary>
    /// Minimum percentage of the ROI area that must contain valid clothing pixels.
    /// </summary>
    public float MinValidPixelAreaPercentage { get; set; } = 0.08f;

    /// <summary>
    /// Upper bound on CIELAB Chroma for Achromatic Grey.
    /// </summary>
    public float AchromaticMaxChroma { get; set; } = 5.2f;

    /// <summary>
    /// Maximum CIELAB Lightness (L*) for Black in Achromatic mode.
    /// </summary>
    public float BlackMaxLightness { get; set; } = 24.0f;

    /// <summary>
    /// Minimum CIELAB Lightness (L*) for White in Achromatic mode.
    /// </summary>
    public float WhiteMinLightness { get; set; } = 65.0f;

    /// <summary>
    /// Number of clusters for K-Means color decomposition.
    /// </summary>
    public int KMeansClusters { get; set; } = 3;

    /// <summary>
    /// Minimum cluster share (fraction of valid pixels) to be considered significant.
    /// Clusters below this are discarded as specular highlights, seams, or noise.
    /// </summary>
    public float MinClusterShare { get; set; } = 0.08f;

    /// <summary>
    /// Minimum primary cluster share required for high single-color confidence.
    /// If primary share is below this and secondary is above SecondaryColorMinShare, garment is flagged IsMultiColor.
    /// </summary>
    public float MultiColorPrimaryThreshold { get; set; } = 0.65f;

    /// <summary>
    /// Minimum secondary cluster share to report a SecondaryColor.
    /// </summary>
    public float SecondaryColorMinShare { get; set; } = 0.20f;

    /// <summary>
    /// Minimum primary color vote share to return a specific color vs "Unknown".
    /// </summary>
    public float MinPrimaryVoteShare { get; set; } = 0.20f;

    /// <summary>
    /// Enables Pose-guided polygon masking for torso and leg columns.
    /// </summary>
    public bool EnablePoseGuidedMasking { get; set; } = true;

    /// <summary>
    /// Enables skin tone subtraction. Default is false to prevent fabric deletion.
    /// </summary>
    public bool EnableSkinExclusion { get; set; } = false;

    /// <summary>
    /// Enables Gray-World color constancy normalization under severe color casts.
    /// </summary>
    public bool EnableColorConstancy { get; set; } = false;
}
