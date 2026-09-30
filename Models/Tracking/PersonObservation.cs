using VisionAttributeAI.Models.AI;
using VisionAttributeAI.Models.Pose;

namespace VisionAttributeAI.Models.Tracking;

/// <summary>
/// Multi-attribute numerical observation record for a single frame evaluation of a tracked person.
/// Contains NO raw image pixels, byte arrays, or OpenCV Mat references.
/// </summary>
public record PersonObservation
{
    public int PersonId { get; init; }
    public long FrameIndex { get; init; }
    public DateTime TimestampUtc { get; init; } = DateTime.UtcNow;

    // Detection & Global Quality Indicators
    public float YoloConfidence { get; init; }
    public int CropWidth { get; init; }
    public int CropHeight { get; init; }
    public double SharpnessVariance { get; init; }
    public float QualityScore { get; init; }
    public float ComputedWeight { get; init; }

    // Pose, Orientation & Frame Boundary Visibility
    public BodyVisibilityResult Visibility { get; init; } = new();
    public PoseOrientation Orientation => Visibility.Orientation;

    // Independent Attribute-Specific Quality Scores
    public AttributeQualityScores QualityScores { get; init; } = new();

    // 1. Appearance Sex
    public bool HasAppearanceObservation { get; init; }
    public float MaleScore { get; init; }
    public float FemaleScore { get; init; }
    public string SexPrediction { get; init; } = "Unknown";
    public float SexConfidence { get; init; }
    public float SexMargin => Math.Abs(MaleScore - FemaleScore);

    // 2. Upper Clothing (Evaluated on Torso Crop only if UpperBodyVisible)
    public bool HasUpperObservation { get; init; }
    public string UpperType { get; init; } = "Unknown";
    public float UpperTypeConfidence { get; init; }
    public float UpperTypeMargin { get; init; }
    public Dictionary<string, float> UpperTypeScores { get; init; } = new();

    public string UpperColor { get; init; } = "Unknown";
    public float UpperColorConfidence { get; init; }
    public Dictionary<string, float> UpperColorScores { get; init; } = new();
    public ColorClassificationResult? UpperColorDetails { get; init; }

    // 3. Lower Clothing (Evaluated on Lower Crop only if LowerBodyVisible)
    public bool HasLowerObservation { get; init; }
    public string LowerType { get; init; } = "Not Visible";
    public float LowerTypeConfidence { get; init; }
    public float LowerTypeMargin { get; init; }
    public Dictionary<string, float> LowerTypeScores { get; init; } = new();

    public string LowerColor { get; init; } = "Not Visible";
    public float LowerColorConfidence { get; init; }
    public Dictionary<string, float> LowerColorScores { get; init; } = new();
    public ColorClassificationResult? LowerColorDetails { get; init; }

    // 4. Shoes (Evaluated on Feet Crop only if FeetVisible)
    public bool HasShoesObservation { get; init; }
    public string ShoesType { get; init; } = "Not Visible";
    public float ShoesConfidence { get; init; }
    public float ShoesMargin { get; init; }
    public Dictionary<string, float> ShoesScores { get; init; } = new();

    // 5. Watch (Evaluated on Wrist Crops only if WristsVisible & passed quality gate)
    public bool HasWatchObservation { get; init; }
    public string WatchStatus { get; init; } = "NotVisible";
    public float WatchConfidence { get; init; }
    public string WatchDetails { get; init; } = string.Empty;
    public bool LeftWristDetected { get; init; }
    public bool RightWristDetected { get; init; }

    // 6. Brand & Logo Evidence
    public bool HasBrandObservation => Brands?.HasAnyAcceptedBrand ?? false;
    public VisionAttributeAI.Models.Brand.PersonBrandResult? Brands { get; init; }
}
