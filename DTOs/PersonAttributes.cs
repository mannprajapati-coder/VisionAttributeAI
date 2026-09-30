namespace VisionAttributeAI.DTOs;

/// <summary>
/// Structured DTO containing recognized visual pedestrian attributes from the PAR model.
/// Gender/sex is explicitly categorized as visual appearance estimation rather than verified identity.
/// </summary>
public class PersonAttributes
{
    /// <summary>
    /// Visual appearance estimation ("Female" or "Male") based on model classification.
    /// Note: This is an automated appearance prediction, not verified identity.
    /// </summary>
    public string AppearanceSex { get; set; } = "Unknown";

    /// <summary>
    /// Confidence score (0.0 to 1.0) for the predicted appearance sex.
    /// </summary>
    public float AppearanceSexConfidence { get; set; }

    /// <summary>
    /// Estimated age group ("Age < 18", "Age 18-60", "Age > 60", or null).
    /// </summary>
    public string? AgeGroup { get; set; }

    /// <summary>
    /// Confidence score for the estimated age group.
    /// </summary>
    public float AgeGroupConfidence { get; set; }

    /// <summary>
    /// Estimated pedestrian viewpoint orientation ("Front", "Side", "Back", or null).
    /// </summary>
    public string? Orientation { get; set; }

    /// <summary>
    /// Confidence score for orientation.
    /// </summary>
    public float OrientationConfidence { get; set; }

    /// <summary>
    /// Predicted sleeve type ("ShortSleeve", "LongSleeve", or "Unknown").
    /// </summary>
    public string SleeveType { get; set; } = "Unknown";

    /// <summary>
    /// Confidence score for sleeve type.
    /// </summary>
    public float SleeveTypeConfidence { get; set; }

    /// <summary>
    /// Whether trousers / long pants are detected.
    /// </summary>
    public bool Trousers { get; set; }
    public float TrousersConfidence { get; set; }

    /// <summary>
    /// Whether shorts are detected.
    /// </summary>
    public bool Shorts { get; set; }
    public float ShortsConfidence { get; set; }

    /// <summary>
    /// Whether skirt or dress is detected.
    /// </summary>
    public bool SkirtOrDress { get; set; }
    public float SkirtOrDressConfidence { get; set; }

    /// <summary>
    /// Whether long coat is detected.
    /// </summary>
    public bool LongCoat { get; set; }
    public float LongCoatConfidence { get; set; }

    /// <summary>
    /// Whether a hat or headwear is detected.
    /// </summary>
    public bool HasHat { get; set; }
    public float HatConfidence { get; set; }

    /// <summary>
    /// Whether glasses or eyewear are detected.
    /// </summary>
    public bool HasGlasses { get; set; }
    public float GlassesConfidence { get; set; }

    /// <summary>
    /// Whether a handbag is detected.
    /// </summary>
    public bool HasHandBag { get; set; }
    public float HandBagConfidence { get; set; }

    /// <summary>
    /// Whether a shoulder bag is detected.
    /// </summary>
    public bool HasShoulderBag { get; set; }
    public float ShoulderBagConfidence { get; set; }

    /// <summary>
    /// Whether a backpack is detected.
    /// </summary>
    public bool HasBackpack { get; set; }
    public float BackpackConfidence { get; set; }

    /// <summary>
    /// Whether objects are being held in front of the body.
    /// </summary>
    public bool HoldObjectsInFront { get; set; }
    public float HoldObjectsInFrontConfidence { get; set; }

    /// <summary>
    /// Whether boots are detected.
    /// </summary>
    public bool HasBoots { get; set; }
    public float BootsConfidence { get; set; }

    /// <summary>
    /// List of detected clothing patterns (e.g. "UpperStripe", "UpperLogo", "UpperPlaid", "UpperSplice", "LowerStripe", "LowerPattern").
    /// </summary>
    public List<string> ClothingPatterns { get; set; } = [];

    /// <summary>
    /// All 26 raw attribute probabilities keyed by official label name.
    /// </summary>
    public Dictionary<string, float> RawAttributes { get; set; } = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// Detailed diagnostic information for each of the 26 attribute heads.
    /// </summary>
    public List<AttributeDiagnosticItem> Diagnostics { get; set; } = [];
}

/// <summary>
/// Diagnostic item representing full evaluation metadata for a single model attribute head.
/// </summary>
public class AttributeDiagnosticItem
{
    public int Index { get; set; }
    public string Label { get; set; } = string.Empty;
    public float RawOutput { get; set; }
    public float PostActivation { get; set; }
    public float Threshold { get; set; }
    public string Result { get; set; } = string.Empty;
    public bool IsActive { get; set; }
}
