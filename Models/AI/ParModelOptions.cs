namespace VisionAttributeAI.Models.AI;

/// <summary>
/// Fine-grained confidence threshold configuration for individual attribute groups.
/// </summary>
public class ParThresholdOptions
{
    /// <summary>
    /// Minimum confidence threshold for binary appearance sex classification (e.g. 0.60).
    /// Below this, appearance sex is marked as "Unknown".
    /// </summary>
    public float AppearanceSex { get; set; } = 0.60f;

    /// <summary>
    /// Minimum confidence threshold for age group classification (e.g. 0.50).
    /// </summary>
    public float AgeGroup { get; set; } = 0.50f;

    /// <summary>
    /// Minimum confidence threshold for viewpoint orientation classification (e.g. 0.50).
    /// </summary>
    public float Orientation { get; set; } = 0.50f;

    /// <summary>
    /// Minimum confidence threshold for sleeve length classification (e.g. 0.50).
    /// </summary>
    public float SleeveType { get; set; } = 0.50f;

    /// <summary>
    /// Minimum confidence threshold for lower body attire (trousers, shorts, skirt, long coat) (e.g. 0.50).
    /// </summary>
    public float LowerAttire { get; set; } = 0.50f;

    /// <summary>
    /// Minimum confidence threshold for hat / headwear detection (e.g. 0.50).
    /// </summary>
    public float Hat { get; set; } = 0.50f;

    /// <summary>
    /// Minimum confidence threshold for eyewear / glasses detection (e.g. 0.40).
    /// </summary>
    public float Glasses { get; set; } = 0.40f;

    /// <summary>
    /// Minimum confidence threshold for bags / carried objects (handbag, shoulder bag, backpack) (e.g. 0.50).
    /// </summary>
    public float Bags { get; set; } = 0.50f;

    /// <summary>
    /// Minimum confidence threshold for boots detection (e.g. 0.50).
    /// </summary>
    public float Boots { get; set; } = 0.50f;

    /// <summary>
    /// Minimum confidence threshold for clothing patterns (stripes, logo, plaid, splice) (e.g. 0.50).
    /// </summary>
    public float ClothingPatterns { get; set; } = 0.50f;
}

/// <summary>
/// Configuration options for Pedestrian Attribute Recognition (PAR) ONNX model.
/// </summary>
public class ParModelOptions
{
    public const string SectionName = "VisionAi:Par";

    /// <summary>
    /// Path to the PAR ONNX model file (e.g. models/par/pulc_person_attribute.onnx).
    /// </summary>
    public string ModelPath { get; set; } = "models/par/pulc_person_attribute.onnx";

    /// <summary>
    /// Path to the Fashion-CLIP vision ONNX model file (e.g. models/par/fashion_clip_vision.onnx).
    /// </summary>
    public string ClipModelPath { get; set; } = "models/par/fashion_clip_vision.onnx";

    /// <summary>
    /// Whether to use Fashion-CLIP for zero-shot appearance sex classification.
    /// </summary>
    public bool UseClipForAppearanceSex { get; set; } = true;

    /// <summary>
    /// Input width expected by the PAR model (192 for PULC).
    /// </summary>
    public int InputWidth { get; set; } = 192;

    /// <summary>
    /// Input height expected by the PAR model (256 for PULC).
    /// </summary>
    public int InputHeight { get; set; } = 256;

    /// <summary>
    /// Default fallback threshold for binary attributes.
    /// </summary>
    public float ConfidenceThreshold { get; set; } = 0.50f;

    /// <summary>
    /// Group-specific and attribute-specific confidence thresholds.
    /// </summary>
    public ParThresholdOptions Thresholds { get; set; } = new();

    /// <summary>
    /// GPU Device ID if using DirectML/CUDA, or -1 for CPU.
    /// </summary>
    public int GpuDeviceId { get; set; } = -1;

    /// <summary>
    /// Attribute list defined by the PULC model checkpoint (26 heads in exact export order).
    /// </summary>
    public List<string> AttributeLabels { get; set; } =
    [
        "Hat",                // 0
        "Glasses",            // 1
        "ShortSleeve",        // 2
        "LongSleeve",         // 3
        "UpperStripe",        // 4
        "UpperLogo",          // 5
        "UpperPlaid",         // 6
        "UpperSplice",        // 7
        "LowerStripe",        // 8
        "LowerPattern",       // 9
        "LongCoat",           // 10
        "Trousers",           // 11
        "Shorts",             // 12
        "Skirt&Dress",        // 13
        "Boots",              // 14
        "HandBag",            // 15
        "ShoulderBag",        // 16
        "Backpack",           // 17
        "HoldObjectsInFront", // 18
        "AgeOver60",          // 19
        "Age18-60",           // 20
        "AgeLess18",          // 21
        "Female",             // 22
        "Front",              // 23
        "Side",               // 24
        "Back"                // 25
    ];
}
