namespace VisionAttributeAI.Models.Brand;

/// <summary>
/// Configuration options for the Brand & Logo Recognition pipeline.
/// </summary>
public class BrandOptions
{
    public const string SectionName = "VisionAi:BrandRecognition";

    public bool Enabled { get; set; } = true;
    public string LogosRetinaNetModelPath { get; set; } = "models/brand/logos_retinanet.onnx";
    public string DinoV2ModelPath { get; set; } = "models/brand/dinov2_vits14.onnx";
    public string GalleryPath { get; set; } = "models/brand/gallery";
    public string GalleryCachePath { get; set; } = "models/brand/gallery_embeddings.bin";

    public int MinPersonHeight { get; set; } = 180;
    public int MinRegionDimension { get; set; } = 32;
    public int MinLogoCropSize { get; set; } = 24;

    public float DetectionConfidenceThreshold { get; set; } = 0.20f;
    public float DetectionNmsThreshold { get; set; } = 0.40f;

    public float MinCosineSimilarity { get; set; } = 0.70f;
    public float MinRunnerUpMargin { get; set; } = 0.04f;

    public int SampleIntervalFrames { get; set; } = 12;
    public int RequiredStableObservations { get; set; } = 2;

    public static readonly string[] SupportedBrands =
    [
        "Nike",
        "Adidas",
        "Puma",
        "Under Armour",
        "New Balance",
        "Gucci",
        "Louis Vuitton"
    ];
}
