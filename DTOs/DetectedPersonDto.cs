using VisionAttributeAI.DTOs.LiveCamera;

namespace VisionAttributeAI.DTOs;

public class DetectedPersonDto
{
    public int PersonId { get; set; }
    public int PersonIndex
    {
        get => PersonId;
        set => PersonId = value;
    }
    public float DetectionConfidence { get; set; }
    public required BoundingBoxDto BoundingBox { get; set; }
    public int CropWidth { get; set; }
    public int CropHeight { get; set; }
    public string? CropDataUrl { get; set; }

    // Multi-Region & Attribute Fields
    public string AppearanceSex { get; set; } = "Unknown";
    public float AppearanceConfidence { get; set; }
    public string AppearanceState { get; set; } = "SingleImage";

    public string UpperType { get; set; } = "Unknown";
    public float UpperTypeConfidence { get; set; }
    public string UpperColor { get; set; } = "Unknown";
    public float UpperColorConfidence { get; set; }

    public string LowerType { get; set; } = "Not Visible";
    public float LowerTypeConfidence { get; set; }
    public string LowerColor { get; set; } = "Not Visible";
    public float LowerColorConfidence { get; set; }

    public string ShoesType { get; set; } = "Not Visible";
    public float ShoesConfidence { get; set; }

    public bool WatchDetected { get; set; }
    public float WatchConfidence { get; set; }
    public string WatchDetails { get; set; } = "Unknown";

    // Brand Recognition
    public PersonBrandDto? Brands { get; set; }
    public string BrandName { get; set; } = "BrandUnknown";
    public float BrandConfidence { get; set; }
    public string BrandState { get; set; } = "Analyzing";

    public string Orientation { get; set; } = "Uncertain";

    public AttributeQualityDto? QualityScores { get; set; }
    public BodyVisibilityDto? Visibility { get; set; }

    public List<AttributeItemDto> Attributes { get; set; } = [];
}
