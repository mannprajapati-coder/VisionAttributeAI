namespace VisionAttributeAI.DTOs.LiveCamera;

/// <summary>
/// Data transfer object for a tracked person in the live video stream with complete multi-attribute breakdown,
/// quality scores, states, consensus ratios, and best-frame pools.
/// </summary>
public record TrackedPersonDto
{
    public int PersonId { get; init; }
    public int DisplayId { get; set; } = 1;
    public BoundingBoxDto BoundingBox { get; init; } = new(0, 0, 0, 0);
    public float DetectionConfidence { get; init; }

    // Pose Keypoints, Orientation & Body Regions
    public BodyVisibilityDto Visibility { get; init; } = new();
    public BodyRegionsDto? Regions { get; init; }
    public List<KeypointDto> Keypoints { get; init; } = new();

    // Independent Attribute Quality Scores
    public AttributeQualityDto QualityScores { get; init; } = new();

    // Tracking Diagnostics
    public TrackingDiagnosticsDto TrackingDiagnostics { get; init; } = new();

    // 1. Appearance Sex
    public string AppearanceSex { get; init; } = "Analyzing...";
    public float AppearanceConfidence { get; init; }
    public string AppearanceState { get; init; } = "Analyzing";
    public float AppearanceConsensus { get; init; }
    public bool IsAppearanceStable { get; init; }

    // 2. Upper Clothing
    public string UpperType { get; init; } = "Unknown";
    public float UpperTypeConfidence { get; init; }
    public string UpperTypeState { get; init; } = "Analyzing";
    public float UpperTypeConsensus { get; init; }
    public string UpperColor { get; init; } = "Unknown";
    public float UpperColorConfidence { get; init; }
    public string UpperColorState { get; init; } = "Analyzing";
    public float UpperColorConsensus { get; init; }
    public bool IsUpperStable { get; init; }

    // 3. Lower Clothing
    public string LowerType { get; init; } = "Not Visible";
    public float LowerTypeConfidence { get; init; }
    public string LowerTypeState { get; init; } = "NotVisible";
    public float LowerTypeConsensus { get; init; }
    public string LowerColor { get; init; } = "Not Visible";
    public float LowerColorConfidence { get; init; }
    public string LowerColorState { get; init; } = "NotVisible";
    public float LowerColorConsensus { get; init; }
    public bool IsLowerStable { get; init; }

    // 4. Shoes
    public string ShoesType { get; init; } = "Not Visible";
    public float ShoesConfidence { get; init; }
    public string ShoesState { get; init; } = "NotVisible";
    public float ShoesConsensus { get; init; }
    public bool IsShoesStable { get; init; }

    // 5. Watch
    public string WatchStatus { get; init; } = "Unknown";
    public float WatchConfidence { get; init; }
    public string WatchState { get; init; } = "Uncertain";

    // 6. Brand Recognition
    public PersonBrandDto? Brands { get; init; }
    public string BrandName { get; init; } = "BrandUnknown";
    public float BrandConfidence { get; init; }
    public string BrandState { get; init; } = "Analyzing";
    public bool IsBrandStable { get; init; }

    // Best-Frame Pools
    public BestFramePoolsDto BestFrames { get; init; } = new();

    // Tracking & Stability Indicators
    public int ValidFrameCount { get; init; }
    public int MinFramesRequired { get; init; } = 5;
    public bool IsFullyStable { get; init; }
    public TrackedPersonDebugDto? DebugInfo { get; init; }
}
