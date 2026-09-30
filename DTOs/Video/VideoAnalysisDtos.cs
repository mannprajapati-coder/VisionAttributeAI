using VisionAttributeAI.DTOs.LiveCamera;

namespace VisionAttributeAI.DTOs.Video;

public record VideoAnalysisOptionsDto
{
    public double AnalysisFps { get; init; } = 3.0; // Default 3 frames per second sampling
}

public record VideoSummaryDto
{
    public int TotalUniquePersons { get; init; }
    public double VideoDurationSeconds { get; init; }
    public double SourceFps { get; init; }
    public double AnalysisFps { get; init; }
    public int TotalVideoFrames { get; init; }
    public int SampledFramesAnalyzed { get; init; }
    public double TotalElapsedMs { get; init; }
    public double DetectionMs { get; init; }
    public double AttributeMs { get; init; }
}

public record VideoPersonResultDto
{
    public int PersonId { get; init; }
    public string FirstSeenTimestamp { get; init; } = "00:00.0";
    public string LastSeenTimestamp { get; init; } = "00:00.0";
    public double FirstSeenSeconds { get; init; }
    public double LastSeenSeconds { get; init; }
    public int FramesObserved { get; init; }
    public string TrackStatus { get; init; } = "Finalized"; // "Finalized" or "ActiveUntilEnd"
    public string? BestCropPreviewUrl { get; init; }

    // Final Aggregated Attributes
    public string AppearanceSex { get; init; } = "Unknown";
    public float AppearanceConfidence { get; init; }
    public string AppearanceState { get; init; } = "Analyzing";

    public string UpperType { get; init; } = "Unknown";
    public float UpperTypeConfidence { get; init; }
    public string UpperTypeState { get; init; } = "Analyzing";

    public string UpperColor { get; init; } = "Unknown";
    public float UpperColorConfidence { get; init; }
    public string UpperColorState { get; init; } = "Analyzing";

    public string LowerType { get; init; } = "Not Visible";
    public float LowerTypeConfidence { get; init; }
    public string LowerTypeState { get; init; } = "NotVisible";

    public string LowerColor { get; init; } = "Not Visible";
    public float LowerColorConfidence { get; init; }
    public string LowerColorState { get; init; } = "NotVisible";

    public string ShoesType { get; init; } = "Not Visible";
    public float ShoesConfidence { get; init; }
    public string ShoesState { get; init; } = "NotVisible";

    public bool WatchDetected { get; init; }
    public float WatchConfidence { get; init; }
    public string WatchDetails { get; init; } = "Unknown";

    // 6. Brand Recognition
    public PersonBrandDto? Brands { get; init; }
    public string BrandName { get; init; } = "BrandUnknown";
    public float BrandConfidence { get; init; }
    public string BrandState { get; init; } = "Analyzing";
    public bool IsBrandStable { get; init; }

    // Tracking Diagnostics
    public TrackingDiagnosticsDto TrackingDiagnostics { get; init; } = new();

    // Best-Frame Metadata Pools (Top-5 observations per attribute)
    public List<BestFrameItemDto> BestAppearanceFrames { get; init; } = new();
    public List<BestFrameItemDto> BestUpperFrames { get; init; } = new();
    public List<BestFrameItemDto> BestLowerFrames { get; init; } = new();
    public List<BestFrameItemDto> BestShoesFrames { get; init; } = new();
    public List<BestFrameItemDto> BestWristFrames { get; init; } = new();
}

public record VideoAnalysisResponseDto
{
    public bool Success { get; init; } = true;
    public string? ErrorMessage { get; init; }
    public VideoSummaryDto? Summary { get; init; }
    public List<VideoPersonResultDto> Persons { get; init; } = new();
}
