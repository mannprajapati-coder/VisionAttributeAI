namespace VisionAttributeAI.DTOs.LiveCamera;

/// <summary>
/// Detailed real-time debug diagnostics for developers. Contains NO image bytes or Base64 data.
/// </summary>
public record TrackedPersonDebugDto
{
    public long FrameIndex { get; init; }
    public float YoloConfidence { get; init; }
    public int CropWidth { get; init; }
    public int CropHeight { get; init; }
    public double SharpnessVariance { get; init; }
    public float QualityScore { get; init; }

    // Pose Orientation & Visibility
    public string Orientation { get; init; } = "Uncertain";
    public bool HeadVisible { get; init; }
    public bool UpperVisible { get; init; }
    public bool LowerVisible { get; init; }
    public bool FeetVisible { get; init; }
    public bool LeftWristVisible { get; init; }
    public bool RightWristVisible { get; init; }

    // Quality breakdown
    public float AppearanceQuality { get; init; }
    public float UpperQuality { get; init; }
    public float LowerQuality { get; init; }
    public float ShoesQuality { get; init; }
    public float LeftWristQuality { get; init; }
    public float RightWristQuality { get; init; }

    // Raw Model Classification Scores & Margins
    public float MaleScore { get; init; }
    public float FemaleScore { get; init; }
    public float SexMargin { get; init; }

    public Dictionary<string, float>? UpperTypeRawScores { get; init; }
    public float UpperTypeMargin { get; init; }
    public Dictionary<string, float>? UpperColorRawScores { get; init; }
    public Dictionary<string, float>? LowerTypeRawScores { get; init; }
    public float LowerTypeMargin { get; init; }
    public Dictionary<string, float>? LowerColorRawScores { get; init; }
    public Dictionary<string, float>? ShoesRawScores { get; init; }
    public float ShoesMargin { get; init; }

    // Color distributions
    public string UpperPrimaryColor { get; init; } = string.Empty;
    public float UpperPrimaryPercent { get; init; }
    public string UpperSecondaryColor { get; init; } = string.Empty;
    public float UpperSecondaryPercent { get; init; }

    public string LowerPrimaryColor { get; init; } = string.Empty;
    public float LowerPrimaryPercent { get; init; }
    public string LowerSecondaryColor { get; init; } = string.Empty;
    public float LowerSecondaryPercent { get; init; }

    // Tracking Diagnostics
    public int TrackAgeFrames { get; init; }
    public float LastMatchedIoU { get; init; }
    public float CenterDisplacement { get; init; }
    public bool PossibleIdSwitch { get; init; }
    public string PossibleIdSwitchReason { get; init; } = string.Empty;

    public bool FrameAccepted { get; init; }
    public string RejectionReason { get; init; } = string.Empty;
    public string CurrentAggregatedResult { get; init; } = string.Empty;
}
