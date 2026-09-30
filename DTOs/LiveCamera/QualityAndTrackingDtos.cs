namespace VisionAttributeAI.DTOs.LiveCamera;

public record AttributeQualityDto
{
    public float Appearance { get; init; }
    public float Upper { get; init; }
    public float Lower { get; init; }
    public float Shoes { get; init; }
    public float LeftWrist { get; init; }
    public float RightWrist { get; init; }

    public bool HasAppearanceEvidence { get; init; }
    public bool HasUpperEvidence { get; init; }
    public bool HasLowerEvidence { get; init; }
    public bool HasShoesEvidence { get; init; }
    public bool HasWristEvidence { get; init; }

    public string AppearanceRejection { get; init; } = string.Empty;
    public string UpperRejection { get; init; } = string.Empty;
    public string LowerRejection { get; init; } = string.Empty;
    public string ShoesRejection { get; init; } = string.Empty;
    public string WristRejection { get; init; } = string.Empty;
}

public record TrackingDiagnosticsDto
{
    public int PersonId { get; init; }
    public int TrackAgeFrames { get; init; }
    public float LastMatchedIoU { get; init; }
    public float CenterDisplacement { get; init; }
    public int MissedFrames { get; init; }
    public bool PossibleIdSwitch { get; init; }
    public string PossibleIdSwitchReason { get; init; } = string.Empty;
}

public record BestFrameItemDto
{
    public long FrameIndex { get; init; }
    public float QualityScore { get; init; }
    public string Prediction { get; init; } = string.Empty;
    public float Confidence { get; init; }
    public string Details { get; init; } = string.Empty;
}

public record BestFramePoolsDto
{
    public List<BestFrameItemDto> Appearance { get; init; } = new();
    public List<BestFrameItemDto> Upper { get; init; } = new();
    public List<BestFrameItemDto> Lower { get; init; } = new();
    public List<BestFrameItemDto> Shoes { get; init; } = new();
    public List<BestFrameItemDto> Wrist { get; init; } = new();
}
