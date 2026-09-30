namespace VisionAttributeAI.Models.Tracking;

/// <summary>
/// Independent attribute-specific visual quality scores normalized to [0.0 .. 1.0].
/// Evaluated from keypoint confidences, resolution, sharpness, and frame boundary clipping.
/// </summary>
public record AttributeQualityScores
{
    public float AppearanceQuality { get; init; }
    public float UpperQuality { get; init; }
    public float LowerQuality { get; init; }
    public float ShoesQuality { get; init; }
    public float LeftWristQuality { get; init; }
    public float RightWristQuality { get; init; }

    // Diagnostic measurements
    public float FeetWidth { get; init; }
    public float FeetHeight { get; init; }
    public double FeetSharpness { get; init; }
    public bool FeetGatePassed { get; init; }

    public float LeftWristWidth { get; init; }
    public float LeftWristHeight { get; init; }
    public double LeftWristSharpness { get; init; }
    public bool LeftWristGatePassed { get; init; }

    public float RightWristWidth { get; init; }
    public float RightWristHeight { get; init; }
    public double RightWristSharpness { get; init; }
    public bool RightWristGatePassed { get; init; }

    // Visual Evidence Gating Flags
    public bool HasAppearanceEvidence { get; init; }
    public bool HasUpperEvidence { get; init; }
    public bool HasLowerEvidence { get; init; }
    public bool HasShoesEvidence { get; init; }
    public bool HasWristEvidence { get; init; }

    public string AppearanceRejectionReason { get; init; } = string.Empty;
    public string UpperRejectionReason { get; init; } = string.Empty;
    public string LowerRejectionReason { get; init; } = string.Empty;
    public string ShoesRejectionReason { get; init; } = string.Empty;
    public string WristRejectionReason { get; init; } = string.Empty;
}
