namespace VisionAttributeAI.Models.Tracking;

public enum AttributeState
{
    Stable,
    Analyzing,
    Uncertain,
    NotVisible,
    InsufficientVisualEvidence
}

/// <summary>
/// Metadata record for a Top-K best quality observation frame for an individual attribute.
/// Stores ZERO raw image pixels/buffers to preserve privacy and memory.
/// </summary>
public record BestFrameRecord
{
    public long FrameIndex { get; init; }
    public float QualityScore { get; init; }
    public string Prediction { get; init; } = string.Empty;
    public float Confidence { get; init; }
    public DateTime TimestampUtc { get; init; } = DateTime.UtcNow;
    public string Details { get; init; } = string.Empty;
}
