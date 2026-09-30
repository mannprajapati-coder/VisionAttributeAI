namespace VisionAttributeAI.DTOs.LiveCamera;

/// <summary>
/// Response payload for a processed live camera frame.
/// </summary>
public record LiveFrameResponseDto
{
    public bool Success { get; init; } = true;
    public string? ErrorMessage { get; init; }
    public long FrameIndex { get; init; }
    public int ImageWidth { get; init; }
    public int ImageHeight { get; init; }
    public double TotalElapsedMs { get; init; }
    public double DetectionMs { get; init; }
    public double AttributeMs { get; init; }
    public int ActivePersonCount => Persons.Count;
    public List<TrackedPersonDto> Persons { get; init; } = new();
}
