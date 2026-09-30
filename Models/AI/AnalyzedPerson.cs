namespace VisionAttributeAI.Models.AI;

/// <summary>
/// Combines person identification, detection bounding box, cropped image preview, and recognized attributes.
/// </summary>
public class AnalyzedPerson
{
    public int PersonId { get; init; }
    public int PersonIndex => PersonId;
    public required DetectionResult Detection { get; init; }
    public string? CropBase64 { get; init; }
    public int CropWidth { get; init; }
    public int CropHeight { get; init; }
    public required PersonAttributeResult Attributes { get; init; }
}
