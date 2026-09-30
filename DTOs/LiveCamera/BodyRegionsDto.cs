namespace VisionAttributeAI.DTOs.LiveCamera;

public record BodyRegionsDto
{
    public BoundingBoxDto? Head { get; init; }
    public BoundingBoxDto? UpperTorso { get; init; }
    public BoundingBoxDto? LeftWrist { get; init; }
    public BoundingBoxDto? RightWrist { get; init; }
    public BoundingBoxDto? LowerBody { get; init; }
    public BoundingBoxDto? Feet { get; init; }
}
