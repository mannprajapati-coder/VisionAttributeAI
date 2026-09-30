using VisionAttributeAI.Models.AI;

namespace VisionAttributeAI.Models.Pose;

/// <summary>
/// Clamped bounding box coordinates for individual isolated anatomical body regions.
/// </summary>
public record BodyRegions
{
    public BoundingBox? HeadRegion { get; init; }
    public BoundingBox? UpperTorsoRegion { get; init; }
    public BoundingBox? LeftWristRegion { get; init; }
    public BoundingBox? RightWristRegion { get; init; }
    public BoundingBox? LowerBodyRegion { get; init; }
    public BoundingBox? FeetRegion { get; init; }
    public BoundingBox? BagRegion { get; init; }
}
