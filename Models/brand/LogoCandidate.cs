using VisionAttributeAI.Models.AI;

namespace VisionAttributeAI.Models.Brand;

public class LogoCandidate
{
    public BoundingBox Box { get; set; } = new(0, 0, 0, 0);
    public float Confidence { get; set; }
    public int ClassId { get; set; } // 0: Text, 1: NoText
    public string RegionName { get; set; } = string.Empty;
}
