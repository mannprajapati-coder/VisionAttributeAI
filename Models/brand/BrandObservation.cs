namespace VisionAttributeAI.Models.Brand;

public class BrandObservation
{
    public long FrameIndex { get; set; }
    public string Region { get; set; } = string.Empty;
    public string BrandName { get; set; } = "BrandUnknown";
    public float Similarity { get; set; }
    public string RunnerUpBrand { get; set; } = string.Empty;
    public float RunnerUpSimilarity { get; set; }
    public float Margin { get; set; }
    public bool PassedDualGate { get; set; }
    public string MatchedExemplarFile { get; set; } = string.Empty;
    public DateTime TimestampUtc { get; set; } = DateTime.UtcNow;
}
