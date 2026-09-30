namespace VisionAttributeAI.DTOs;

public record BrandDto
{
    public string BrandName { get; init; } = "BrandUnknown";
    public string State { get; init; } = "Analyzing"; // NotVisible, Analyzing, BrandUnknown, BrandCandidate, BrandStable
    public float Similarity { get; init; }
    public string RunnerUpBrand { get; init; } = string.Empty;
    public float RunnerUpSimilarity { get; init; }
    public float Margin { get; init; }
    public int ObservationCount { get; init; }
    public bool IsStable { get; init; }
    public string Region { get; init; } = string.Empty;
}

public record PersonBrandDto
{
    public BrandDto UpperBrand { get; init; } = new() { Region = "Upper", State = "NotVisible", BrandName = "Not Visible" };
    public BrandDto LowerBrand { get; init; } = new() { Region = "Lower", State = "NotVisible", BrandName = "Not Visible" };
    public BrandDto WatchBrand { get; init; } = new() { Region = "Watch", State = "NotVisible", BrandName = "Not Visible" };
    public BrandDto ShoeBrand { get; init; } = new() { Region = "Shoes", State = "NotVisible", BrandName = "Not Visible" };
    public BrandDto BagBrand { get; init; } = new() { Region = "Bag", State = "NotVisible", BrandName = "Not Visible" };
    public string TopDetectedBrand { get; init; } = "BrandUnknown";
    public float TopBrandSimilarity { get; init; }
    public bool HasAnyAcceptedBrand { get; init; }
    public bool HasAnyStableBrand { get; init; }
}
