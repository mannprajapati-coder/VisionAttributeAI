namespace VisionAttributeAI.Models.Benchmark;

/// <summary>
/// Ground truth annotations set by developer for accuracy benchmarking against a live tracked person.
/// </summary>
public record PersonGroundTruth
{
    public int PersonId { get; init; }
    public string ExpectedSex { get; init; } = "Skip"; // Male, Female, Skip
    public string ExpectedUpperType { get; init; } = "Skip"; // T-Shirt, Shirt, Jacket, Blazer, Hoodie, Sweater, Other, Skip
    public string ExpectedUpperColor { get; init; } = "Skip";
    public string ExpectedLowerType { get; init; } = "Skip"; // Trousers, Jeans, Shorts, Skirt, Dress, Other, Skip
    public string ExpectedLowerColor { get; init; } = "Skip";
    public string ExpectedShoes { get; init; } = "Skip"; // Sneakers, Boots, Formal Shoes, Sandals, Other, Skip
    public DateTime CreatedAtUtc { get; init; } = DateTime.UtcNow;
}

public record AttributeAccuracyMetric
{
    public string AttributeName { get; init; } = string.Empty;
    public int TotalEvaluated { get; init; }
    public int CorrectCount { get; init; }
    public int IncorrectCount { get; init; }
    public int SkippedCount { get; init; }
    public int UnknownCount { get; init; }
    public int InsufficientEvidenceCount { get; init; }
    public float AccuracyPercent => TotalEvaluated > 0 ? ((float)CorrectCount / TotalEvaluated) * 100f : 0f;
}

public record BenchmarkEvaluationReport
{
    public DateTime GeneratedAtUtc { get; init; } = DateTime.UtcNow;
    public int TotalTracksBenchmarked { get; set; }
    public float OverallAccuracyPercent { get; set; }
    public List<AttributeAccuracyMetric> AttributeMetrics { get; init; } = new();
    public List<TrackComparisonDetail> TrackDetails { get; init; } = new();
}

public record TrackComparisonDetail
{
    public int PersonId { get; init; }
    public string AttributeName { get; init; } = string.Empty;
    public string Expected { get; init; } = string.Empty;
    public string Predicted { get; init; } = string.Empty;
    public float Confidence { get; init; }
    public bool IsMatch { get; init; }
    public string State { get; init; } = string.Empty;
}
