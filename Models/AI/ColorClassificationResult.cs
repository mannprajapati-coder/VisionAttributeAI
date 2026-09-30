namespace VisionAttributeAI.Models.AI;

/// <summary>
/// Detailed fabric color classification result with primary/secondary percentage distribution.
/// </summary>
public record ColorClassificationResult
{
    public string PrimaryColor { get; init; } = "Unknown";
    public float PrimaryPercentage { get; init; }
    public string SecondaryColor { get; init; } = "None";
    public float SecondaryPercentage { get; init; }
    public float ColorConfidence { get; init; }
    public bool IsMultiColor { get; init; }
    public Dictionary<string, float> AllColorPercentages { get; init; } = new();

    public string Summary => IsMultiColor ? $"MultiColor ({PrimaryColor} {PrimaryPercentage:P0} / {SecondaryColor} {SecondaryPercentage:P0})"
        : $"{PrimaryColor} ({PrimaryPercentage:P0})";
}
