namespace VisionAttributeAI.Models.AI;

/// <summary>
/// Represents individual recognized attribute with its score/confidence.
/// </summary>
public record AttributeValue(string Value, float Confidence);

/// <summary>
/// Represents the set of recognized attributes for a detected person.
/// </summary>
public class PersonAttributeResult
{
    public Dictionary<string, AttributeValue> Attributes { get; init; } = new(StringComparer.OrdinalIgnoreCase);

    public float? OverallConfidence { get; init; }

    public void SetAttribute(string key, string value, float confidence)
    {
        Attributes[key] = new AttributeValue(value, confidence);
    }
}
