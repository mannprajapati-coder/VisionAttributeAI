namespace VisionAttributeAI.DTOs.LiveCamera;

public record BodyVisibilityDto
{
    public bool Head { get; init; }
    public bool Upper { get; init; }
    public bool LeftArm { get; init; }
    public bool RightArm { get; init; }
    public bool LeftWrist { get; init; }
    public bool RightWrist { get; init; }
    public bool Lower { get; init; }
    public bool LeftLeg { get; init; }
    public bool RightLeg { get; init; }
    public bool Feet { get; init; }
    public float PoseConfidence { get; init; }
    public int KeypointCount { get; init; }

    // Gestures / Raised Hands
    public bool LeftHandRaised { get; init; }
    public bool RightHandRaised { get; init; }

    // Pose Orientation & Boundary Clipping
    public string Orientation { get; init; } = "Uncertain";
    public bool HeadClipped { get; init; }
    public bool UpperClipped { get; init; }
    public bool LowerClipped { get; init; }
    public bool FeetClipped { get; init; }
    public bool LeftWristClipped { get; init; }
    public bool RightWristClipped { get; init; }
}
