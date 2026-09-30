namespace VisionAttributeAI.Models.AI;

/// <summary>
/// Represents a 2D bounding box in pixel coordinates.
/// </summary>
public record BoundingBox
{
    public float X { get; init; }
    public float Y { get; init; }
    public float Width { get; init; }
    public float Height { get; init; }

    public BoundingBox() { }

    public BoundingBox(float x, float y, float width, float height)
    {
        X = x;
        Y = y;
        Width = width;
        Height = height;
    }

    public float Right => X + Width;
    public float Bottom => Y + Height;

    /// <summary>
    /// Clips the bounding box coordinates to ensure they remain inside the specified image boundaries.
    /// </summary>
    public BoundingBox Clamp(int imageWidth, int imageHeight)
    {
        var clampedX = Math.Clamp(X, 0, Math.Max(0, imageWidth - 1));
        var clampedY = Math.Clamp(Y, 0, Math.Max(0, imageHeight - 1));
        var clampedWidth = Math.Clamp(Width, 1, Math.Max(1, imageWidth - clampedX));
        var clampedHeight = Math.Clamp(Height, 1, Math.Max(1, imageHeight - clampedY));

        return this with
        {
            X = clampedX,
            Y = clampedY,
            Width = clampedWidth,
            Height = clampedHeight
        };
    }
}
