using OpenCvSharp;

namespace VisionAttributeAI.Models.AI;

/// <summary>
/// Encapsulates the result of cropping an individual person from a larger image.
/// </summary>
public class PersonCropResult : IDisposable
{
    private bool _disposed;

    public int PersonId { get; init; }
    public required Mat CroppedMat { get; init; }
    public int CropWidth => CroppedMat.Width;
    public int CropHeight => CroppedMat.Height;
    public required BoundingBox ClampedBoundingBox { get; init; }
    public string? Base64DataUrl { get; set; }

    public void Dispose()
    {
        if (_disposed) return;
        CroppedMat.Dispose();
        _disposed = true;
        GC.SuppressFinalize(this);
    }
}
