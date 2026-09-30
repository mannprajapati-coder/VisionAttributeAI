using OpenCvSharp;
using VisionAttributeAI.Models.AI;

namespace VisionAttributeAI.Services.Cropping;

/// <summary>
/// Service contract for cropping person regions of interest (ROI) from images.
/// </summary>
public interface IPersonCropper
{
    /// <summary>
    /// Safely crops an individual person ROI from the source image, ensuring bounds clamping, high visual quality, and optional Base64 preview generation.
    /// </summary>
    PersonCropResult CropPerson(Mat sourceImage, BoundingBox box, int personId, bool generateBase64Preview = true);
}
