using OpenCvSharp;
using VisionAttributeAI.Models.AI;

namespace VisionAttributeAI.Services.Cropping;

/// <summary>
/// High-performance person cropping service using OpenCvSharp.
/// Ensures strict image boundary clamping, prevents memory faults, and encodes clean debugging previews.
/// </summary>
public class PersonCropper : IPersonCropper
{
    private readonly ILogger<PersonCropper> _logger;

    public PersonCropper(ILogger<PersonCropper> logger)
    {
        _logger = logger;
    }

    public PersonCropResult CropPerson(Mat sourceImage, BoundingBox box, int personId, bool generateBase64Preview = true)
    {
        if (sourceImage.Empty())
        {
            throw new ArgumentException("Source image is empty or invalid.", nameof(sourceImage));
        }

        int imgWidth = sourceImage.Width;
        int imgHeight = sourceImage.Height;

        // 1. Validate and strictly clamp bounding box inside image boundaries
        int x = (int)Math.Floor(Math.Clamp(box.X, 0, Math.Max(0, imgWidth - 1)));
        int y = (int)Math.Floor(Math.Clamp(box.Y, 0, Math.Max(0, imgHeight - 1)));
        int width = (int)Math.Ceiling(Math.Clamp(box.Width, 1, Math.Max(1, imgWidth - x)));
        int height = (int)Math.Ceiling(Math.Clamp(box.Height, 1, Math.Max(1, imgHeight - y)));

        // Double check bounds to avoid OpenCV ROI assertion errors
        if (x + width > imgWidth) width = imgWidth - x;
        if (y + height > imgHeight) height = imgHeight - y;

        var clampedBox = new BoundingBox
        {
            X = x,
            Y = y,
            Width = width,
            Height = height
        };

        var roi = new Rect(x, y, width, height);

        // 2. Extract cropped person region (Clone ensures independent memory allocation)
        using var roiView = new Mat(sourceImage, roi);
        var crop = roiView.Clone();

        string? base64Preview = null;
        if (generateBase64Preview && !crop.Empty())
        {
            // Encode high-quality JPEG for UI debugging inspection
            var encodingParams = new[]
            {
                new ImageEncodingParam(ImwriteFlags.JpegQuality, 95)
            };

            Cv2.ImEncode(".jpg", crop, out var buffer, encodingParams);
            base64Preview = "data:image/jpeg;base64," + Convert.ToBase64String(buffer);
        }

        _logger.LogDebug("Cropped Person #{Id} ROI: [X:{X}, Y:{Y}, W:{W}, H:{H}] from {OrigW}x{OrigH}",
            personId, x, y, width, height, imgWidth, imgHeight);

        return new PersonCropResult
        {
            PersonId = personId,
            CroppedMat = crop,
            ClampedBoundingBox = clampedBox,
            Base64DataUrl = base64Preview
        };
    }
}
