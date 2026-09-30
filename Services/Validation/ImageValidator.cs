namespace VisionAttributeAI.Services.Validation;

/// <summary>
/// Validates uploaded image files against allowed extensions (JPG, JPEG, PNG), MIME types, file size limits, and binary signatures.
/// </summary>
public class ImageValidator : IImageValidator
{
    public const long DefaultMaxFileSize = 10 * 1024 * 1024; // 10 MB

    private static readonly HashSet<string> AllowedExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".jpg",
        ".jpeg",
        ".jfif",
        ".png"
    };

    private static readonly HashSet<string> AllowedContentTypes = new(StringComparer.OrdinalIgnoreCase)
    {
        "image/jpeg",
        "image/jpg",
        "image/jfif",
        "image/x-jfif",
        "image/pjpeg",
        "image/png",
        "image/x-png",
        "application/octet-stream"
    };

    private static readonly byte[] JpegHeader = [0xFF, 0xD8, 0xFF];
    private static readonly byte[] PngHeader = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

    private readonly long _maxFileSizeBytes;

    public ImageValidator(long maxFileSizeBytes = DefaultMaxFileSize)
    {
        _maxFileSizeBytes = maxFileSizeBytes;
    }

    public ImageValidationResult Validate(IFormFile? file)
    {
        if (file == null || file.Length == 0)
        {
            return ImageValidationResult.Failure("No image file was provided or the file is empty.");
        }

        if (file.Length > _maxFileSizeBytes)
        {
            var maxMb = _maxFileSizeBytes / (1024.0 * 1024.0);
            return ImageValidationResult.Failure($"File size ({file.Length / (1024.0 * 1024.0):F2} MB) exceeds the maximum allowed limit of {maxMb:F0} MB.");
        }

        var extension = Path.GetExtension(file.FileName);
        if (string.IsNullOrWhiteSpace(extension) || !AllowedExtensions.Contains(extension))
        {
            return ImageValidationResult.Failure($"Invalid file extension '{extension}'. Only .jpg, .jpeg, .jfif, and .png files are supported.");
        }

        if (!string.IsNullOrWhiteSpace(file.ContentType) && !AllowedContentTypes.Contains(file.ContentType))
        {
            return ImageValidationResult.Failure($"Invalid content type '{file.ContentType}'. Expected JPEG or PNG image.");
        }

        // Validate binary magic bytes by reading header
        using var stream = file.OpenReadStream();
        var headerBuffer = new byte[8];
        var bytesRead = stream.Read(headerBuffer, 0, headerBuffer.Length);

        if (!MatchesImageSignature(headerBuffer, bytesRead, out var detectedType))
        {
            return ImageValidationResult.Failure("File header does not match a valid JPEG or PNG image signature. The file may be corrupted or renamed.");
        }

        return ImageValidationResult.Success(extension.ToLowerInvariant(), file.ContentType ?? detectedType, file.Length);
    }

    public ImageValidationResult ValidateBytes(byte[]? imageBytes)
    {
        if (imageBytes == null || imageBytes.Length == 0)
        {
            return ImageValidationResult.Failure("Image buffer is empty.");
        }

        if (imageBytes.Length > _maxFileSizeBytes)
        {
            var maxMb = _maxFileSizeBytes / (1024.0 * 1024.0);
            return ImageValidationResult.Failure($"Image buffer size exceeds the maximum allowed limit of {maxMb:F0} MB.");
        }

        var headerBuffer = imageBytes.Take(8).ToArray();
        if (!MatchesImageSignature(headerBuffer, headerBuffer.Length, out var detectedType))
        {
            return ImageValidationResult.Failure("Buffer header does not match a valid JPEG or PNG image signature.");
        }

        var extension = detectedType.Contains("png", StringComparison.OrdinalIgnoreCase) ? ".png" : ".jpg";
        return ImageValidationResult.Success(extension, detectedType, imageBytes.Length);
    }

    private static bool MatchesImageSignature(byte[] buffer, int length, out string detectedType)
    {
        detectedType = string.Empty;

        // Check JPEG signature: starts with FF D8 FF
        if (length >= 3 && buffer[0] == JpegHeader[0] && buffer[1] == JpegHeader[1] && buffer[2] == JpegHeader[2])
        {
            detectedType = "image/jpeg";
            return true;
        }

        // Check PNG signature: starts with 89 50 4E 47 0D 0A 1A 0A
        if (length >= 8 && buffer.Take(8).SequenceEqual(PngHeader))
        {
            detectedType = "image/png";
            return true;
        }

        return false;
    }
}
