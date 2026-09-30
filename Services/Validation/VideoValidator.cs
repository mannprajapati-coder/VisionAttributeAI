namespace VisionAttributeAI.Services.Validation;

/// <summary>
/// Validates uploaded video files against allowed extensions (.mp4, .mov, .avi, .mkv, .webm), MIME types, and size limits.
/// </summary>
public class VideoValidator : IVideoValidator
{
    public const long DefaultMaxVideoSizeBytes = 100 * 1024 * 1024; // 100 MB

    private static readonly HashSet<string> AllowedExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm"
    };

    private static readonly HashSet<string> AllowedContentTypes = new(StringComparer.OrdinalIgnoreCase)
    {
        "video/mp4",
        "video/quicktime",
        "video/x-msvideo",
        "video/x-matroska",
        "video/webm",
        "application/octet-stream"
    };

    private readonly long _maxFileSizeBytes;

    public VideoValidator(long maxFileSizeBytes = DefaultMaxVideoSizeBytes)
    {
        _maxFileSizeBytes = maxFileSizeBytes;
    }

    public VideoValidationResult Validate(IFormFile? file)
    {
        if (file == null || file.Length == 0)
        {
            return VideoValidationResult.Failure("No video file was provided or the file is empty.");
        }

        if (file.Length > _maxFileSizeBytes)
        {
            var maxMb = _maxFileSizeBytes / (1024.0 * 1024.0);
            return VideoValidationResult.Failure($"Video file size ({file.Length / (1024.0 * 1024.0):F2} MB) exceeds the maximum allowed limit of {maxMb:F0} MB.");
        }

        var extension = Path.GetExtension(file.FileName);
        if (string.IsNullOrWhiteSpace(extension) || !AllowedExtensions.Contains(extension))
        {
            return VideoValidationResult.Failure($"Invalid video extension '{extension}'. Supported formats: .mp4, .mov, .avi, .mkv, .webm");
        }

        if (!string.IsNullOrWhiteSpace(file.ContentType) && !AllowedContentTypes.Contains(file.ContentType))
        {
            return VideoValidationResult.Failure($"Invalid content type '{file.ContentType}'. Expected a supported video format.");
        }

        return VideoValidationResult.Success(extension.ToLowerInvariant(), file.ContentType ?? "video/mp4", file.Length);
    }
}
