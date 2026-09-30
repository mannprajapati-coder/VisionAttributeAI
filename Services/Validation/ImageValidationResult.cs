namespace VisionAttributeAI.Services.Validation;

/// <summary>
/// Result of image validation containing validation status, error message, and detected format details.
/// </summary>
public record ImageValidationResult
{
    public bool IsValid { get; init; }
    public string? ErrorMessage { get; init; }
    public string? FileExtension { get; init; }
    public string? ContentType { get; init; }
    public long FileSizeBytes { get; init; }

    public static ImageValidationResult Success(string extension, string contentType, long sizeBytes) =>
        new()
        {
            IsValid = true,
            FileExtension = extension,
            ContentType = contentType,
            FileSizeBytes = sizeBytes
        };

    public static ImageValidationResult Failure(string errorMessage) =>
        new()
        {
            IsValid = false,
            ErrorMessage = errorMessage
        };
}
