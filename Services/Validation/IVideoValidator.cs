namespace VisionAttributeAI.Services.Validation;

public record VideoValidationResult
{
    public bool IsValid { get; init; }
    public string? ErrorMessage { get; init; }
    public string Extension { get; init; } = string.Empty;
    public string ContentType { get; init; } = string.Empty;
    public long FileSizeBytes { get; init; }

    public static VideoValidationResult Success(string extension, string contentType, long fileSizeBytes) =>
        new() { IsValid = true, Extension = extension, ContentType = contentType, FileSizeBytes = fileSizeBytes };

    public static VideoValidationResult Failure(string errorMessage) =>
        new() { IsValid = false, ErrorMessage = errorMessage };
}

public interface IVideoValidator
{
    VideoValidationResult Validate(IFormFile? file);
}
