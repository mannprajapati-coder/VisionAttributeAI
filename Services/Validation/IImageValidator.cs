using Microsoft.AspNetCore.Http;

namespace VisionAttributeAI.Services.Validation;

/// <summary>
/// Service contract for validating uploaded image files before pipeline ingestion.
/// </summary>
public interface IImageValidator
{
    /// <summary>
    /// Validates presence, file size, extension, MIME type, and magic bytes for an uploaded file.
    /// </summary>
    ImageValidationResult Validate(IFormFile? file);

    /// <summary>
    /// Validates raw in-memory image bytes against allowed image formats and signatures.
    /// </summary>
    ImageValidationResult ValidateBytes(byte[]? imageBytes);
}
