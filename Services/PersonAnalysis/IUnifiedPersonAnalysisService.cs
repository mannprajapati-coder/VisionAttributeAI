using OpenCvSharp;
using VisionAttributeAI.Models.AI;
using VisionAttributeAI.Models.Pose;
using VisionAttributeAI.Models.Tracking;

namespace VisionAttributeAI.Services.PersonAnalysis;

/// <summary>
/// Result of analyzing an individual person detection in a single frame.
/// </summary>
public record PersonAnalysisOutput
{
    public required int PersonId { get; init; }
    public required BoundingBox BoundingBox { get; init; }
    public required float DetectionConfidence { get; init; }
    public required PersonPoseResult Pose { get; init; }
    public required AttributeQualityScores QualityScores { get; init; }
    public required TrackQualityResult GlobalQuality { get; init; }
    public required PersonObservation Observation { get; init; }
    public string? CropBase64 { get; init; }
    public int CropWidth { get; init; }
    public int CropHeight { get; init; }
    public VisionAttributeAI.Models.Brand.PersonBrandResult? Brands => Observation.Brands;
}

/// <summary>
/// Common person-analysis engine used across Image, Video, and Live Camera pipelines.
/// Analyzes a single person's pose, body regions, attribute-specific quality, Fashion-CLIP, color, watch, and brands.
/// </summary>
public interface IUnifiedPersonAnalysisService
{
    Task<PersonAnalysisOutput> AnalyzePersonAsync(
        Mat frameMat,
        BoundingBox boundingBox,
        float confidence,
        int personId,
        long frameIndex,
        PersonPoseResult pose,
        bool generateBase64Crop = false,
        VisionAttributeAI.Models.Brand.TrackedBrandState? brandState = null,
        bool isVideoMode = false,
        bool possibleIdSwitch = false,
        CancellationToken cancellationToken = default);
}
