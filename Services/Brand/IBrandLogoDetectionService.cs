using OpenCvSharp;
using VisionAttributeAI.Models.Brand;

namespace VisionAttributeAI.Services.Brand;

public interface IBrandLogoDetectionService
{
    bool IsModelLoaded { get; }
    Task<IReadOnlyList<LogoCandidate>> DetectLogoCandidatesAsync(Mat regionCrop, string regionName, CancellationToken cancellationToken = default);
}
