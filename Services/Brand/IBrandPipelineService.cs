using OpenCvSharp;
using VisionAttributeAI.Models.AI;
using VisionAttributeAI.Models.Brand;
using VisionAttributeAI.Models.Pose;

namespace VisionAttributeAI.Services.Brand;

public interface IBrandPipelineService
{
    bool IsEnabled { get; }
    Task<PersonBrandResult> ProcessPersonBrandsAsync(
        Mat frameMat,
        BoundingBox personBox,
        PersonPoseResult pose,
        TrackedBrandState brandState,
        long frameIndex,
        bool isVideoMode,
        bool possibleIdSwitch,
        CancellationToken cancellationToken = default);
}
