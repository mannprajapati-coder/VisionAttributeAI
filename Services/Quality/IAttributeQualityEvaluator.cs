using OpenCvSharp;
using VisionAttributeAI.Models.AI;
using VisionAttributeAI.Models.Pose;
using VisionAttributeAI.Models.Tracking;

namespace VisionAttributeAI.Services.Quality;

public interface IAttributeQualityEvaluator
{
    AttributeQualityScores EvaluateQuality(
        Mat frameMat,
        BoundingBox personBox,
        float yoloConfidence,
        PersonPoseResult pose,
        int frameWidth,
        int frameHeight);
}
