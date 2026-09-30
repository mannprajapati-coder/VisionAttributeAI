using VisionAttributeAI.Models.Benchmark;
using VisionAttributeAI.Models.Tracking;

namespace VisionAttributeAI.Services.Benchmark;

public interface IBenchmarkService
{
    void SetGroundTruth(PersonGroundTruth groundTruth);
    PersonGroundTruth? GetGroundTruth(int personId);
    IReadOnlyDictionary<int, PersonGroundTruth> GetAllGroundTruths();
    void RemoveGroundTruth(int personId);
    void Reset();
    BenchmarkEvaluationReport EvaluateTracks(IReadOnlyList<TrackedPersonState> tracks);
}
