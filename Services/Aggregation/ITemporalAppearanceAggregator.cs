using VisionAttributeAI.Models.Tracking;

namespace VisionAttributeAI.Services.Aggregation;

/// <summary>
/// Contract for multi-frame temporal appearance aggregation per PersonId.
/// </summary>
public interface ITemporalAppearanceAggregator
{
    /// <summary>
    /// Updates the rolling observation history and calculates weighted appearance prediction for a tracked person.
    /// </summary>
    void Aggregate(TrackedPersonState track, PersonObservation newObservation);

    /// <summary>
    /// Finalizes candidate attributes and resolves consensus states upon video/stream completion.
    /// </summary>
    void FinalizeTrack(TrackedPersonState track);
}
