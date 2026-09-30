using VisionAttributeAI.DTOs.LiveCamera;

namespace VisionAttributeAI.Services.LiveCamera;

/// <summary>
/// Contract for orchestrating real-time live camera frame analysis:
/// In-Memory Ingestion -> YOLO Detection -> Tracking -> Quality Filtering -> Fashion-CLIP -> Temporal Aggregation.
/// </summary>
public interface ILiveCameraAnalysisService
{
    /// <summary>
    /// Processes a single live camera frame in-memory and returns updated multi-person tracking and aggregated predictions.
    /// </summary>
    Task<LiveFrameResponseDto> ProcessLiveFrameAsync(byte[] frameBytes, long frameIndex, CancellationToken cancellationToken = default);

    /// <summary>
    /// Returns the most recently processed live frame response and active tracks for remote spectator/monitoring clients.
    /// </summary>
    LiveFrameResponseDto? GetLatestState();

    /// <summary>
    /// Resets all active tracking states and clears aggregation histories.
    /// </summary>
    void Reset();
}
