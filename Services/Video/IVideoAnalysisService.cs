using VisionAttributeAI.DTOs.Video;

namespace VisionAttributeAI.Services.Video;

public interface IVideoAnalysisService
{
    Task<VideoAnalysisResponseDto> AnalyzeVideoAsync(
        Stream videoStream,
        string fileExtension,
        double targetAnalysisFps = 3.0,
        CancellationToken cancellationToken = default);
}
