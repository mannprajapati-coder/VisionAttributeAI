using OpenCvSharp;
using VisionAttributeAI.Models.Brand;

namespace VisionAttributeAI.Services.Brand;

public interface IBrandRecognitionService
{
    bool IsModelLoaded { get; }
    int GalleryExemplarCount { get; }
    Task<BrandResult> RecognizeBrandAsync(Mat logoCrop, string regionName, CancellationToken cancellationToken = default);
}
