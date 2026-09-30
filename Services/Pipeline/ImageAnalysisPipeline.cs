using System.Diagnostics;
using OpenCvSharp;
using VisionAttributeAI.DTOs;
using VisionAttributeAI.DTOs.LiveCamera;
using VisionAttributeAI.Models.AI;
using VisionAttributeAI.Models.Pose;
using VisionAttributeAI.Services.Detection;
using VisionAttributeAI.Services.PersonAnalysis;
using VisionAttributeAI.Services.Pose;
using VisionAttributeAI.Services.Validation;

namespace VisionAttributeAI.Services.Pipeline;

/// <summary>
/// Implements the unified multi-person image analysis pipeline:
/// Decodes Image -> YOLO Detect ALL People -> Multi-Signal Candidate Validation -> Assign Person #1..#N ->
/// For each person: Crop + YOLO-Pose + Body Regions + Attribute Quality + Fashion-CLIP/Color ->
/// Output independent single-image person cards and bounding boxes.
/// </summary>
public class ImageAnalysisPipeline : IImageAnalysisPipeline
{
    private readonly IPersonDetector _personDetector;
    private readonly IPoseEstimationService _poseService;
    private readonly IUnifiedPersonAnalysisService _personAnalysisService;
    private readonly IPersonCandidateValidator _candidateValidator;
    private readonly ILogger<ImageAnalysisPipeline> _logger;

    public ImageAnalysisPipeline(
        IPersonDetector personDetector,
        IPoseEstimationService poseService,
        IUnifiedPersonAnalysisService personAnalysisService,
        IPersonCandidateValidator candidateValidator,
        ILogger<ImageAnalysisPipeline> logger)
    {
        _personDetector = personDetector;
        _poseService = poseService;
        _personAnalysisService = personAnalysisService;
        _candidateValidator = candidateValidator;
        _logger = logger;
    }

    public Task<ImageAnalysisResult> AnalyzeAsync(Mat image, CancellationToken cancellationToken = default)
    {
        return ProcessCoreAsync(image, decodeMs: 0, cancellationToken);
    }

    public async Task<ImageAnalysisResult> AnalyzeAsync(byte[] imageBytes, CancellationToken cancellationToken = default)
    {
        var stopwatch = Stopwatch.StartNew();
        using var mat = Cv2.ImDecode(imageBytes, ImreadModes.Color);
        stopwatch.Stop();

        if (mat == null || mat.Empty())
        {
            throw new ArgumentException("Failed to decode image from provided byte array.");
        }

        return await ProcessCoreAsync(mat, stopwatch.Elapsed.TotalMilliseconds, cancellationToken);
    }

    public async Task<ImageAnalysisResult> AnalyzeAsync(Stream imageStream, CancellationToken cancellationToken = default)
    {
        using var memoryStream = new MemoryStream();
        await imageStream.CopyToAsync(memoryStream, cancellationToken);
        var bytes = memoryStream.ToArray();

        return await AnalyzeAsync(bytes, cancellationToken);
    }

    private async Task<ImageAnalysisResult> ProcessCoreAsync(Mat image, double decodeMs, CancellationToken cancellationToken)
    {
        var totalStopwatch = Stopwatch.StartNew();
        var metrics = new AnalysisMetrics
        {
            ImageDecodeMs = decodeMs
        };

        // 1. YOLO Person Detection (Find ALL people)
        var detectStopwatch = Stopwatch.StartNew();
        var detections = await _personDetector.DetectPersonsAsync(image, cancellationToken);
        detectStopwatch.Stop();
        metrics.DetectionMs = detectStopwatch.Elapsed.TotalMilliseconds;

        // 2. Pose Estimation on All Detections
        var poseResults = await _poseService.EstimatePoseAsync(image, detections, cancellationToken);

        // 3. Multi-Signal Person Candidate Validation Gate
        var validatedCandidates = _candidateValidator.FilterValidCandidates(detections, poseResults, image.Width, image.Height);

        // 4. For EACH validated human candidate, run independent unified person analysis
        var attrStopwatch = Stopwatch.StartNew();
        var analyzedPersons = new List<AnalyzedPerson>();

        for (int i = 0; i < validatedCandidates.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var candidate = validatedCandidates[i];
            var detection = candidate.Detection;
            var pose = candidate.Pose;
            int personId = i + 1;

            var analysisOutput = await _personAnalysisService.AnalyzePersonAsync(
                image,
                detection.Box,
                detection.Confidence,
                personId,
                frameIndex: 0,
                pose,
                generateBase64Crop: true,
                brandState: null,
                isVideoMode: false,
                possibleIdSwitch: false,
                cancellationToken: cancellationToken);

            // Populate PersonAttributeResult
            var personAttrs = new PersonAttributeResult();
            var obs = analysisOutput.Observation;

            personAttrs.SetAttribute("Appearance", obs.SexPrediction, obs.SexConfidence);
            personAttrs.SetAttribute("UpperClothing", obs.UpperType, obs.UpperTypeConfidence);
            personAttrs.SetAttribute("UpperColor", obs.UpperColor, obs.UpperColorConfidence);
            personAttrs.SetAttribute("LowerClothing", obs.LowerType, obs.LowerTypeConfidence);
            personAttrs.SetAttribute("LowerColor", obs.LowerColor, obs.LowerColorConfidence);
            personAttrs.SetAttribute("Shoes", obs.ShoesType, obs.ShoesConfidence);
            personAttrs.SetAttribute("Watch", obs.WatchStatus, obs.WatchConfidence);

            if (obs.Brands != null)
            {
                personAttrs.SetAttribute("UpperBrand", obs.Brands.UpperBrand.BrandName, obs.Brands.UpperBrand.Similarity);
                personAttrs.SetAttribute("LowerBrand", obs.Brands.LowerBrand.BrandName, obs.Brands.LowerBrand.Similarity);
                personAttrs.SetAttribute("WatchBrand", obs.Brands.WatchBrand.BrandName, obs.Brands.WatchBrand.Similarity);
                personAttrs.SetAttribute("ShoeBrand", obs.Brands.ShoeBrand.BrandName, obs.Brands.ShoeBrand.Similarity);
                personAttrs.SetAttribute("BagBrand", obs.Brands.BagBrand.BrandName, obs.Brands.BagBrand.Similarity);

                var topBrand = obs.Brands.TopDetectedBrand;
                if (topBrand != null)
                {
                    personAttrs.SetAttribute("Brand", $"{topBrand.BrandName} ({topBrand.Region})", topBrand.Similarity);
                }
                else
                {
                    personAttrs.SetAttribute("Brand", "BrandUnknown", 0f);
                }
            }

            analyzedPersons.Add(new AnalyzedPerson
            {
                PersonId = personId,
                Detection = detection,
                CropBase64 = analysisOutput.CropBase64,
                CropWidth = analysisOutput.CropWidth,
                CropHeight = analysisOutput.CropHeight,
                Attributes = personAttrs
            });
        }

        attrStopwatch.Stop();
        metrics.AttributeRecognitionMs = attrStopwatch.Elapsed.TotalMilliseconds;

        totalStopwatch.Stop();
        metrics.TotalElapsedMs = totalStopwatch.Elapsed.TotalMilliseconds + decodeMs;

        _logger.LogInformation(
            "Analyzed image ({Width}x{Height}) - Found {Count} person(s) in {TotalMs:F1}ms (Det: {DetMs:F1}ms, Attr: {AttrMs:F1}ms)",
            image.Width,
            image.Height,
            analyzedPersons.Count,
            metrics.TotalElapsedMs,
            metrics.DetectionMs,
            metrics.AttributeRecognitionMs);

        return new ImageAnalysisResult
        {
            ImageWidth = image.Width,
            ImageHeight = image.Height,
            Persons = analyzedPersons,
            Metrics = metrics
        };
    }
}
