using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Options;
using VisionAttributeAI.DTOs;
using VisionAttributeAI.DTOs.Video;
using VisionAttributeAI.Models.AI;
using VisionAttributeAI.Services.Aggregation;
using VisionAttributeAI.Services.Attributes;
using VisionAttributeAI.Services.Cropping;
using VisionAttributeAI.Services.Detection;
using VisionAttributeAI.Services.PersonAnalysis;
using VisionAttributeAI.Services.Pipeline;
using VisionAttributeAI.Services.Pose;
using VisionAttributeAI.Services.Quality;
using VisionAttributeAI.Services.Validation;
using VisionAttributeAI.Services.Video;
using VisionAttributeAI.Tests;

namespace VisionAttributeAI.Controllers;

[ApiController]
[Route("api/[controller]")]
public class AnalysisController : ControllerBase
{
    private readonly IImageAnalysisPipeline _imagePipeline;
    private readonly IVideoAnalysisService _videoService;
    private readonly IImageValidator _imageValidator;
    private readonly IVideoValidator _videoValidator;
    private readonly IPersonDetector _personDetector;
    private readonly IPedestrianAttributeRecognizer _attributeRecognizer;
    private readonly IPedestrianAttributeService _attributeService;
    private readonly IUnifiedPersonAnalysisService _unifiedAnalysisService;
    private readonly IPoseEstimationService _poseService;
    private readonly IPersonCandidateValidator _candidateValidator;
    private readonly IPersonCropper _cropper;
    private readonly ITemporalAppearanceAggregator _aggregator;
    private readonly IFashionClipService _clipService;
    private readonly IOptions<PersonAnalysisOptions> _options;
    private readonly IOptions<VisionAttributeAI.Models.Brand.BrandOptions> _brandOptions;
    private readonly ILogger<AnalysisController> _logger;

    public AnalysisController(
        IImageAnalysisPipeline imagePipeline,
        IVideoAnalysisService videoService,
        IImageValidator imageValidator,
        IVideoValidator videoValidator,
        IPersonDetector personDetector,
        IPedestrianAttributeRecognizer attributeRecognizer,
        IPedestrianAttributeService attributeService,
        IUnifiedPersonAnalysisService unifiedAnalysisService,
        IPoseEstimationService poseService,
        IPersonCandidateValidator candidateValidator,
        IPersonCropper cropper,
        ITemporalAppearanceAggregator aggregator,
        IFashionClipService clipService,
        IOptions<PersonAnalysisOptions> options,
        IOptions<VisionAttributeAI.Models.Brand.BrandOptions> brandOptions,
        ILogger<AnalysisController> logger)
    {
        _imagePipeline = imagePipeline;
        _videoService = videoService;
        _imageValidator = imageValidator;
        _videoValidator = videoValidator;
        _personDetector = personDetector;
        _attributeRecognizer = attributeRecognizer;
        _attributeService = attributeService;
        _unifiedAnalysisService = unifiedAnalysisService;
        _poseService = poseService;
        _candidateValidator = candidateValidator;
        _cropper = cropper;
        _aggregator = aggregator;
        _clipService = clipService;
        _options = options;
        _brandOptions = brandOptions;
        _logger = logger;
    }

    /// <summary>
    /// TAB 1: Upload a full image (JPG, JPEG, PNG) for multi-person detection and independent attribute recognition.
    /// Ingests into in-memory OpenCV Mat without permanent disk persistence.
    /// </summary>
    [HttpPost("image")]
    [Consumes("multipart/form-data")]
    [ProducesResponseType(typeof(ImageAnalysisResponseDto), StatusCodes.Status200OK)]
    [ProducesResponseType(typeof(ImageAnalysisResponseDto), StatusCodes.Status400BadRequest)]
    public async Task<IActionResult> AnalyzeImage(IFormFile? file, CancellationToken cancellationToken)
    {
        var validation = _imageValidator.Validate(file);
        if (!validation.IsValid)
        {
            return BadRequest(new ImageAnalysisResponseDto
            {
                Success = false,
                ErrorMessage = validation.ErrorMessage
            });
        }

        try
        {
            await using var stream = file!.OpenReadStream();
            var result = await _imagePipeline.AnalyzeAsync(stream, cancellationToken);

            var response = MapToResponseDto(result);
            return Ok(response);
        }
        catch (ArgumentException argEx)
        {
            _logger.LogWarning(argEx, "Invalid image format received.");
            return BadRequest(new ImageAnalysisResponseDto
            {
                Success = false,
                ErrorMessage = $"Invalid image data: {argEx.Message}"
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Unexpected error processing uploaded image analysis request.");
            return StatusCode(StatusCodes.Status500InternalServerError, new ImageAnalysisResponseDto
            {
                Success = false,
                ErrorMessage = $"An error occurred while processing the image: {ex.Message}"
            });
        }
    }

    /// <summary>
    /// TAB 2: Upload a video file (.mp4, .mov, .avi, .mkv, .webm) for multi-person tracking and temporal attribute aggregation.
    /// Supports configurable frame sampling (1, 2, 3, 5 FPS). Guarantees temporary file deletion on complete/fail.
    /// </summary>
    [HttpPost("video")]
    [Consumes("multipart/form-data")]
    [ProducesResponseType(typeof(VideoAnalysisResponseDto), StatusCodes.Status200OK)]
    [ProducesResponseType(typeof(VideoAnalysisResponseDto), StatusCodes.Status400BadRequest)]
    public async Task<IActionResult> AnalyzeVideo(IFormFile? file, [FromForm] double analysisFps = 3.0, CancellationToken cancellationToken = default)
    {
        var validation = _videoValidator.Validate(file);
        if (!validation.IsValid)
        {
            return BadRequest(new VideoAnalysisResponseDto
            {
                Success = false,
                ErrorMessage = validation.ErrorMessage
            });
        }

        try
        {
            await using var stream = file!.OpenReadStream();
            var result = await _videoService.AnalyzeVideoAsync(stream, validation.Extension, analysisFps, cancellationToken);
            return Ok(result);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Unexpected error processing uploaded video analysis request.");
            return StatusCode(StatusCodes.Status500InternalServerError, new VideoAnalysisResponseDto
            {
                Success = false,
                ErrorMessage = $"Video analysis failed: {ex.Message}"
            });
        }
    }

    /// <summary>
    /// Standalone PAR Testing Endpoint (Developer Diagnostics):
    /// Directly accepts ONE cropped full-person image, runs PULC PAR ONNX inference, and returns raw attributes.
    /// </summary>
    [HttpPost("test-par")]
    [Consumes("multipart/form-data")]
    [ProducesResponseType(typeof(PersonAttributes), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    [ProducesResponseType(StatusCodes.Status500InternalServerError)]
    public async Task<IActionResult> TestPedestrianAttributes(IFormFile? file, CancellationToken cancellationToken)
    {
        var validation = _imageValidator.Validate(file);
        if (!validation.IsValid)
        {
            return BadRequest(new { Success = false, ErrorMessage = validation.ErrorMessage });
        }

        if (!_attributeService.IsModelLoaded)
        {
            return StatusCode(StatusCodes.Status503ServiceUnavailable, new
            {
                Success = false,
                ErrorMessage = "PULC PAR ONNX model is not loaded. Verify the model file exists."
            });
        }

        try
        {
            await using var memoryStream = new MemoryStream();
            await file!.CopyToAsync(memoryStream, cancellationToken);
            var imageBytes = memoryStream.ToArray();

            var sw = System.Diagnostics.Stopwatch.StartNew();
            var attributes = await _attributeService.RecognizeAttributesAsync(imageBytes, cancellationToken);
            sw.Stop();

            return Ok(new
            {
                Success = true,
                ElapsedMs = sw.Elapsed.TotalMilliseconds,
                ModelMetadata = new
                {
                    InputName = _attributeService.InputName,
                    OutputName = _attributeService.OutputName,
                    InputDimensions = _attributeService.InputDimensions,
                    InputType = _attributeService.InputElementType.Name,
                    OutputType = _attributeService.OutputElementType.Name
                },
                Attributes = attributes
            });
        }
        catch (ArgumentException argEx)
        {
            return BadRequest(new { Success = false, ErrorMessage = argEx.Message });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error executing standalone PAR test inference.");
            return StatusCode(StatusCodes.Status500InternalServerError, new
            {
                Success = false,
                ErrorMessage = $"PAR Inference failed: {ex.Message}"
            });
        }
    }

    /// <summary>
    /// Check the status and readiness of AI models and backend health.
    /// </summary>
    [HttpGet("health")]
    public IActionResult GetHealth()
    {
        return Ok(new
        {
            Status = "Online",
            YoloModelLoaded = _personDetector.IsModelLoaded,
            ParModelLoaded = _attributeRecognizer.IsModelLoaded
        });
    }

    private static ImageAnalysisResponseDto MapToResponseDto(ImageAnalysisResult result)
    {
        return new ImageAnalysisResponseDto
        {
            Success = true,
            ImageWidth = result.ImageWidth,
            ImageHeight = result.ImageHeight,
            TotalElapsedMs = result.Metrics.TotalElapsedMs,
            DetectionMs = result.Metrics.DetectionMs,
            AttributeRecognitionMs = result.Metrics.AttributeRecognitionMs,
            Persons = result.Persons.Select(p =>
            {
                var attrs = p.Attributes.Attributes;
                string appSex = attrs.GetValueOrDefault("Appearance")?.Value ?? "Unknown";
                float appConf = attrs.GetValueOrDefault("Appearance")?.Confidence ?? 0f;

                string upType = attrs.GetValueOrDefault("UpperClothing")?.Value ?? "Unknown";
                float upConf = attrs.GetValueOrDefault("UpperClothing")?.Confidence ?? 0f;

                string upColor = attrs.GetValueOrDefault("UpperColor")?.Value ?? "Unknown";
                float upColorConf = attrs.GetValueOrDefault("UpperColor")?.Confidence ?? 0f;

                string lowType = attrs.GetValueOrDefault("LowerClothing")?.Value ?? "Not Visible";
                float lowConf = attrs.GetValueOrDefault("LowerClothing")?.Confidence ?? 0f;

                string lowColor = attrs.GetValueOrDefault("LowerColor")?.Value ?? "Not Visible";
                float lowColorConf = attrs.GetValueOrDefault("LowerColor")?.Confidence ?? 0f;

                string shoes = attrs.GetValueOrDefault("Shoes")?.Value ?? "Not Visible";
                float shoesConf = attrs.GetValueOrDefault("Shoes")?.Confidence ?? 0f;

                string rawWatch = attrs.GetValueOrDefault("Watch")?.Value ?? "Not Detected";
                float watchConf = attrs.GetValueOrDefault("Watch")?.Confidence ?? 0f;
                string watch = NormalizeWatchStatus(rawWatch);
                bool isWatchDetected = watch.Equals("Detected", StringComparison.OrdinalIgnoreCase);

                string brand = attrs.GetValueOrDefault("Brand")?.Value ?? "BrandUnknown";
                float brandConf = attrs.GetValueOrDefault("Brand")?.Confidence ?? 0f;

                string upBrand = attrs.GetValueOrDefault("UpperBrand")?.Value ?? "Not Visible";
                float upBrandConf = attrs.GetValueOrDefault("UpperBrand")?.Confidence ?? 0f;

                string lowBrand = attrs.GetValueOrDefault("LowerBrand")?.Value ?? "Not Visible";
                float lowBrandConf = attrs.GetValueOrDefault("LowerBrand")?.Confidence ?? 0f;

                string watchBrand = isWatchDetected 
                    ? (attrs.GetValueOrDefault("WatchBrand")?.Value ?? "Not Detected")
                    : "Not Detected";
                if (watchBrand == "Model Unavailable" || watchBrand == "ModelUnavailable" || watchBrand == "Not Visible" || watchBrand == "Unknown" || watchBrand == "Unsupported Region")
                {
                    watchBrand = "Not Detected";
                }
                float watchBrandConf = isWatchDetected ? (attrs.GetValueOrDefault("WatchBrand")?.Confidence ?? 0f) : 0f;

                string shoeBrand = attrs.GetValueOrDefault("ShoeBrand")?.Value ?? "Not Visible";
                float shoeBrandConf = attrs.GetValueOrDefault("ShoeBrand")?.Confidence ?? 0f;

                string bagBrand = attrs.GetValueOrDefault("BagBrand")?.Value ?? "Not Visible";
                float bagBrandConf = attrs.GetValueOrDefault("BagBrand")?.Confidence ?? 0f;

                var personBrandDto = new PersonBrandDto
                {
                    UpperBrand = new BrandDto { Region = "Upper", BrandName = upBrand, Similarity = upBrandConf, State = DetermineBrandState(upBrand) },
                    LowerBrand = new BrandDto { Region = "Lower", BrandName = lowBrand, Similarity = lowBrandConf, State = DetermineBrandState(lowBrand) },
                    WatchBrand = new BrandDto { Region = "Watch", BrandName = watchBrand, Similarity = watchBrandConf, State = DetermineBrandState(watchBrand) },
                    ShoeBrand = new BrandDto { Region = "Shoes", BrandName = shoeBrand, Similarity = shoeBrandConf, State = DetermineBrandState(shoeBrand) },
                    BagBrand = new BrandDto { Region = "Bag", BrandName = bagBrand, Similarity = bagBrandConf, State = DetermineBrandState(bagBrand) },
                    TopDetectedBrand = brand,
                    TopBrandSimilarity = brandConf,
                    HasAnyAcceptedBrand = brand != "BrandUnknown" && brand != "Unknown" && brand != "Not Visible" && !brand.StartsWith("No Logo") && !brand.StartsWith("Insufficient") && !brand.StartsWith("Not Detected")
                };

                return new DetectedPersonDto
                {
                    PersonId = p.PersonId,
                    PersonIndex = p.PersonId,
                    DetectionConfidence = p.Detection.Confidence,
                    BoundingBox = new BoundingBoxDto(
                        p.Detection.Box.X,
                        p.Detection.Box.Y,
                        p.Detection.Box.Width,
                        p.Detection.Box.Height),
                    CropWidth = p.CropWidth,
                    CropHeight = p.CropHeight,
                    CropDataUrl = p.CropBase64,

                    AppearanceSex = appSex,
                    AppearanceConfidence = appConf,
                    AppearanceState = "SingleImage",

                    UpperType = upType,
                    UpperTypeConfidence = upConf,
                    UpperColor = upColor,
                    UpperColorConfidence = upColorConf,

                    LowerType = lowType,
                    LowerTypeConfidence = lowConf,
                    LowerColor = lowColor,
                    LowerColorConfidence = lowColorConf,

                    ShoesType = shoes,
                    ShoesConfidence = shoesConf,

                    WatchDetected = isWatchDetected,
                    WatchConfidence = watchConf,
                    WatchDetails = watch,

                    Brands = personBrandDto,
                    BrandName = brand,
                    BrandConfidence = brandConf,
                    BrandState = personBrandDto.HasAnyAcceptedBrand ? "BrandCandidate" : "BrandUnknown",

                    Attributes = p.Attributes.Attributes.Select(attr => new AttributeItemDto(
                        attr.Key,
                        attr.Value.Value,
                        attr.Value.Confidence)).ToList()
                };
            }).ToList()
        };
    }

    private static string DetermineBrandState(string brandName) => brandName switch
    {
        "Not Visible" => "NotVisible",
        "Not Detected" => "NoLogoCandidate",
        "Insufficient Visual Evidence" => "InsufficientVisualEvidence",
        "No Logo Candidate" => "NoLogoCandidate",
        "Model Unavailable" => "ModelUnavailable",
        "Unsupported Region" => "UnsupportedRegion",
        "BrandUnknown" or "Unknown" => "BrandUnknown",
        _ => "BrandCandidate"
    };

    private static string NormalizeWatchStatus(string? val)
    {
        if (string.IsNullOrWhiteSpace(val)) return "Not Detected";
        string v = val.Trim();
        if (v.Equals("Detected", StringComparison.OrdinalIgnoreCase) ||
            v.Equals("WatchDetected", StringComparison.OrdinalIgnoreCase) ||
            v.Equals("Watch Detected", StringComparison.OrdinalIgnoreCase))
        {
            return "Detected";
        }
        if (v.Equals("InsufficientVisualEvidence", StringComparison.OrdinalIgnoreCase) ||
            v.Equals("Insufficient Evidence", StringComparison.OrdinalIgnoreCase))
        {
            return "Insufficient Evidence";
        }
        if (v.Equals("NotVisible", StringComparison.OrdinalIgnoreCase) ||
            v.Equals("Not Visible", StringComparison.OrdinalIgnoreCase))
        {
            return "Not Visible";
        }
        if (v.Equals("ModelUnavailable", StringComparison.OrdinalIgnoreCase) ||
            v.Equals("Model Unavailable", StringComparison.OrdinalIgnoreCase))
        {
            return "Not Detected";
        }
        if (v.Equals("NoWatchDetected", StringComparison.OrdinalIgnoreCase) ||
            v.Equals("No Watch", StringComparison.OrdinalIgnoreCase) ||
            v.Equals("No Watch Detected", StringComparison.OrdinalIgnoreCase) ||
            v.Equals("NotDetected", StringComparison.OrdinalIgnoreCase) ||
            v.Equals("Not Detected", StringComparison.OrdinalIgnoreCase) ||
            v.Equals("Unknown", StringComparison.OrdinalIgnoreCase))
        {
            return "Not Detected";
        }
        return v;
    }

    /// <summary>
    /// Runs the comprehensive empirical pipeline validation test suite on real images and video.
    /// </summary>
    [HttpGet("validate-pipeline")]
    public async Task<IActionResult> RunPipelineValidation(CancellationToken cancellationToken)
    {
        var report = await VisionAttributeAI.Tests.EmpiricalPipelineValidation.ExecuteValidationAsync(
            _unifiedAnalysisService,
            _personDetector,
            _poseService,
            _candidateValidator,
            _cropper,
            _aggregator,
            _options);

        return Ok(report);
    }

    /// <summary>
    /// Runs the Brand Recognition unit and temporal state machine test suite.
    /// </summary>
    [HttpGet("brand-test")]
    public IActionResult RunBrandTestSuite()
    {
        var results = VisionAttributeAI.Tests.BrandRecognitionTestSuite.RunAllTests();
        return Ok(new
        {
            Success = results.All(r => r.Passed),
            TotalTests = results.Count,
            PassedTests = results.Count(r => r.Passed),
            FailedTests = results.Count(r => !r.Passed),
            Results = results
        });
    }
}
