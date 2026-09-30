using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Options;
using VisionAttributeAI.DTOs.LiveCamera;
using VisionAttributeAI.Models.Tracking;
using VisionAttributeAI.Services.LiveCamera;
using VisionAttributeAI.Services.Validation;

namespace VisionAttributeAI.Controllers;

[ApiController]
[Route("api/live")]
[Route("api/[controller]")]
public class LiveAnalysisController : ControllerBase
{
    private readonly ILiveCameraAnalysisService _liveService;
    private readonly IImageValidator _imageValidator;
    private readonly VisionAttributeAI.Services.Warmup.IModelWarmupService _warmupService;
    private readonly LiveTrackingOptions _options;
    private readonly ILogger<LiveAnalysisController> _logger;

    public LiveAnalysisController(
        ILiveCameraAnalysisService liveService,
        IImageValidator imageValidator,
        VisionAttributeAI.Services.Warmup.IModelWarmupService warmupService,
        IOptions<LiveTrackingOptions> options,
        ILogger<LiveAnalysisController> logger)
    {
        _liveService = liveService;
        _imageValidator = imageValidator;
        _warmupService = warmupService;
        _options = options.Value;
        _logger = logger;
    }

    /// <summary>
    /// Returns current AI model readiness and pre-warm status.
    /// </summary>
    [HttpGet("status")]
    public IActionResult GetStatus()
    {
        return Ok(new
        {
            IsReady = _warmupService.IsReady,
            Status = _warmupService.Status.ToString(),
            TotalWarmupMs = _warmupService.TotalWarmupMs,
            ErrorMessage = _warmupService.ErrorMessage
        });
    }

    /// <summary>
    /// Processes a single live camera frame in-memory:
    /// Ingestion -> YOLO Detection -> Multi-Person Tracking -> Quality Filter -> Fashion-CLIP -> Temporal Aggregation.
    /// Stores ZERO permanent files on disk.
    /// </summary>
    [HttpPost("process-frame")]
    [Consumes("multipart/form-data")]
    [ProducesResponseType(typeof(LiveFrameResponseDto), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    [ProducesResponseType(StatusCodes.Status500InternalServerError)]
    public async Task<IActionResult> ProcessFrame(IFormFile? file, [FromForm] long frameIndex = 0, CancellationToken cancellationToken = default)
    {
        var validation = _imageValidator.Validate(file);
        if (!validation.IsValid)
        {
            return BadRequest(new LiveFrameResponseDto
            {
                Success = false,
                ErrorMessage = validation.ErrorMessage
            });
        }

        try
        {
            // Read frame stream into memory (never saved to disk)
            await using var memoryStream = new MemoryStream();
            await file!.CopyToAsync(memoryStream, cancellationToken);
            var frameBytes = memoryStream.ToArray();

            var response = await _liveService.ProcessLiveFrameAsync(frameBytes, frameIndex, cancellationToken);
            return Ok(response);
        }
        catch (ArgumentException argEx)
        {
            return BadRequest(new LiveFrameResponseDto
            {
                Success = false,
                ErrorMessage = argEx.Message
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error executing live camera frame analysis on frame #{FrameIndex}.", frameIndex);
            return StatusCode(StatusCodes.Status500InternalServerError, new LiveFrameResponseDto
            {
                Success = false,
                ErrorMessage = $"Live frame processing failed: {ex.Message}"
            });
        }
    }

    /// <summary>
    /// Resets all active multi-person tracks and clears rolling aggregation histories (e.g. on Stop Camera).
    /// </summary>
    [HttpPost("reset")]
    [ProducesResponseType(StatusCodes.Status200OK)]
    public IActionResult ResetTracking()
    {
        _liveService.Reset();
        return Ok(new { Success = true, Message = "Tracking and aggregation histories reset." });
    }

    /// <summary>
    /// Returns the most recently processed live frame state and active tracks for remote monitoring clients (e.g. PC dashboard).
    /// </summary>
    [HttpGet("latest-state")]
    [ProducesResponseType(typeof(LiveFrameResponseDto), StatusCodes.Status200OK)]
    public IActionResult GetLatestState()
    {
        var state = _liveService.GetLatestState();
        if (state == null)
        {
            return Ok(new LiveFrameResponseDto
            {
                Success = true,
                FrameIndex = 0,
                Persons = new List<TrackedPersonDto>()
            });
        }

        return Ok(state);
    }

    /// <summary>
    /// Returns current live camera configuration parameters.
    /// </summary>
    [HttpGet("config")]
    public IActionResult GetConfig()
    {
        return Ok(new
        {
            TargetFps = _options.TargetFps,
            MinValidObservations = _options.MinValidObservations,
            HistoryWindowSize = _options.HistoryWindowSize,
            MinDetectionConfidence = _options.MinDetectionConfidence,
            MinCropWidth = _options.MinCropWidth,
            MinCropHeight = _options.MinCropHeight,
            MinCropArea = _options.MinCropArea,
            MinSharpnessVariance = _options.MinSharpnessVariance,
            AmbiguityMarginThreshold = _options.AmbiguityMarginThreshold,
            TrackTimeoutSeconds = _options.TrackTimeoutSeconds
        });
    }
}
