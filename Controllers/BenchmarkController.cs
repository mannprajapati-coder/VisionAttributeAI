using Microsoft.AspNetCore.Mvc;
using VisionAttributeAI.Models.Benchmark;
using VisionAttributeAI.Services.Benchmark;
using VisionAttributeAI.Services.Tracking;

namespace VisionAttributeAI.Controllers;

[ApiController]
[Route("api/benchmark")]
public class BenchmarkController : ControllerBase
{
    private readonly IBenchmarkService _benchmarkService;
    private readonly IPersonTracker _tracker;
    private readonly ILogger<BenchmarkController> _logger;

    public BenchmarkController(
        IBenchmarkService benchmarkService,
        IPersonTracker tracker,
        ILogger<BenchmarkController> logger)
    {
        _benchmarkService = benchmarkService;
        _tracker = tracker;
        _logger = logger;
    }

    [HttpPost("ground-truth")]
    public IActionResult SetGroundTruth([FromBody] PersonGroundTruth groundTruth)
    {
        if (groundTruth.PersonId <= 0)
        {
            return BadRequest(new { Success = false, Message = "Invalid PersonId." });
        }

        _benchmarkService.SetGroundTruth(groundTruth);
        return Ok(new { Success = true, Message = $"Ground truth set for Person #{groundTruth.PersonId}." });
    }

    [HttpGet("ground-truth")]
    public IActionResult GetAllGroundTruths()
    {
        return Ok(_benchmarkService.GetAllGroundTruths());
    }

    [HttpDelete("ground-truth/{personId}")]
    public IActionResult RemoveGroundTruth(int personId)
    {
        _benchmarkService.RemoveGroundTruth(personId);
        return Ok(new { Success = true, Message = $"Ground truth removed for Person #{personId}." });
    }

    [HttpGet("evaluation")]
    public IActionResult GetEvaluationReport()
    {
        var activeTracks = _tracker.GetActiveTracks();
        var report = _benchmarkService.EvaluateTracks(activeTracks);
        return Ok(report);
    }

    [HttpPost("reset")]
    public IActionResult Reset()
    {
        _benchmarkService.Reset();
        return Ok(new { Success = true, Message = "Benchmark ground truths reset." });
    }
}
