using System.Diagnostics;
using Microsoft.Extensions.Options;
using OpenCvSharp;
using VisionAttributeAI.DTOs;
using VisionAttributeAI.DTOs.LiveCamera;
using VisionAttributeAI.DTOs.Video;
using VisionAttributeAI.Models.AI;
using VisionAttributeAI.Models.Pose;
using VisionAttributeAI.Models.Tracking;
using VisionAttributeAI.Services.Aggregation;
using VisionAttributeAI.Services.Detection;
using VisionAttributeAI.Services.PersonAnalysis;
using VisionAttributeAI.Services.Pose;
using VisionAttributeAI.Services.Tracking;
using VisionAttributeAI.Services.Validation;

namespace VisionAttributeAI.Services.Video;

/// <summary>
/// Implements sequential multi-person video analysis with configurable FPS sampling,
/// tracking, per-person observation histories, per-person Top-5 best-frame pools,
/// per-person weighted temporal aggregation, and zero permanent disk persistence.
/// </summary>
public class VideoAnalysisService : IVideoAnalysisService
{
    private readonly IPersonDetector _personDetector;
    private readonly IPoseEstimationService _poseService;
    private readonly IUnifiedPersonAnalysisService _personAnalysisService;
    private readonly ITemporalAppearanceAggregator _aggregator;
    private readonly IPersonCandidateValidator _candidateValidator;
    private readonly IOptions<LiveTrackingOptions> _trackingOptions;
    private readonly ILoggerFactory _loggerFactory;
    private readonly ILogger<VideoAnalysisService> _logger;

    public VideoAnalysisService(
        IPersonDetector personDetector,
        IPoseEstimationService poseService,
        IUnifiedPersonAnalysisService personAnalysisService,
        ITemporalAppearanceAggregator aggregator,
        IPersonCandidateValidator candidateValidator,
        IOptions<LiveTrackingOptions> trackingOptions,
        ILoggerFactory loggerFactory,
        ILogger<VideoAnalysisService> logger)
    {
        _personDetector = personDetector;
        _poseService = poseService;
        _personAnalysisService = personAnalysisService;
        _aggregator = aggregator;
        _candidateValidator = candidateValidator;
        _trackingOptions = trackingOptions;
        _loggerFactory = loggerFactory;
        _logger = logger;
    }

    public async Task<VideoAnalysisResponseDto> AnalyzeVideoAsync(
        Stream videoStream,
        string fileExtension,
        double targetAnalysisFps = 3.0,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(videoStream);
        if (targetAnalysisFps <= 0) targetAnalysisFps = 3.0;

        string tempFilePath = Path.Combine(Path.GetTempPath(), $"video_analysis_{Guid.NewGuid():N}{fileExtension}");
        var totalStopwatch = Stopwatch.StartNew();
        double totalDetectionMs = 0;
        double totalAttributeMs = 0;

        try
        {
            // 1. Write stream to controlled temporary file for OpenCV VideoCapture decoding
            await using (var fileStream = new FileStream(tempFilePath, FileMode.Create, FileAccess.Write, FileShare.None))
            {
                await videoStream.CopyToAsync(fileStream, cancellationToken);
            }

            // 2. Open VideoCapture
            using var capture = new VideoCapture(tempFilePath);
            if (!capture.IsOpened())
            {
                throw new InvalidOperationException("Failed to open uploaded video stream with OpenCV VideoCapture.");
            }

            double sourceFps = capture.Fps;
            if (sourceFps <= 1.0 || double.IsNaN(sourceFps)) sourceFps = 30.0; // Fallback

            int totalFrames = capture.FrameCount;
            double durationSeconds = totalFrames > 0 ? totalFrames / sourceFps : 0;

            int frameStep = Math.Max(1, (int)Math.Round(sourceFps / targetAnalysisFps));
            double effectiveAnalysisFps = sourceFps / frameStep;

            _logger.LogInformation(
                "Starting video analysis: {TotalFrames} frames, Source FPS: {SourceFps:F1}, Target Analysis FPS: {TargetFps:F1} (Step: {Step})",
                totalFrames, sourceFps, targetAnalysisFps, frameStep);

            // 3. Isolated multi-person tracker for this video
            var trackerLogger = _loggerFactory.CreateLogger<IoUPersonTracker>();
            var videoTracker = new IoUPersonTracker(_trackingOptions, trackerLogger);

            // Person tracking metadata: PersonId -> (State, FirstSeenSec, LastSeenSec, BestCropBase64, BestCropQuality)
            var allObservedPersons = new Dictionary<int, TrackMetadata>();

            int currentFrameIndex = 0;
            int sampledFramesAnalyzed = 0;

            using var frameMat = new Mat();

            while (capture.Read(frameMat))
            {
                cancellationToken.ThrowIfCancellationRequested();

                if (frameMat.Empty()) break;

                if (currentFrameIndex % frameStep == 0)
                {
                    sampledFramesAnalyzed++;
                    double currentTimestampSeconds = currentFrameIndex / sourceFps;

                    // A. YOLO Person Detection
                    var detSw = Stopwatch.StartNew();
                    var detections = await _personDetector.DetectPersonsAsync(frameMat, cancellationToken);
                    detSw.Stop();
                    totalDetectionMs += detSw.Elapsed.TotalMilliseconds;

                    // B. YOLO Pose Estimation on Detections
                    var poseResults = await _poseService.EstimatePoseAsync(frameMat, detections, cancellationToken);

                    // C. Multi-Signal Candidate Validation Gate
                    var validatedCandidates = _candidateValidator.FilterValidCandidates(detections, poseResults, frameMat.Width, frameMat.Height);
                    var validDetections = validatedCandidates.Select(c => c.Detection).ToList();
                    var validPoses = validatedCandidates.Select(c => c.Pose).ToList();

                    // D. Explicit 1-to-1 Spatial Multi-Person Tracking & Association
                    var frameObservations = videoTracker.AssociateAndTrack(
                        validDetections,
                        validPoses,
                        isOfflineVideo: true,
                        frameIndex: currentFrameIndex,
                        timestampSec: currentTimestampSeconds);

                    // E. Analyze EACH CONFIRMED person independently (Tentative tracks are evaluated silently)
                    var attrSw = Stopwatch.StartNew();

                    foreach (var frameObs in frameObservations)
                    {
                        var track = frameObs.TrackState;

                        // Only confirmed tracks are assigned public PersonIds and processed for attributes/reporting
                        if (!frameObs.IsConfirmed || track.PersonId <= 0)
                        {
                            continue;
                        }

                        var pose = frameObs.Pose;

                        track.LatestVisibility = pose.Visibility;
                        track.LatestRegions = pose.Regions;
                        track.LatestKeypoints = pose.Keypoints;

                        // Unified Person Analysis
                        var analysisOutput = await _personAnalysisService.AnalyzePersonAsync(
                            frameMat,
                            frameObs.BoundingBox,
                            frameObs.Detection.Confidence,
                            track.PersonId,
                            currentFrameIndex,
                            pose,
                            generateBase64Crop: true,
                            brandState: track.BrandState,
                            isVideoMode: true,
                            possibleIdSwitch: frameObs.PossibleIdSwitch || track.PossibleIdSwitch,
                            cancellationToken: cancellationToken);

                        track.LatestQualityScores = analysisOutput.QualityScores;

                        // Register track in global observed persons dictionary
                        if (!allObservedPersons.TryGetValue(track.PersonId, out var meta))
                        {
                            meta = new TrackMetadata
                            {
                                TrackState = track,
                                FirstSeenSeconds = currentTimestampSeconds,
                                LastSeenSeconds = currentTimestampSeconds,
                                BestCropPreview = analysisOutput.CropBase64,
                                BestCropQuality = analysisOutput.GlobalQuality.QualityScore
                            };
                            allObservedPersons[track.PersonId] = meta;
                        }
                        else
                        {
                            meta.LastSeenSeconds = currentTimestampSeconds;

                            // PROTECT BEST CROP: Only update when AssociationAccepted, NOT PossibleIdSwitch, and Higher Quality
                            if (frameObs.AssociationAccepted && !frameObs.PossibleIdSwitch && !track.PossibleIdSwitch)
                            {
                                if (analysisOutput.GlobalQuality.QualityScore > meta.BestCropQuality && !string.IsNullOrEmpty(analysisOutput.CropBase64))
                                {
                                    meta.BestCropPreview = analysisOutput.CropBase64;
                                    meta.BestCropQuality = analysisOutput.GlobalQuality.QualityScore;
                                }
                            }
                        }

                        // Add observation & run temporal aggregation ONLY if association was accepted without ID-switch risk
                        if (frameObs.AssociationAccepted && !frameObs.PossibleIdSwitch && !track.PossibleIdSwitch)
                        {
                            _aggregator.Aggregate(track, analysisOutput.Observation);
                        }
                    }

                    attrSw.Stop();
                    totalAttributeMs += attrSw.Elapsed.TotalMilliseconds;
                }

                currentFrameIndex++;
            }

            totalStopwatch.Stop();

            // 4. Build Final Multi-Person Summary Report for all persons observed in the video
            var personResults = new List<VideoPersonResultDto>();

            foreach (var kvp in allObservedPersons.OrderBy(k => k.Key))
            {
                var personId = kvp.Key;
                var meta = kvp.Value;
                var track = meta.TrackState;

                // Finalize track lifecycle states before generating report
                _aggregator.FinalizeTrack(track);

                personResults.Add(new VideoPersonResultDto
                {
                    PersonId = personId,
                    FirstSeenTimestamp = FormatTimestamp(meta.FirstSeenSeconds),
                    LastSeenTimestamp = FormatTimestamp(meta.LastSeenSeconds),
                    FirstSeenSeconds = meta.FirstSeenSeconds,
                    LastSeenSeconds = meta.LastSeenSeconds,
                    FramesObserved = track.TotalValidObservations,
                    TrackStatus = (meta.LastSeenSeconds >= (durationSeconds - 1.0)) ? "ActiveUntilEnd" : "Finalized",
                    BestCropPreviewUrl = meta.BestCropPreview,

                    // Attributes
                    AppearanceSex = track.AppearanceSex,
                    AppearanceConfidence = track.AppearanceConfidence,
                    AppearanceState = track.AppearanceState.ToString(),

                    UpperType = track.UpperType,
                    UpperTypeConfidence = track.UpperTypeConfidence,
                    UpperTypeState = track.UpperTypeState.ToString(),

                    UpperColor = track.UpperColor,
                    UpperColorConfidence = track.UpperColorConfidence,
                    UpperColorState = track.UpperColorState.ToString(),

                    LowerType = track.LowerType,
                    LowerTypeConfidence = track.LowerTypeConfidence,
                    LowerTypeState = track.LowerTypeState.ToString(),

                    LowerColor = track.LowerColor,
                    LowerColorConfidence = track.LowerColorConfidence,
                    LowerColorState = track.LowerColorState.ToString(),

                    ShoesType = track.ShoesType,
                    ShoesConfidence = track.ShoesConfidence,
                    ShoesState = track.ShoesState.ToString(),

                    WatchDetected = track.WatchStatus.Equals("Detected", StringComparison.OrdinalIgnoreCase),
                    WatchConfidence = track.WatchConfidence,
                    WatchDetails = track.WatchStatus,

                    // 6. Brand Recognition
                    Brands = new PersonBrandDto
                    {
                        UpperBrand = new BrandDto { Region = "Upper", BrandName = track.BrandState.UpperBrand.BrandName, Similarity = track.BrandState.UpperBrand.Similarity, Margin = track.BrandState.UpperBrand.Margin, State = track.BrandState.UpperBrand.State.ToString(), IsStable = track.BrandState.UpperBrand.IsStable, ObservationCount = track.BrandState.UpperBrand.ObservationCount, RunnerUpBrand = track.BrandState.UpperBrand.RunnerUpBrand },
                        LowerBrand = new BrandDto { Region = "Lower", BrandName = track.BrandState.LowerBrand.BrandName, Similarity = track.BrandState.LowerBrand.Similarity, Margin = track.BrandState.LowerBrand.Margin, State = track.BrandState.LowerBrand.State.ToString(), IsStable = track.BrandState.LowerBrand.IsStable, ObservationCount = track.BrandState.LowerBrand.ObservationCount, RunnerUpBrand = track.BrandState.LowerBrand.RunnerUpBrand },
                        WatchBrand = new BrandDto { Region = "Watch", BrandName = track.BrandState.WatchBrand.BrandName, Similarity = track.BrandState.WatchBrand.Similarity, Margin = track.BrandState.WatchBrand.Margin, State = track.BrandState.WatchBrand.State.ToString(), IsStable = track.BrandState.WatchBrand.IsStable, ObservationCount = track.BrandState.WatchBrand.ObservationCount, RunnerUpBrand = track.BrandState.WatchBrand.RunnerUpBrand },
                        ShoeBrand = new BrandDto { Region = "Shoes", BrandName = track.BrandState.ShoeBrand.BrandName, Similarity = track.BrandState.ShoeBrand.Similarity, Margin = track.BrandState.ShoeBrand.Margin, State = track.BrandState.ShoeBrand.State.ToString(), IsStable = track.BrandState.ShoeBrand.IsStable, ObservationCount = track.BrandState.ShoeBrand.ObservationCount, RunnerUpBrand = track.BrandState.ShoeBrand.RunnerUpBrand },
                        BagBrand = new BrandDto { Region = "Bag", BrandName = track.BrandState.BagBrand.BrandName, Similarity = track.BrandState.BagBrand.Similarity, Margin = track.BrandState.BagBrand.Margin, State = track.BrandState.BagBrand.State.ToString(), IsStable = track.BrandState.BagBrand.IsStable, ObservationCount = track.BrandState.BagBrand.ObservationCount, RunnerUpBrand = track.BrandState.BagBrand.RunnerUpBrand },
                        TopDetectedBrand = track.BrandState.ToPersonBrandResult().TopDetectedBrand?.BrandName ?? "BrandUnknown",
                        TopBrandSimilarity = track.BrandState.ToPersonBrandResult().TopDetectedBrand?.Similarity ?? 0f,
                        HasAnyAcceptedBrand = track.BrandState.ToPersonBrandResult().HasAnyAcceptedBrand,
                        HasAnyStableBrand = track.BrandState.ToPersonBrandResult().HasAnyStableBrand
                    },
                    BrandName = track.BrandState.ToPersonBrandResult().TopDetectedBrand?.BrandName ?? "BrandUnknown",
                    BrandConfidence = track.BrandState.ToPersonBrandResult().TopDetectedBrand?.Similarity ?? 0f,
                    BrandState = (track.BrandState.ToPersonBrandResult().TopDetectedBrand?.State.ToString()) ?? "Analyzing",
                    IsBrandStable = track.BrandState.ToPersonBrandResult().HasAnyStableBrand,

                    // Tracking Diagnostics
                    TrackingDiagnostics = new TrackingDiagnosticsDto
                    {
                        PersonId = track.PersonId,
                        TrackAgeFrames = track.TrackAgeFrames,
                        LastMatchedIoU = track.LastMatchedIoU,
                        CenterDisplacement = track.CenterDisplacement,
                        MissedFrames = track.MissedFrames,
                        PossibleIdSwitch = track.PossibleIdSwitch,
                        PossibleIdSwitchReason = track.PossibleIdSwitchReason
                    },

                    // Best frame pools
                    BestAppearanceFrames = track.BestAppearanceFrames.Select(MapBestFrame).ToList(),
                    BestUpperFrames = track.BestUpperFrames.Select(MapBestFrame).ToList(),
                    BestLowerFrames = track.BestLowerFrames.Select(MapBestFrame).ToList(),
                    BestShoesFrames = track.BestShoesFrames.Select(MapBestFrame).ToList(),
                    BestWristFrames = track.BestWristFrames.Select(MapBestFrame).ToList()
                });
            }

            var summary = new VideoSummaryDto
            {
                TotalUniquePersons = personResults.Count,
                VideoDurationSeconds = durationSeconds,
                SourceFps = sourceFps,
                AnalysisFps = effectiveAnalysisFps,
                TotalVideoFrames = totalFrames,
                SampledFramesAnalyzed = sampledFramesAnalyzed,
                TotalElapsedMs = totalStopwatch.Elapsed.TotalMilliseconds,
                DetectionMs = totalDetectionMs,
                AttributeMs = totalAttributeMs
            };

            return new VideoAnalysisResponseDto
            {
                Success = true,
                Summary = summary,
                Persons = personResults
            };
        }
        finally
        {
            // Guaranteed cleanup of temporary upload file
            try
            {
                if (File.Exists(tempFilePath))
                {
                    File.Delete(tempFilePath);
                    _logger.LogDebug("Deleted temporary video analysis file: {Path}", tempFilePath);
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to clean up temporary video file {Path}", tempFilePath);
            }
        }
    }

    private static string FormatTimestamp(double seconds)
    {
        var ts = TimeSpan.FromSeconds(seconds);
        return $"{ts.Minutes:D2}:{ts.Seconds:D2}.{ts.Milliseconds / 100:D1}";
    }

    private static BestFrameItemDto MapBestFrame(BestFrameRecord r) => new()
    {
        FrameIndex = r.FrameIndex,
        QualityScore = r.QualityScore,
        Prediction = r.Prediction,
        Confidence = r.Confidence,
        Details = r.Details
    };

    private class TrackMetadata
    {
        public required TrackedPersonState TrackState { get; init; }
        public double FirstSeenSeconds { get; set; }
        public double LastSeenSeconds { get; set; }
        public string? BestCropPreview { get; set; }
        public float BestCropQuality { get; set; }
    }
}
