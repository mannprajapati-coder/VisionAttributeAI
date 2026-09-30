using System.Diagnostics;
using Microsoft.Extensions.Options;
using OpenCvSharp;
using VisionAttributeAI.DTOs;
using VisionAttributeAI.DTOs.LiveCamera;
using VisionAttributeAI.Models.AI;
using VisionAttributeAI.Models.Pose;
using VisionAttributeAI.Models.Tracking;
using VisionAttributeAI.Services.Aggregation;
using VisionAttributeAI.Services.Detection;
using VisionAttributeAI.Services.PersonAnalysis;
using VisionAttributeAI.Services.Pose;
using VisionAttributeAI.Services.Tracking;
using VisionAttributeAI.Services.Validation;

namespace VisionAttributeAI.Services.LiveCamera;

/// <summary>
/// Orchestrates the complete in-memory multi-frame + body-region live camera analysis pipeline:
/// Ingestion -> YOLO Detection -> Multi-Signal Candidate Validation -> Tracking -> Pose & Orientation -> Body Region Crops ->
/// Attribute-Specific Quality Gating -> Multi-Attribute Classification -> Weighted Aggregation.
/// Guarantees ZERO permanent image storage on disk.
/// </summary>
public class LiveCameraAnalysisService : ILiveCameraAnalysisService
{
    private readonly IPersonDetector _personDetector;
    private readonly IPoseEstimationService _poseService;
    private readonly IUnifiedPersonAnalysisService _personAnalysisService;
    private readonly IPersonTracker _personTracker;
    private readonly ICameraMotionEstimator _cameraMotionEstimator;
    private readonly ITemporalAppearanceAggregator _aggregator;
    private readonly IPersonCandidateValidator _candidateValidator;
    private readonly PersonAnalysisOptions _options;
    private readonly ILogger<LiveCameraAnalysisService> _logger;

    private LiveFrameResponseDto? _latestState;
    private readonly object _stateLock = new();
    private Mat? _prevFrameMat;
    private readonly object _frameLock = new();

    public LiveCameraAnalysisService(
        IPersonDetector personDetector,
        IPoseEstimationService poseService,
        IUnifiedPersonAnalysisService personAnalysisService,
        IPersonTracker personTracker,
        ICameraMotionEstimator cameraMotionEstimator,
        ITemporalAppearanceAggregator aggregator,
        IPersonCandidateValidator candidateValidator,
        IOptions<PersonAnalysisOptions> options,
        ILogger<LiveCameraAnalysisService> logger)
    {
        _personDetector = personDetector;
        _poseService = poseService;
        _personAnalysisService = personAnalysisService;
        _personTracker = personTracker;
        _cameraMotionEstimator = cameraMotionEstimator;
        _aggregator = aggregator;
        _candidateValidator = candidateValidator;
        _options = options.Value;
        _logger = logger;
    }

    public async Task<LiveFrameResponseDto> ProcessLiveFrameAsync(byte[] frameBytes, long frameIndex, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(frameBytes);

        var totalStopwatch = Stopwatch.StartNew();
        var detectStopwatch = Stopwatch.StartNew();

        // 1. In-memory decode into OpenCV Mat (NEVER written to disk)
        using var frameMat = Cv2.ImDecode(frameBytes, ImreadModes.Color);
        if (frameMat == null || frameMat.Empty())
        {
            throw new ArgumentException("Failed to decode camera frame bytes in memory.");
        }

        int imgWidth = frameMat.Width;
        int imgHeight = frameMat.Height;

        // 2. YOLO Person Detection
        var detections = await _personDetector.DetectPersonsAsync(frameMat, cancellationToken);
        detectStopwatch.Stop();
        double detectMs = detectStopwatch.Elapsed.TotalMilliseconds;

        // 3. Pose, Orientation & Keypoint Body Region Analysis
        var poseStopwatch = Stopwatch.StartNew();
        var poseResults = await _poseService.EstimatePoseAsync(frameMat, detections, cancellationToken);
        poseStopwatch.Stop();

        // 4. Multi-Signal Person Candidate Validation Gate
        var validatedCandidates = _candidateValidator.FilterValidCandidates(detections, poseResults, imgWidth, imgHeight);
        var validDetections = validatedCandidates.Select(c => c.Detection).ToList();
        var validPoses = validatedCandidates.Select(c => c.Pose).ToList();

        // 5. Global Camera Motion Compensation (CMC) Estimation
        CameraMotionResult cameraMotion = CameraMotionResult.Stable();
        lock (_frameLock)
        {
            if (_prevFrameMat != null && !_prevFrameMat.Empty())
            {
                cameraMotion = _cameraMotionEstimator.EstimateMotion(_prevFrameMat, frameMat);
            }
            _prevFrameMat?.Dispose();
            _prevFrameMat = frameMat.Clone();
        }

        // 6. Multi-Person Tracking with Explicit 1-to-1 Spatial Binding & CMC Compensation
        var frameObservations = _personTracker.AssociateAndTrack(
            validDetections,
            validPoses,
            isOfflineVideo: false,
            frameIndex: frameIndex,
            timestampSec: totalStopwatch.Elapsed.TotalSeconds,
            cameraMotion: cameraMotion);

        // 7. Body-Region Cropping & Multi-Attribute Analysis (Only for Confirmed Tracks)
        var attrStopwatch = Stopwatch.StartNew();
        var personDtos = new List<TrackedPersonDto>();

        foreach (var frameObs in frameObservations)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var track = frameObs.TrackState;

            // Tentative tracks are tracked silently until promoted to Confirmed
            if (!frameObs.IsConfirmed || track.PersonId <= 0)
            {
                continue;
            }

            var pose = frameObs.Pose;

            track.LatestVisibility = pose.Visibility;
            track.LatestRegions = pose.Regions;
            track.LatestKeypoints = pose.Keypoints;

            // Shared Unified Person Analysis Engine
            var analysisOutput = await _personAnalysisService.AnalyzePersonAsync(
                frameMat,
                frameObs.BoundingBox,
                frameObs.Detection.Confidence,
                track.PersonId,
                frameIndex,
                pose,
                generateBase64Crop: false,
                brandState: track.BrandState,
                isVideoMode: true,
                possibleIdSwitch: frameObs.PossibleIdSwitch || track.PossibleIdSwitch,
                cancellationToken: cancellationToken);

            track.LatestQualityScores = analysisOutput.QualityScores;

            TrackedPersonDebugDto debugDto;
            var obs = analysisOutput.Observation;
            var globalQuality = analysisOutput.GlobalQuality;
            var attrQuality = analysisOutput.QualityScores;

            if (globalQuality.IsAccepted && frameObs.AssociationAccepted && !frameObs.PossibleIdSwitch)
            {
                // Add to rolling history, update Best-Frame Pools, and re-aggregate across all attributes
                _aggregator.Aggregate(track, obs);

                debugDto = new TrackedPersonDebugDto
                {
                    FrameIndex = frameIndex,
                    YoloConfidence = track.DetectionConfidence,
                    CropWidth = analysisOutput.CropWidth,
                    CropHeight = analysisOutput.CropHeight,
                    SharpnessVariance = globalQuality.SharpnessVariance,
                    QualityScore = globalQuality.QualityScore,
                    Orientation = pose.Visibility.Orientation.ToString(),
                    HeadVisible = pose.Visibility.HeadVisible,
                    UpperVisible = pose.Visibility.UpperBodyVisible,
                    LowerVisible = pose.Visibility.LowerBodyVisible,
                    FeetVisible = pose.Visibility.FeetVisible,
                    LeftWristVisible = pose.Visibility.LeftWristVisible,
                    RightWristVisible = pose.Visibility.RightWristVisible,

                    AppearanceQuality = attrQuality.AppearanceQuality,
                    UpperQuality = attrQuality.UpperQuality,
                    LowerQuality = attrQuality.LowerQuality,
                    ShoesQuality = attrQuality.ShoesQuality,
                    LeftWristQuality = attrQuality.LeftWristQuality,
                    RightWristQuality = attrQuality.RightWristQuality,

                    MaleScore = obs.MaleScore,
                    FemaleScore = obs.FemaleScore,
                    SexMargin = Math.Abs(obs.MaleScore - obs.FemaleScore),

                    UpperTypeRawScores = obs.UpperTypeScores,
                    UpperTypeMargin = obs.UpperTypeMargin,
                    UpperColorRawScores = obs.UpperColorScores,
                    UpperPrimaryColor = obs.UpperColorDetails?.PrimaryColor ?? obs.UpperColor,
                    UpperPrimaryPercent = obs.UpperColorDetails?.PrimaryPercentage ?? 0f,
                    UpperSecondaryColor = obs.UpperColorDetails?.SecondaryColor ?? string.Empty,
                    UpperSecondaryPercent = obs.UpperColorDetails?.SecondaryPercentage ?? 0f,

                    LowerTypeRawScores = obs.LowerTypeScores,
                    LowerTypeMargin = obs.LowerTypeMargin,
                    LowerColorRawScores = obs.LowerColorScores,
                    LowerPrimaryColor = obs.LowerColorDetails?.PrimaryColor ?? obs.LowerColor,
                    LowerPrimaryPercent = obs.LowerColorDetails?.PrimaryPercentage ?? 0f,
                    LowerSecondaryColor = obs.LowerColorDetails?.SecondaryColor ?? string.Empty,
                    LowerSecondaryPercent = obs.LowerColorDetails?.SecondaryPercentage ?? 0f,

                    ShoesRawScores = obs.ShoesScores,
                    ShoesMargin = obs.ShoesMargin,

                    TrackAgeFrames = track.TrackAgeFrames,
                    LastMatchedIoU = track.LastMatchedIoU,
                    CenterDisplacement = track.CenterDisplacement,
                    PossibleIdSwitch = track.PossibleIdSwitch,
                    PossibleIdSwitchReason = track.PossibleIdSwitchReason,

                    FrameAccepted = true,
                    RejectionReason = string.Empty,
                    CurrentAggregatedResult = $"{track.AppearanceSex} ({track.AppearanceState}) | Upper: {track.UpperColor} {track.UpperType} | Lower: {track.LowerColor} {track.LowerType} | Shoes: {track.ShoesType}"
                };
            }
            else
            {
                // Frame Rejected
                debugDto = new TrackedPersonDebugDto
                {
                    FrameIndex = frameIndex,
                    YoloConfidence = track.DetectionConfidence,
                    CropWidth = analysisOutput.CropWidth,
                    CropHeight = analysisOutput.CropHeight,
                    SharpnessVariance = globalQuality.SharpnessVariance,
                    QualityScore = 0f,
                    Orientation = pose.Visibility.Orientation.ToString(),
                    HeadVisible = pose.Visibility.HeadVisible,
                    UpperVisible = pose.Visibility.UpperBodyVisible,
                    LowerVisible = pose.Visibility.LowerBodyVisible,
                    FeetVisible = pose.Visibility.FeetVisible,
                    LeftWristVisible = pose.Visibility.LeftWristVisible,
                    RightWristVisible = pose.Visibility.RightWristVisible,
                    AppearanceQuality = attrQuality.AppearanceQuality,
                    UpperQuality = attrQuality.UpperQuality,
                    LowerQuality = attrQuality.LowerQuality,
                    ShoesQuality = attrQuality.ShoesQuality,
                    TrackAgeFrames = track.TrackAgeFrames,
                    LastMatchedIoU = track.LastMatchedIoU,
                    CenterDisplacement = track.CenterDisplacement,
                    PossibleIdSwitch = track.PossibleIdSwitch,
                    PossibleIdSwitchReason = track.PossibleIdSwitchReason,
                    FrameAccepted = false,
                    RejectionReason = globalQuality.RejectionReason ?? "Low quality / blur / size filter",
                    CurrentAggregatedResult = $"{track.AppearanceSex} ({track.AppearanceState}) | Upper: {track.UpperColor} {track.UpperType} | Lower: {track.LowerColor} {track.LowerType} | Shoes: {track.ShoesType}"
                };
            }

            // Map DTO
            var personDto = new TrackedPersonDto
            {
                PersonId = track.PersonId,
                DetectionConfidence = track.DetectionConfidence,
                BoundingBox = new BoundingBoxDto(track.CurrentBox.X, track.CurrentBox.Y, track.CurrentBox.Width, track.CurrentBox.Height),

                // Keypoints & Pose
                Keypoints = (pose.Keypoints ?? new List<Keypoint>()).Select(k => new KeypointDto(k.Index, k.Name, k.X, k.Y, k.Confidence)).ToList(),

                // Appearance
                AppearanceSex = track.AppearanceSex,
                AppearanceConfidence = track.AppearanceConfidence,
                AppearanceState = track.AppearanceState.ToString(),
                AppearanceConsensus = track.AppearanceConsensus,
                IsAppearanceStable = track.IsAppearanceStable,

                // Upper
                UpperType = track.UpperType,
                UpperTypeConfidence = track.UpperTypeConfidence,
                UpperTypeState = track.UpperTypeState.ToString(),
                UpperTypeConsensus = track.UpperTypeConsensus,
                UpperColor = track.UpperColor,
                UpperColorConfidence = track.UpperColorConfidence,
                UpperColorState = track.UpperColorState.ToString(),
                UpperColorConsensus = track.UpperColorConsensus,
                IsUpperStable = track.IsUpperStable,

                // Lower
                LowerType = track.LowerType,
                LowerTypeConfidence = track.LowerTypeConfidence,
                LowerTypeState = track.LowerTypeState.ToString(),
                LowerTypeConsensus = track.LowerTypeConsensus,
                LowerColor = track.LowerColor,
                LowerColorConfidence = track.LowerColorConfidence,
                LowerColorState = track.LowerColorState.ToString(),
                LowerColorConsensus = track.LowerColorConsensus,
                IsLowerStable = track.IsLowerStable,

                // Shoes
                ShoesType = track.ShoesType,
                ShoesConfidence = track.ShoesConfidence,
                ShoesState = track.ShoesState.ToString(),
                ShoesConsensus = track.ShoesConsensus,
                IsShoesStable = track.IsShoesStable,

                // Watch
                WatchStatus = track.WatchStatus,
                WatchConfidence = track.WatchConfidence,
                WatchState = track.WatchState.ToString(),

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

                // Best Frame Pools
                BestFrames = new BestFramePoolsDto
                {
                    Appearance = track.BestAppearanceFrames.Select(MapBestFrame).ToList(),
                    Upper = track.BestUpperFrames.Select(MapBestFrame).ToList(),
                    Lower = track.BestLowerFrames.Select(MapBestFrame).ToList(),
                    Shoes = track.BestShoesFrames.Select(MapBestFrame).ToList(),
                    Wrist = track.BestWristFrames.Select(MapBestFrame).ToList()
                },

                // Pose Visibility & Quality
                Visibility = MapVisibility(pose.Visibility),
                QualityScores = MapQualityScores(attrQuality),

                ValidFrameCount = track.TotalValidObservations,
                IsFullyStable = track.IsFullyStable,
                DebugInfo = debugDto
            };

            personDtos.Add(personDto);
        }

        // Assign 1-based sequential display IDs for currently visible active persons (Person 1, Person 2, ...)
        personDtos.Sort((a, b) => a.PersonId.CompareTo(b.PersonId));
        for (int i = 0; i < personDtos.Count; i++)
        {
            personDtos[i].DisplayId = i + 1;
        }

        attrStopwatch.Stop();
        double attrMs = attrStopwatch.Elapsed.TotalMilliseconds;
        totalStopwatch.Stop();

        var responseDto = new LiveFrameResponseDto
        {
            Success = true,
            FrameIndex = frameIndex,
            ImageWidth = imgWidth,
            ImageHeight = imgHeight,
            TotalElapsedMs = totalStopwatch.Elapsed.TotalMilliseconds,
            DetectionMs = detectMs,
            AttributeMs = attrMs,
            Persons = personDtos
        };

        lock (_stateLock)
        {
            _latestState = responseDto;
        }

        return responseDto;
    }

    public LiveFrameResponseDto? GetLatestState()
    {
        lock (_stateLock)
        {
            return _latestState;
        }
    }

    public void Reset()
    {
        _personTracker.Reset();
        _cameraMotionEstimator.Reset();
        lock (_frameLock)
        {
            _prevFrameMat?.Dispose();
            _prevFrameMat = null;
        }
        lock (_stateLock)
        {
            _latestState = null;
        }
        _logger.LogInformation("Reset multi-person live tracking state, camera motion estimator, and aggregation histories.");
    }

    private static BestFrameItemDto MapBestFrame(BestFrameRecord r) => new()
    {
        FrameIndex = r.FrameIndex,
        QualityScore = r.QualityScore,
        Prediction = r.Prediction,
        Confidence = r.Confidence,
        Details = r.Details
    };

    private static BodyVisibilityDto MapVisibility(BodyVisibilityResult v) => new()
    {
        Head = v.HeadVisible,
        Upper = v.UpperBodyVisible,
        Lower = v.LowerBodyVisible,
        Feet = v.FeetVisible,
        LeftWrist = v.LeftWristVisible,
        RightWrist = v.RightWristVisible,
        LeftHandRaised = v.LeftHandRaised,
        RightHandRaised = v.RightHandRaised,
        Orientation = v.Orientation.ToString(),
        HeadClipped = v.HeadClipped,
        UpperClipped = v.UpperClipped,
        LowerClipped = v.LowerClipped,
        FeetClipped = v.FeetClipped,
        LeftWristClipped = v.LeftWristClipped,
        RightWristClipped = v.RightWristClipped
    };

    private static AttributeQualityDto MapQualityScores(AttributeQualityScores q) => new()
    {
        Appearance = q.AppearanceQuality,
        Upper = q.UpperQuality,
        Lower = q.LowerQuality,
        Shoes = q.ShoesQuality,
        LeftWrist = q.LeftWristQuality,
        RightWrist = q.RightWristQuality,
        HasAppearanceEvidence = q.HasAppearanceEvidence,
        HasUpperEvidence = q.HasUpperEvidence,
        HasLowerEvidence = q.HasLowerEvidence,
        HasShoesEvidence = q.HasShoesEvidence,
        HasWristEvidence = q.HasWristEvidence,
        AppearanceRejection = q.AppearanceRejectionReason,
        UpperRejection = q.UpperRejectionReason,
        LowerRejection = q.LowerRejectionReason,
        ShoesRejection = q.ShoesRejectionReason,
        WristRejection = q.WristRejectionReason
    };
}
