using System.Diagnostics;
using OpenCvSharp;
using VisionAttributeAI.Models.AI;
using VisionAttributeAI.Models.Brand;
using VisionAttributeAI.Models.Pose;
using VisionAttributeAI.Models.Tracking;
using VisionAttributeAI.Services.Attributes;
using VisionAttributeAI.Services.Brand;
using VisionAttributeAI.Services.Cropping;
using VisionAttributeAI.Services.Quality;
using VisionAttributeAI.Services.Watch;

namespace VisionAttributeAI.Services.PersonAnalysis;

/// <summary>
/// Implements the shared, unified single-person analysis engine:
/// Person Crop -> Attribute Quality & Gating -> Regional Subcrops ->
/// Fashion-CLIP Classification -> OpenCV HSV Color -> Watch Detection -> Brand Recognition -> PersonObservation.
/// </summary>
public class UnifiedPersonAnalysisService : IUnifiedPersonAnalysisService
{
    private readonly IPersonCropper _personCropper;
    private readonly IFrameQualityFilter _qualityFilter;
    private readonly IAttributeQualityEvaluator _attrQualityEvaluator;
    private readonly IFashionClipService _clipService;
    private readonly IClothingColorService _colorService;
    private readonly IWatchDetectionService _watchService;
    private readonly IBrandPipelineService _brandPipelineService;
    private readonly ILogger<UnifiedPersonAnalysisService> _logger;

    public UnifiedPersonAnalysisService(
        IPersonCropper personCropper,
        IFrameQualityFilter qualityFilter,
        IAttributeQualityEvaluator attrQualityEvaluator,
        IFashionClipService clipService,
        IClothingColorService colorService,
        IWatchDetectionService watchService,
        IBrandPipelineService brandPipelineService,
        ILogger<UnifiedPersonAnalysisService> logger)
    {
        _personCropper = personCropper;
        _qualityFilter = qualityFilter;
        _attrQualityEvaluator = attrQualityEvaluator;
        _clipService = clipService;
        _colorService = colorService;
        _watchService = watchService;
        _brandPipelineService = brandPipelineService;
        _logger = logger;
    }

    public async Task<PersonAnalysisOutput> AnalyzePersonAsync(
        Mat frameMat,
        BoundingBox boundingBox,
        float confidence,
        int personId,
        long frameIndex,
        PersonPoseResult pose,
        bool generateBase64Crop = false,
        TrackedBrandState? brandState = null,
        bool isVideoMode = false,
        bool possibleIdSwitch = false,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(frameMat);
        ArgumentNullException.ThrowIfNull(pose);

        int imgWidth = frameMat.Width;
        int imgHeight = frameMat.Height;

        // 1. Crop individual person ROI
        using var personCrop = _personCropper.CropPerson(frameMat, boundingBox, personId, generateBase64Preview: generateBase64Crop);

        // 2. Global Frame Quality Check
        var globalQuality = _qualityFilter.EvaluateCropQuality(personCrop.CroppedMat, boundingBox, confidence);

        // 3. Attribute-Specific Quality Scores & Resolution Gating
        var attrQuality = _attrQualityEvaluator.EvaluateQuality(frameMat, boundingBox, confidence, pose, imgWidth, imgHeight);

        // 4. Attribute Evaluation
        ClipClassificationResult? sexResult = null;
        float maleScore = 0.5f;
        float femaleScore = 0.5f;

        ClipClassificationResult? upperTypeResult = null;
        ColorClassificationResult? upperColorDetails = null;

        ClipClassificationResult? lowerTypeResult = null;
        ColorClassificationResult? lowerColorDetails = null;

        ClipClassificationResult? shoesResult = null;
        WatchResult? watchResult = null;
        PersonBrandResult? brandResults = null;

        if (globalQuality.IsAccepted)
        {
            // A. Appearance Sex (Only if HasAppearanceEvidence)
            if (attrQuality.HasAppearanceEvidence && pose.Regions.HeadRegion != null)
            {
                using var headCrop = CropSubRegion(frameMat, pose.Regions.HeadRegion);
                if (!headCrop.Empty())
                {
                    sexResult = await _clipService.ClassifySexAsync(headCrop, cancellationToken);
                    maleScore = sexResult.CategoryScores.GetValueOrDefault("Male", 0.5f);
                    femaleScore = sexResult.CategoryScores.GetValueOrDefault("Female", 0.5f);
                }
            }

            // B. Upper Clothing (Type & Color - Only if HasUpperEvidence)
            if (attrQuality.HasUpperEvidence && pose.Regions.UpperTorsoRegion != null)
            {
                using var torsoCrop = CropSubRegion(frameMat, pose.Regions.UpperTorsoRegion);
                if (!torsoCrop.Empty())
                {
                    upperTypeResult = await _clipService.ClassifyUpperClothingTypeAsync(torsoCrop, cancellationToken);
                    upperColorDetails = _colorService.ClassifyDetailedColor(torsoCrop, frameMat, "Upper", pose, pose.Regions.UpperTorsoRegion);
                }
            }

            // C. Lower Clothing (Type & Color - Only if HasLowerEvidence)
            if (attrQuality.HasLowerEvidence && pose.Regions.LowerBodyRegion != null)
            {
                using var lowerCrop = CropSubRegion(frameMat, pose.Regions.LowerBodyRegion);
                if (!lowerCrop.Empty())
                {
                    lowerTypeResult = await _clipService.ClassifyLowerClothingTypeAsync(lowerCrop, cancellationToken);
                    lowerColorDetails = _colorService.ClassifyDetailedColor(lowerCrop, frameMat, "Lower", pose, pose.Regions.LowerBodyRegion);
                }
            }

            // D. Shoes (Only if HasShoesEvidence)
            if (attrQuality.HasShoesEvidence && pose.Regions.FeetRegion != null)
            {
                using var feetCrop = CropSubRegion(frameMat, pose.Regions.FeetRegion);
                if (!feetCrop.Empty())
                {
                    shoesResult = await _clipService.ClassifyShoesTypeAsync(feetCrop, cancellationToken);
                }
            }

            // E. Watch (Only if HasWristEvidence)
            if (attrQuality.HasWristEvidence)
            {
                using var lWristCrop = pose.Regions.LeftWristRegion != null ? CropSubRegion(frameMat, pose.Regions.LeftWristRegion) : new Mat();
                using var rWristCrop = pose.Regions.RightWristRegion != null ? CropSubRegion(frameMat, pose.Regions.RightWristRegion) : new Mat();

                watchResult = await _watchService.DetectWatchAsync(
                    lWristCrop.Empty() ? null : lWristCrop,
                    rWristCrop.Empty() ? null : rWristCrop,
                    cancellationToken);
            }

            // F. Brand & Logo Recognition (Only if person passes global gate)
            if (_brandPipelineService.IsEnabled)
            {
                var activeBrandState = brandState ?? new TrackedBrandState();
                brandResults = await _brandPipelineService.ProcessPersonBrandsAsync(
                    frameMat,
                    boundingBox,
                    pose,
                    activeBrandState,
                    frameIndex,
                    isVideoMode,
                    possibleIdSwitch,
                    cancellationToken);
            }
        }

        // 5. Construct Observation
        var observation = new PersonObservation
        {
            PersonId = personId,
            FrameIndex = frameIndex,
            TimestampUtc = DateTime.UtcNow,
            YoloConfidence = confidence,
            CropWidth = personCrop.CropWidth,
            CropHeight = personCrop.CropHeight,
            SharpnessVariance = globalQuality.SharpnessVariance,
            QualityScore = globalQuality.QualityScore,
            ComputedWeight = globalQuality.ComputedWeight,
            Visibility = pose.Visibility,
            QualityScores = attrQuality,

            // Appearance Sex
            HasAppearanceObservation = sexResult != null,
            MaleScore = maleScore,
            FemaleScore = femaleScore,
            SexPrediction = sexResult?.TopCategory ?? "Unknown",
            SexConfidence = sexResult?.TopConfidence ?? 0f,

            // Upper
            HasUpperObservation = upperTypeResult != null,
            UpperType = upperTypeResult?.TopCategory ?? "Unknown",
            UpperTypeConfidence = upperTypeResult?.TopConfidence ?? 0f,
            UpperTypeMargin = upperTypeResult?.Margin ?? 0f,
            UpperTypeScores = upperTypeResult?.CategoryScores ?? new Dictionary<string, float>(),
            UpperColor = upperColorDetails?.PrimaryColor ?? "Unknown",
            UpperColorConfidence = upperColorDetails?.ColorConfidence ?? 0f,
            UpperColorScores = upperColorDetails?.AllColorPercentages ?? new Dictionary<string, float>(),
            UpperColorDetails = upperColorDetails,

            // Lower
            HasLowerObservation = lowerTypeResult != null,
            LowerType = lowerTypeResult?.TopCategory ?? "Not Visible",
            LowerTypeConfidence = lowerTypeResult?.TopConfidence ?? 0f,
            LowerTypeMargin = lowerTypeResult?.Margin ?? 0f,
            LowerTypeScores = lowerTypeResult?.CategoryScores ?? new Dictionary<string, float>(),
            LowerColor = lowerColorDetails?.PrimaryColor ?? "Not Visible",
            LowerColorConfidence = lowerColorDetails?.ColorConfidence ?? 0f,
            LowerColorScores = lowerColorDetails?.AllColorPercentages ?? new Dictionary<string, float>(),
            LowerColorDetails = lowerColorDetails,

            // Shoes (Pose Feet ROI -> Feet Quality Gate -> Fashion-CLIP)
            HasShoesObservation = shoesResult != null,
            ShoesType = shoesResult != null ? shoesResult.TopCategory : (!pose.Visibility.FeetVisible ? "Not Visible" : "Insufficient Evidence"),
            ShoesConfidence = shoesResult?.TopConfidence ?? 0f,
            ShoesMargin = shoesResult?.Margin ?? 0f,
            ShoesScores = shoesResult?.CategoryScores ?? new Dictionary<string, float>(),

            // Watch (Pose Wrist ROIs -> Wrist Quality Gate -> Watch Detector)
            HasWatchObservation = watchResult != null,
            WatchStatus = watchResult != null 
                ? (watchResult.Status == "WatchDetected" || watchResult.Status == "Detected" ? "Detected" : (watchResult.Status == "NoWatchDetected" || watchResult.Status == "Not Detected" || watchResult.Status == "ModelUnavailable" ? "Not Detected" : watchResult.Status))
                : (!pose.Visibility.LeftWristVisible && !pose.Visibility.RightWristVisible ? "Not Visible" : "Insufficient Evidence"),
            WatchConfidence = watchResult?.Confidence ?? 0f,
            WatchDetails = watchResult?.Status == "Detected" || watchResult?.Status == "WatchDetected" ? "Detected" : "Not Detected",
            LeftWristDetected = watchResult?.LeftWristDetected ?? false,
            RightWristDetected = watchResult?.RightWristDetected ?? false,

            // Brand & Logo Evidence
            Brands = brandResults
        };

        return new PersonAnalysisOutput
        {
            PersonId = personId,
            BoundingBox = boundingBox,
            DetectionConfidence = confidence,
            Pose = pose,
            QualityScores = attrQuality,
            GlobalQuality = globalQuality,
            Observation = observation,
            CropBase64 = personCrop.Base64DataUrl,
            CropWidth = personCrop.CropWidth,
            CropHeight = personCrop.CropHeight
        };
    }

    private static Mat CropSubRegion(Mat sourceMat, BoundingBox region)
    {
        int rw = (int)Math.Round(region.Width);
        int rh = (int)Math.Round(region.Height);

        if (rw <= 0 || rh <= 0)
        {
            return new Mat();
        }

        int rx = (int)Math.Round(region.X);
        int ry = (int)Math.Round(region.Y);

        int clampedX = Math.Clamp(rx, 0, sourceMat.Width - 1);
        int clampedY = Math.Clamp(ry, 0, sourceMat.Height - 1);
        int clampedW = Math.Clamp(rw, 1, sourceMat.Width - clampedX);
        int clampedH = Math.Clamp(rh, 1, sourceMat.Height - clampedY);

        var validRect = new Rect(clampedX, clampedY, clampedW, clampedH);
        return new Mat(sourceMat, validRect).Clone();
    }
}
