using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using OpenCvSharp;
using VisionAttributeAI.Models.AI;
using VisionAttributeAI.Models.Brand;
using VisionAttributeAI.Models.Pose;

namespace VisionAttributeAI.Services.Brand;

/// <summary>
/// Orchestrates regional logo detection and DINOv2 brand recognition across person body regions
/// (UpperTorso, LowerBody, Feet, Bags) with strict resolution and temporal stability gates.
/// </summary>
public class BrandPipelineService : IBrandPipelineService
{
    private readonly BrandOptions _options;
    private readonly IBrandLogoDetectionService _detector;
    private readonly IBrandRecognitionService _recognizer;
    private readonly ILogger<BrandPipelineService> _logger;

    public bool IsEnabled => _options.Enabled && _detector.IsModelLoaded && _recognizer.IsModelLoaded;

    public BrandPipelineService(
        IOptions<BrandOptions> options,
        IBrandLogoDetectionService detector,
        IBrandRecognitionService recognizer,
        ILogger<BrandPipelineService> logger)
    {
        _options = options.Value;
        _detector = detector;
        _recognizer = recognizer;
        _logger = logger;
    }

    public async Task<PersonBrandResult> ProcessPersonBrandsAsync(
        Mat frameMat,
        BoundingBox personBox,
        PersonPoseResult pose,
        TrackedBrandState brandState,
        long frameIndex,
        bool isVideoMode,
        bool possibleIdSwitch,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(frameMat);
        ArgumentNullException.ThrowIfNull(pose);
        ArgumentNullException.ThrowIfNull(brandState);

        if (!IsEnabled || frameMat.Empty())
        {
            return brandState.ToPersonBrandResult();
        }

        // 1. Quality & Resolution Gate: Person Height must be >= MinPersonHeight (default 180px)
        if (personBox.Height < _options.MinPersonHeight)
        {
            return brandState.ToPersonBrandResult();
        }

        // 2. Sample Interval Gate: In video mode, only sample every SampleIntervalFrames (e.g. 12 frames)
        if (isVideoMode && frameIndex > 0 && (frameIndex % _options.SampleIntervalFrames != 0))
        {
            return brandState.ToPersonBrandResult();
        }

        // 3. Process Regions Sequentially

        // A. Upper Torso
        if (pose.Regions.UpperTorsoRegion != null && (pose.Visibility.UpperBodyVisible || pose.Regions.UpperTorsoRegion.Height >= _options.MinRegionDimension))
        {
            if (!brandState.IsRegionStable("Upper"))
            {
                await ProcessSingleRegionAsync(frameMat, pose.Regions.UpperTorsoRegion, "Upper", brandState, frameIndex, isVideoMode, possibleIdSwitch, cancellationToken);
            }
        }
        else
        {
            brandState.SetResultForRegion("Upper", BrandResult.NotVisible("Upper"));
        }

        // B. Lower Body
        if (pose.Regions.LowerBodyRegion != null && (pose.Visibility.LowerBodyVisible || pose.Regions.LowerBodyRegion.Height >= _options.MinRegionDimension))
        {
            if (!brandState.IsRegionStable("Lower"))
            {
                await ProcessSingleRegionAsync(frameMat, pose.Regions.LowerBodyRegion, "Lower", brandState, frameIndex, isVideoMode, possibleIdSwitch, cancellationToken);
            }
        }
        else
        {
            brandState.SetResultForRegion("Lower", BrandResult.NotVisible("Lower"));
        }

        // C. Shoes / Feet
        if (pose.Regions.FeetRegion != null && (pose.Visibility.FeetVisible || pose.Regions.FeetRegion.Height >= _options.MinRegionDimension))
        {
            if (!brandState.IsRegionStable("Shoes"))
            {
                await ProcessSingleRegionAsync(frameMat, pose.Regions.FeetRegion, "Shoes", brandState, frameIndex, isVideoMode, possibleIdSwitch, cancellationToken);
            }
        }
        else
        {
            brandState.SetResultForRegion("Shoes", BrandResult.NotVisible("Shoes"));
        }

        // D. Watch / Wrist (Watch is not detected)
        brandState.SetResultForRegion("Watch", BrandResult.NotDetected("Watch"));

        // E. Bag / Backpack
        if (pose.Regions.BagRegion != null)
        {
            if (!brandState.IsRegionStable("Bag"))
            {
                await ProcessSingleRegionAsync(frameMat, pose.Regions.BagRegion, "Bag", brandState, frameIndex, isVideoMode, possibleIdSwitch, cancellationToken);
            }
        }
        else
        {
            brandState.SetResultForRegion("Bag", BrandResult.UnsupportedRegion("Bag"));
        }

        return brandState.ToPersonBrandResult();
    }

    private async Task ProcessSingleRegionAsync(
        Mat frameMat,
        BoundingBox regionBox,
        string regionName,
        TrackedBrandState brandState,
        long frameIndex,
        bool isVideoMode,
        bool possibleIdSwitch,
        CancellationToken cancellationToken)
    {
        using var regionMat = CropSubRegion(frameMat, regionBox);
        if (regionMat.Empty() || regionMat.Width < _options.MinRegionDimension || regionMat.Height < _options.MinRegionDimension)
        {
            brandState.SetResultForRegion(regionName, BrandResult.InsufficientVisualEvidence(regionName));
            return;
        }

        // Step 1: Run LOGOS RetinaNet candidate detector
        var candidates = await _detector.DetectLogoCandidatesAsync(regionMat, regionName, cancellationToken);

        if (candidates.Count == 0)
        {
            brandState.SetResultForRegion(regionName, BrandResult.NoLogoCandidate(regionName));
            return;
        }

        // Step 2: Evaluate top logo proposals and select the best DINOv2 match
        BrandResult? bestBrandResult = null;
        foreach (var cand in candidates.Take(4))
        {
            using var logoCrop = CropSubRegion(regionMat, cand.Box);
            if (logoCrop.Empty() || (logoCrop.Width < _options.MinLogoCropSize && logoCrop.Height < _options.MinLogoCropSize))
            {
                continue;
            }

            var result = await _recognizer.RecognizeBrandAsync(logoCrop, regionName, cancellationToken);
            if (bestBrandResult == null || result.Similarity > bestBrandResult.Similarity)
            {
                bestBrandResult = result;
            }
        }

        if (bestBrandResult == null)
        {
            brandState.SetResultForRegion(regionName, BrandResult.InsufficientVisualEvidence(regionName));
            return;
        }

        var brandResult = bestBrandResult;

        // Step 3: Formulate BrandObservation
        var observation = new BrandObservation
        {
            FrameIndex = frameIndex,
            Region = regionName,
            BrandName = brandResult.BrandName,
            Similarity = brandResult.Similarity,
            RunnerUpBrand = brandResult.RunnerUpBrand,
            RunnerUpSimilarity = brandResult.RunnerUpSimilarity,
            Margin = brandResult.Margin,
            PassedDualGate = brandResult.State is BrandState.BrandCandidate or BrandState.BrandStable,
            MatchedExemplarFile = brandResult.MatchedExemplarFile,
            TimestampUtc = DateTime.UtcNow
        };

        // Step 4: Update Temporal State Machine
        brandState.UpdateWithObservation(
            regionName,
            observation,
            isVideoMode,
            possibleIdSwitch,
            _options.RequiredStableObservations);
    }

    private static Mat CropSubRegion(Mat sourceMat, BoundingBox region)
    {
        int rw = (int)Math.Round(region.Width);
        int rh = (int)Math.Round(region.Height);

        if (rw <= 0 || rh <= 0) return new Mat();

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
