using Microsoft.Extensions.Options;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using VisionAttributeAI.Models.AI;
using VisionAttributeAI.Models.Pose;

namespace VisionAttributeAI.Services.Pose;

/// <summary>
/// Pretrained YOLOv8-pose ONNX keypoint estimation service.
/// Extracts 17 COCO keypoints, estimates 3D Pose Orientation (Front, NearFrontal, Side, Back, Uncertain),
/// calculates anatomical body regions, and detects frame-boundary clipping.
/// </summary>
public class YoloPoseEstimationService : IPoseEstimationService
{
    private readonly PersonAnalysisOptions _options;
    private readonly ILogger<YoloPoseEstimationService> _logger;
    private InferenceSession? _session;
    private string _inputName = "images";
    private string _outputName = "output0";
    private bool _disposed;

    public bool IsModelLoaded => _session != null;

    public YoloPoseEstimationService(
        IOptions<PersonAnalysisOptions> options,
        ILogger<YoloPoseEstimationService> logger)
    {
        _options = options.Value;
        _logger = logger;

        InitializeModel();
    }

    private void InitializeModel()
    {
        try
        {
            var modelPath = _options.PoseModelPath;
            if (string.IsNullOrWhiteSpace(modelPath) || !File.Exists(modelPath))
            {
                _logger.LogWarning("YOLO Pose ONNX model not found at '{ModelPath}'. Fallback anatomical region estimation enabled.", modelPath);
                return;
            }

            var sessionOptions = new Microsoft.ML.OnnxRuntime.SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL
            };

            _session = new InferenceSession(modelPath, sessionOptions);

            if (_session.InputMetadata.Count > 0)
            {
                _inputName = _session.InputMetadata.First().Key;
            }
            if (_session.OutputMetadata.Count > 0)
            {
                _outputName = _session.OutputMetadata.First().Key;
            }

            _logger.LogInformation("Successfully loaded YOLO Pose model '{ModelPath}'. Input='{Input}', Output='{Output}'",
                modelPath, _inputName, _outputName);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to initialize YOLO Pose model from '{ModelPath}'.", _options.PoseModelPath);
        }
    }

    public Task<IReadOnlyList<PersonPoseResult>> EstimatePoseAsync(
        Mat image,
        IReadOnlyList<DetectionResult> detections,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(image);
        if (image.Empty() || detections == null || detections.Count == 0)
        {
            return Task.FromResult<IReadOnlyList<PersonPoseResult>>(Array.Empty<PersonPoseResult>());
        }

        cancellationToken.ThrowIfCancellationRequested();

        // If pose ONNX model is loaded, run YOLOv8-pose inference
        if (_session != null)
        {
            try
            {
                var poseResults = RunPoseInference(image, detections, cancellationToken);
                return Task.FromResult<IReadOnlyList<PersonPoseResult>>(poseResults);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Pose inference failed; falling back to geometric anatomical region estimation.");
            }
        }

        // Graceful Fallback: Compute geometric anatomical body regions & visibility
        var fallbackResults = ComputeFallbackAnatomicalRegions(image, detections);
        return Task.FromResult<IReadOnlyList<PersonPoseResult>>(fallbackResults);
    }

    private List<PersonPoseResult> RunPoseInference(Mat image, IReadOnlyList<DetectionResult> detections, CancellationToken cancellationToken)
    {
        int origW = image.Width;
        int origH = image.Height;
        int targetSize = 640;

        // 1. Letterbox Preprocessing
        float scale = Math.Min((float)targetSize / origW, (float)targetSize / origH);
        int newW = (int)Math.Round(origW * scale);
        int newH = (int)Math.Round(origH * scale);
        int padX = (targetSize - newW) / 2;
        int padY = (targetSize - newH) / 2;

        using var resized = new Mat();
        Cv2.Resize(image, resized, new Size(newW, newH), 0, 0, InterpolationFlags.Linear);

        using var letterboxed = new Mat(targetSize, targetSize, MatType.CV_8UC3, new Scalar(114, 114, 114));
        var roi = new Rect(padX, padY, newW, newH);
        using var targetRoi = new Mat(letterboxed, roi);
        resized.CopyTo(targetRoi);

        // 2. Normalization & Float32 NCHW Tensor creation
        var tensor = new DenseTensor<float>(new[] { 1, 3, targetSize, targetSize });

        unsafe
        {
            byte* ptr = letterboxed.DataPointer;
            int step = (int)letterboxed.Step();

            for (int y = 0; y < targetSize; y++)
            {
                byte* row = ptr + (y * step);
                for (int x = 0; x < targetSize; x++)
                {
                    int pIdx = x * 3;
                    tensor[0, 0, y, x] = row[pIdx + 2] / 255.0f; // R
                    tensor[0, 1, y, x] = row[pIdx + 1] / 255.0f; // G
                    tensor[0, 2, y, x] = row[pIdx] / 255.0f;     // B
                }
            }
        }

        // 3. Run Pose ONNX Inference
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_inputName, tensor)
        };

        using var results = _session!.Run(inputs);
        var outputTensor = results.First(r => r.Name == _outputName);
        var outputData = outputTensor.AsEnumerable<float>().ToArray();
        var dims = outputTensor.AsTensor<float>().Dimensions; // typically [1, 56, 8400]

        int numFeatures = dims.Length > 1 ? dims[1] : 56;
        int numAnchors = dims.Length > 2 ? dims[2] : 8400;

        // 4. Parse candidates and match to incoming person detections
        var allCandidates = new List<(BoundingBox Box, float Score, List<Keypoint> Kps)>();
        float minConf = _options.MinPersonConfidence;

        for (int i = 0; i < numAnchors; i++)
        {
            float cx = outputData[0 * numAnchors + i];
            float cy = outputData[1 * numAnchors + i];
            float w = outputData[2 * numAnchors + i];
            float h = outputData[3 * numAnchors + i];
            float score = outputData[4 * numAnchors + i];

            if (score < minConf) continue;

            // Un-letterbox bounding box
            float origBoxX = Math.Clamp((cx - w / 2f - padX) / scale, 0, origW);
            float origBoxY = Math.Clamp((cy - h / 2f - padY) / scale, 0, origH);
            float origBoxW = Math.Clamp(w / scale, 1, origW - origBoxX);
            float origBoxH = Math.Clamp(h / scale, 1, origH - origBoxY);

            // Un-letterbox 17 keypoints
            var keypoints = new List<Keypoint>();
            for (int k = 0; k < 17; k++)
            {
                int kIdx = 5 + k * 3;
                float kx = (outputData[kIdx * numAnchors + i] - padX) / scale;
                float ky = (outputData[(kIdx + 1) * numAnchors + i] - padY) / scale;
                float kconf = outputData[(kIdx + 2) * numAnchors + i];

                string name = k < PersonPoseResult.CocoKeypointNames.Length ? PersonPoseResult.CocoKeypointNames[k] : $"Keypoint_{k}";
                keypoints.Add(new Keypoint(k, name, Math.Clamp(kx, 0, origW), Math.Clamp(ky, 0, origH), kconf));
            }

            allCandidates.Add((new BoundingBox(origBoxX, origBoxY, origBoxW, origBoxH), score, keypoints));
        }

        // 5. Match candidate poses to incoming YOLO person detections via highest IoU
        var resultsList = new List<PersonPoseResult>();
        float kpThresh = _options.MinKeypointConfidence;

        foreach (var det in detections)
        {
            var bestMatch = allCandidates
                .Select(c => (Candidate: c, IoU: CalculateIoU(det.Box, c.Box)))
                .Where(m => m.IoU >= 0.15f)
                .OrderByDescending(m => m.IoU)
                .FirstOrDefault();

            if (bestMatch.Candidate.Kps != null && bestMatch.Candidate.Kps.Count == 17)
            {
                var kps = bestMatch.Candidate.Kps;
                var rawVisibility = CalculateVisibilityFromKeypoints(kps, kpThresh, det.Box);
                var regions = CalculateBodyRegionsFromKeypoints(kps, det.Box, origW, origH, rawVisibility);

                // Check frame boundary clipping for each sub-region
                var finalVisibility = rawVisibility with
                {
                    HeadClipped = IsClipped(regions.HeadRegion, origW, origH),
                    UpperClipped = IsClipped(regions.UpperTorsoRegion, origW, origH),
                    LowerClipped = IsClipped(regions.LowerBodyRegion, origW, origH),
                    FeetClipped = IsClipped(regions.FeetRegion, origW, origH, margin: 4f),
                    LeftWristClipped = IsClipped(regions.LeftWristRegion, origW, origH),
                    RightWristClipped = IsClipped(regions.RightWristRegion, origW, origH)
                };

                resultsList.Add(new PersonPoseResult
                {
                    BoundingBox = det.Box,
                    DetectionConfidence = det.Confidence,
                    Keypoints = kps,
                    Visibility = finalVisibility,
                    Regions = regions
                });
            }
            else
            {
                // Fallback for this individual person
                var (visibility, regions) = ComputeSinglePersonFallback(det.Box, origW, origH);
                resultsList.Add(new PersonPoseResult
                {
                    BoundingBox = det.Box,
                    DetectionConfidence = det.Confidence,
                    Keypoints = new List<Keypoint>(),
                    Visibility = visibility,
                    Regions = regions
                });
            }
        }

        return resultsList;
    }

    private BodyVisibilityResult CalculateVisibilityFromKeypoints(List<Keypoint> kps, float thresh, BoundingBox personBox)
    {
        // 0: Nose, 1: LEye, 2: REye, 3: LEar, 4: REar
        // 5: LShoulder, 6: RShoulder, 7: LElbow, 8: RElbow, 9: LWrist, 10: RWrist
        // 11: LHip, 12: RHip, 13: LKnee, 14: RKnee, 15: LAnkle, 16: RAnkle

        bool noseVis = kps[0].IsVisible(thresh);
        bool eyeVis = kps[1].IsVisible(thresh) || kps[2].IsVisible(thresh);
        bool earVis = kps[3].IsVisible(thresh) || kps[4].IsVisible(thresh);
        bool headVis = noseVis || eyeVis || earVis;

        bool lShoulderVis = kps[5].IsVisible(thresh);
        bool rShoulderVis = kps[6].IsVisible(thresh);
        bool shoulderVis = lShoulderVis || rShoulderVis;

        bool lHipVis = kps[11].IsVisible(thresh);
        bool rHipVis = kps[12].IsVisible(thresh);
        bool hipVis = lHipVis || rHipVis;

        bool upperBodyVis = shoulderVis && (headVis || hipVis || (lShoulderVis && rShoulderVis));

        bool lElbowVis = kps[7].IsVisible(thresh);
        bool rElbowVis = kps[8].IsVisible(thresh);
        bool lWristVis = kps[9].IsVisible(thresh);
        bool rWristVis = kps[10].IsVisible(thresh);

        bool lArmVis = lShoulderVis && (lElbowVis || lWristVis);
        bool rArmVis = rShoulderVis && (rElbowVis || rWristVis);

        // Anatomical Hand-Raised State: Wrist Y < Shoulder Y (in image coords, top is 0)
        bool lHandRaised = lWristVis && lShoulderVis && (kps[9].Y < kps[5].Y);
        bool rHandRaised = rWristVis && rShoulderVis && (kps[10].Y < kps[6].Y);

        bool lKneeVis = kps[13].IsVisible(thresh);
        bool rKneeVis = kps[14].IsVisible(thresh);
        bool lAnkleVis = kps[15].IsVisible(thresh);
        bool rAnkleVis = kps[16].IsVisible(thresh);

        bool kneeVis = lKneeVis || rKneeVis;
        bool ankleVis = lAnkleVis || rAnkleVis;

        bool lowerBodyVis = hipVis && (kneeVis || ankleVis || (lHipVis && rHipVis && personBox.Height > 160));
        bool lLegVis = lHipVis && (lKneeVis || lAnkleVis);
        bool rLegVis = rHipVis && (rKneeVis || rAnkleVis);
        bool feetVis = ankleVis;

        int visibleCount = kps.Count(k => k.IsVisible(thresh));
        float avgPoseConf = kps.Average(k => k.Confidence);

        // Pose Orientation Estimation
        var orientation = EstimateOrientation(kps, thresh);

        return new BodyVisibilityResult
        {
            HeadVisible = headVis,
            UpperBodyVisible = upperBodyVis,
            LeftArmVisible = lArmVis,
            RightArmVisible = rArmVis,
            LeftWristVisible = lWristVis,
            RightWristVisible = rWristVis,
            LeftHandRaised = lHandRaised,
            RightHandRaised = rHandRaised,
            LowerBodyVisible = lowerBodyVis,
            LeftLegVisible = lLegVis,
            RightLegVisible = rLegVis,
            FeetVisible = feetVis,
            OverallPoseConfidence = avgPoseConf,
            VisibleKeypointCount = visibleCount,
            Orientation = orientation
        };
    }

    private static PoseOrientation EstimateOrientation(List<Keypoint> kps, float thresh)
    {
        float noseConf = kps[0].Confidence;
        float lEyeConf = kps[1].Confidence;
        float rEyeConf = kps[2].Confidence;
        float lEarConf = kps[3].Confidence;
        float rEarConf = kps[4].Confidence;
        float lShoulderConf = kps[5].Confidence;
        float rShoulderConf = kps[6].Confidence;

        bool hasFrontFace = noseConf >= thresh && (lEyeConf >= thresh || rEyeConf >= thresh);
        bool hasBothEyes = lEyeConf >= thresh && rEyeConf >= thresh;
        bool hasBothShoulders = lShoulderConf >= thresh && rShoulderConf >= thresh;

        if (hasFrontFace && hasBothEyes && hasBothShoulders)
        {
            return PoseOrientation.Front;
        }

        if (hasFrontFace && (hasBothShoulders || lEyeConf >= thresh || rEyeConf >= thresh))
        {
            return PoseOrientation.NearFrontal;
        }

        if ((lEyeConf >= thresh ^ rEyeConf >= thresh) || (lEarConf >= thresh ^ rEarConf >= thresh))
        {
            return PoseOrientation.Side;
        }

        // Rear view detection: shoulders visible, but nose and eyes very low confidence (< 0.20)
        if (hasBothShoulders && noseConf < 0.20f && lEyeConf < 0.20f && rEyeConf < 0.20f)
        {
            return PoseOrientation.Back;
        }

        if (lEarConf >= thresh && rEarConf >= thresh && noseConf < 0.20f)
        {
            return PoseOrientation.Back;
        }

        return PoseOrientation.Uncertain;
    }

    private BodyRegions CalculateBodyRegionsFromKeypoints(
        List<Keypoint> kps,
        BoundingBox personBox,
        int imageWidth,
        int imageHeight,
        BodyVisibilityResult visibility)
    {
        // 1. Head Region: [Top of box to Shoulder line]
        BoundingBox? headRegion = null;
        if (visibility.HeadVisible)
        {
            float shoulderY = (kps[5].IsVisible(0.2f) || kps[6].IsVisible(0.2f))
                ? Math.Max(kps[5].Y, kps[6].Y)
                : personBox.Y + personBox.Height * 0.25f;

            float hTop = personBox.Y;
            float hHeight = Math.Max(20f, shoulderY - hTop);
            headRegion = ClampBox(personBox.X + personBox.Width * 0.15f, hTop, personBox.Width * 0.70f, hHeight, imageWidth, imageHeight);
        }

        // 2. Upper Torso Region: [Neck/Shoulders down to Hips]
        BoundingBox? upperTorsoRegion = null;
        if (visibility.UpperBodyVisible)
        {
            float topY = (kps[5].IsVisible(0.2f) || kps[6].IsVisible(0.2f))
                ? Math.Min(kps[5].Y, kps[6].Y) - 15f
                : personBox.Y + personBox.Height * 0.12f;

            float botY = (kps[11].IsVisible(0.2f) || kps[12].IsVisible(0.2f))
                ? Math.Max(kps[11].Y, kps[12].Y) + 15f
                : personBox.Y + personBox.Height * 0.60f;

            float uHeight = Math.Max(30f, botY - topY);

            // Compute tight horizontal bounds using shoulders/hips
            float leftX = personBox.X;
            float width = personBox.Width;

            if (kps[5].IsVisible(0.2f) && kps[6].IsVisible(0.2f))
            {
                float sMinX = Math.Min(kps[5].X, kps[6].X);
                float sMaxX = Math.Max(kps[5].X, kps[6].X);
                float span = sMaxX - sMinX;
                leftX = sMinX - span * 0.12f;
                width = span * 1.24f;
            }

            upperTorsoRegion = ClampBox(leftX, topY, width, uHeight, imageWidth, imageHeight);
        }

        // 3. Left & Right Wrist Regions (High-Resolution Bounding Box centered on wrist with forearm context)
        BoundingBox? lWristRegion = null;
        if (visibility.LeftWristVisible && kps[9].IsVisible(0.25f))
        {
            float radius = Math.Clamp(personBox.Width * 0.20f, 25f, 75f);
            float cx = kps[9].X;
            float cy = kps[9].Y;

            // If left elbow (index 7) is visible, offset center slightly along forearm to capture watch on lower forearm
            if (kps[7].IsVisible(0.20f))
            {
                float dx = kps[7].X - kps[9].X;
                float dy = kps[7].Y - kps[9].Y;
                float len = MathF.Sqrt(dx * dx + dy * dy);
                if (len > 5f)
                {
                    cx += (dx / len) * (radius * 0.25f);
                    cy += (dy / len) * (radius * 0.25f);
                }
            }

            lWristRegion = ClampBox(cx - radius, cy - radius, radius * 2, radius * 2, imageWidth, imageHeight);
        }

        BoundingBox? rWristRegion = null;
        if (visibility.RightWristVisible && kps[10].IsVisible(0.25f))
        {
            float radius = Math.Clamp(personBox.Width * 0.20f, 25f, 75f);
            float cx = kps[10].X;
            float cy = kps[10].Y;

            // If right elbow (index 8) is visible, offset center slightly along forearm
            if (kps[8].IsVisible(0.20f))
            {
                float dx = kps[8].X - kps[10].X;
                float dy = kps[8].Y - kps[10].Y;
                float len = MathF.Sqrt(dx * dx + dy * dy);
                if (len > 5f)
                {
                    cx += (dx / len) * (radius * 0.25f);
                    cy += (dy / len) * (radius * 0.25f);
                }
            }

            rWristRegion = ClampBox(cx - radius, cy - radius, radius * 2, radius * 2, imageWidth, imageHeight);
        }

        // 4. Lower Body Region: [Below upper garment hem / pelvis down to Ankles]
        BoundingBox? lowerBodyRegion = null;
        if (visibility.LowerBodyVisible)
        {
            // Start at hip line (clean boundary to avoid upper garment overlap)
            float hipY = (kps[11].IsVisible(0.2f) || kps[12].IsVisible(0.2f))
                ? ((kps[11].IsVisible(0.2f) && kps[12].IsVisible(0.2f)) ? Math.Max(kps[11].Y, kps[12].Y) : (kps[11].IsVisible(0.2f) ? kps[11].Y : kps[12].Y))
                : personBox.Y + personBox.Height * 0.48f;

            float feetY = (kps[15].IsVisible(0.2f) || kps[16].IsVisible(0.2f))
                ? Math.Max(kps[15].Y, kps[16].Y)
                : personBox.Y + personBox.Height * 0.95f;

            float lHeight = Math.Max(30f, feetY - hipY);

            // Compute horizontal leg column bounds using hips and knees/ankles
            float leftX = personBox.X;
            float width = personBox.Width;

            bool hasLHip = kps[11].IsVisible(0.2f);
            bool hasRHip = kps[12].IsVisible(0.2f);

            if (hasLHip && hasRHip)
            {
                float hMinX = Math.Min(kps[11].X, kps[12].X);
                float hMaxX = Math.Max(kps[11].X, kps[12].X);
                float hSpan = hMaxX - hMinX;
                float midX = (hMinX + hMaxX) * 0.5f;

                if (hSpan > 25f)
                {
                    float legWidth = hSpan * 1.5f;
                    leftX = Math.Max(personBox.X, midX - legWidth * 0.5f);
                    width = Math.Min(personBox.Width - (leftX - personBox.X), legWidth);
                }
                else
                {
                    // Profile / Side View
                    float thighWidth = Math.Clamp(personBox.Width * 0.60f, 35f, 150f);
                    leftX = Math.Max(personBox.X, midX - thighWidth * 0.5f);
                    width = Math.Min(personBox.Width - (leftX - personBox.X), thighWidth);
                }
            }
            else if (hasLHip || hasRHip)
            {
                float hipX = hasLHip ? kps[11].X : kps[12].X;
                float thighWidth = Math.Clamp(personBox.Width * 0.60f, 35f, 150f);
                leftX = Math.Max(personBox.X, hipX - thighWidth * 0.5f);
                width = Math.Min(personBox.Width - (leftX - personBox.X), thighWidth);
            }

            lowerBodyRegion = ClampBox(leftX, hipY, width, lHeight, imageWidth, imageHeight);
        }

        // 5. Feet Region: [Ankles to Bottom of Box]
        BoundingBox? feetRegion = null;
        if (visibility.FeetVisible)
        {
            float ankleY = (kps[15].IsVisible(0.2f) || kps[16].IsVisible(0.2f))
                ? Math.Min(kps[15].Y, kps[16].Y) - 10f
                : personBox.Y + personBox.Height * 0.85f;

            float fHeight = Math.Max(20f, (personBox.Y + personBox.Height) - ankleY);
            feetRegion = ClampBox(personBox.X, ankleY, personBox.Width, fHeight, imageWidth, imageHeight);
        }

        return new BodyRegions
        {
            HeadRegion = headRegion,
            UpperTorsoRegion = upperTorsoRegion,
            LeftWristRegion = lWristRegion,
            RightWristRegion = rWristRegion,
            LowerBodyRegion = lowerBodyRegion,
            FeetRegion = feetRegion
        };
    }

    private static bool IsClipped(BoundingBox? box, int imgW, int imgH, float margin = 3f)
    {
        if (box == null) return false;
        return box.X <= margin || box.Y <= margin || (box.X + box.Width >= imgW - margin) || (box.Y + box.Height >= imgH - margin);
    }

    private List<PersonPoseResult> ComputeFallbackAnatomicalRegions(Mat image, IReadOnlyList<DetectionResult> detections)
    {
        var list = new List<PersonPoseResult>();
        int imgW = image.Width;
        int imgH = image.Height;

        foreach (var det in detections)
        {
            var (visibility, regions) = ComputeSinglePersonFallback(det.Box, imgW, imgH);
            list.Add(new PersonPoseResult
            {
                BoundingBox = det.Box,
                DetectionConfidence = det.Confidence,
                Keypoints = new List<Keypoint>(),
                Visibility = visibility,
                Regions = regions
            });
        }

        return list;
    }

    private (BodyVisibilityResult Visibility, BodyRegions Regions) ComputeSinglePersonFallback(BoundingBox box, int imgW, int imgH)
    {
        float aspect = box.Height / Math.Max(1f, box.Width);
        bool isFullBody = aspect >= 1.6f && box.Height > 180f;

        bool headVis = true;
        bool upperVis = true;
        bool lowerVis = isFullBody;
        bool feetVis = isFullBody && (box.Y + box.Height >= imgH * 0.80f || aspect >= 2.0f);

        var visibility = new BodyVisibilityResult
        {
            HeadVisible = headVis,
            UpperBodyVisible = upperVis,
            LeftArmVisible = true,
            RightArmVisible = true,
            LeftWristVisible = false,
            RightWristVisible = false,
            LowerBodyVisible = lowerVis,
            LeftLegVisible = lowerVis,
            RightLegVisible = lowerVis,
            FeetVisible = feetVis,
            OverallPoseConfidence = 0.5f,
            VisibleKeypointCount = isFullBody ? 14 : 7,
            Orientation = PoseOrientation.Uncertain
        };

        var head = ClampBox(box.X + box.Width * 0.15f, box.Y, box.Width * 0.70f, box.Height * 0.25f, imgW, imgH);
        var upper = ClampBox(box.X, box.Y + box.Height * 0.18f, box.Width, isFullBody ? box.Height * 0.35f : box.Height * 0.70f, imgW, imgH);
        BoundingBox? lower = lowerVis ? ClampBox(box.X, box.Y + box.Height * 0.48f, box.Width, box.Height * 0.40f, imgW, imgH) : null;
        BoundingBox? feet = feetVis ? ClampBox(box.X, box.Y + box.Height * 0.85f, box.Width, box.Height * 0.15f, imgW, imgH) : null;

        var regions = new BodyRegions
        {
            HeadRegion = head,
            UpperTorsoRegion = upper,
            LeftWristRegion = null,
            RightWristRegion = null,
            LowerBodyRegion = lower,
            FeetRegion = feet
        };

        return (visibility, regions);
    }

    private static BoundingBox ClampBox(float x, float y, float w, float h, int maxW, int maxH)
    {
        float clX = Math.Clamp(x, 0, maxW - 1);
        float clY = Math.Clamp(y, 0, maxH - 1);
        float clW = Math.Clamp(w, 1, maxW - clX);
        float clH = Math.Clamp(h, 1, maxH - clY);
        return new BoundingBox(clX, clY, clW, clH);
    }

    private static float CalculateIoU(BoundingBox a, BoundingBox b)
    {
        float xA = Math.Max(a.X, b.X);
        float yA = Math.Max(a.Y, b.Y);
        float xB = Math.Min(a.X + a.Width, b.X + b.Width);
        float yB = Math.Min(a.Y + a.Height, b.Y + b.Height);

        float interArea = Math.Max(0, xB - xA) * Math.Max(0, yB - yA);
        if (interArea <= 0) return 0.0f;

        float unionArea = (a.Width * a.Height) + (b.Width * b.Height) - interArea;
        return unionArea > 0 ? interArea / unionArea : 0.0f;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _session?.Dispose();
        _disposed = true;
        GC.SuppressFinalize(this);
    }
}
