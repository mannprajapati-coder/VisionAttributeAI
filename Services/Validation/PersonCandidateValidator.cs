using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using VisionAttributeAI.Models.AI;
using VisionAttributeAI.Models.Pose;
using VisionAttributeAI.Models.Validation;

namespace VisionAttributeAI.Services.Validation;

/// <summary>
/// Production multi-signal person validation gate.
/// Evaluates YOLO detections against geometric bounds, relative frame occupancy,
/// YOLO-Pose human keypoint evidence, and anatomical body consistency before admitting a candidate.
/// </summary>
public class PersonCandidateValidator : IPersonCandidateValidator
{
    private readonly PersonAnalysisOptions _options;
    private readonly ILogger<PersonCandidateValidator> _logger;

    public PersonCandidateValidator(
        IOptions<PersonAnalysisOptions> options,
        ILogger<PersonCandidateValidator> logger)
    {
        _options = options.Value;
        _logger = logger;
    }

    public PersonCandidateValidationResult ValidateCandidate(
        DetectionResult detection,
        PersonPoseResult? pose,
        int frameWidth,
        int frameHeight,
        int candidateIndex = 0)
    {
        ArgumentNullException.ThrowIfNull(detection);

        float width = detection.Box.Width;
        float height = detection.Box.Height;
        float area = width * height;
        float aspectRatio = width / Math.Max(1f, height);
        float frameArea = Math.Max(1f, frameWidth * frameHeight);
        float frameAreaFraction = area / frameArea;

        // 1. Geometric Gating Bounds
        float minAbsHeight = _options.MinCropHeight > 0 ? _options.MinCropHeight : 60f;
        float minRelHeight = frameHeight > 0 ? (frameHeight * 0.035f) : 40f;
        float effectiveMinHeight = Math.Min(minAbsHeight, Math.Max(35f, minRelHeight));

        float minAbsWidth = _options.MinCropWidth > 0 ? _options.MinCropWidth : 24f;
        float minAbsArea = _options.MinCropArea > 0 ? _options.MinCropArea : 1800f;
        float effectiveMinArea = Math.Min(minAbsArea, Math.Max(1200f, effectiveMinHeight * minAbsWidth * 0.8f));

        // 2. Extract Pose Signals & Keypoint Counts
        var keypoints = pose?.Keypoints ?? new List<Keypoint>();
        float kpMinConf = _options.MinKeypointConfidence > 0 ? _options.MinKeypointConfidence : 0.30f;

        var validKps = keypoints.Where(k => k.Confidence >= kpMinConf).ToList();
        int validKeypointCount = validKps.Count;

        // Head (0..4), Upper (5..10), Lower (11..16)
        var headKps = validKps.Where(k => k.Index >= 0 && k.Index <= 4).ToList();
        var upperKps = validKps.Where(k => k.Index >= 5 && k.Index <= 10).ToList();
        var torsoKps = validKps.Where(k => k.Index == 5 || k.Index == 6 || k.Index == 11 || k.Index == 12).ToList();
        var lowerKps = validKps.Where(k => k.Index >= 11 && k.Index <= 16).ToList();
        var wristOnlyKps = validKps.Where(k => k.Index == 9 || k.Index == 10).ToList();

        bool hasHead = headKps.Count > 0;
        bool hasShoulders = validKps.Any(k => k.Index == 5 || k.Index == 6);
        bool hasHips = validKps.Any(k => k.Index == 11 || k.Index == 12);
        bool hasTorso = torsoKps.Count > 0;
        bool hasLimbs = validKps.Any(k => (k.Index >= 7 && k.Index <= 10) || (k.Index >= 13 && k.Index <= 16));

        bool isForegroundPerson = frameAreaFraction >= 0.06f || area >= 10000f;
        bool hasUpperBodyEvidence = hasHead || hasShoulders || (validKeypointCount >= 2 && (hasTorso || hasLimbs));
        bool hasPortraitEvidence = hasHead || hasShoulders || isForegroundPerson;

        // 3. Evaluate Anatomical Consistency Score
        float anatomicalScore = ComputeAnatomicalConsistency(validKps, detection.Box);

        // 4. Decision Logic & Rejection Checks
        bool isValid = false;
        var tier = PersonCandidateTier.Rejected;
        string rejectReason = string.Empty;
        float validationScore = 0f;

        // Rule A: Check for invalid horizontal or extreme aspect ratio
        // Supports close-up portrait / webcam bust crops (up to 1.85) when upper body evidence or large frame occupancy exists
        if (aspectRatio > 1.85f)
        {
            rejectReason = $"InvalidExtremeHorizontalAspectRatio ({aspectRatio:F2} > 1.85)";
        }
        else if (aspectRatio > 1.25f && !hasPortraitEvidence && validKeypointCount < 2 && detection.Confidence < 0.65f)
        {
            rejectReason = $"InvalidHorizontalAspectRatioWithoutUpperBody ({aspectRatio:F2} > 1.25, lacking head/shoulder/foreground cues)";
        }
        else if (aspectRatio < 0.08f)
        {
            rejectReason = $"InvalidVerticalAspectRatio ({aspectRatio:F2} < 0.08, overly narrow)";
        }
        // Rule B: Check for tiny cup/hand sized crops below minimum area/height
        else if ((height < effectiveMinHeight || area < effectiveMinArea) && validKeypointCount < 3 && detection.Confidence < 0.65f)
        {
            rejectReason = $"BelowMinimumSizeThreshold (Height: {height:F0}px < {effectiveMinHeight:F0}px, Area: {area:F0}px² < {effectiveMinArea:F0}px²)";
        }
        // Rule C: Isolated wrist/hand or noise artifact without human torso or head structure
        else if (wristOnlyKps.Count > 0 && !hasHead && !hasShoulders && !hasHips && validKeypointCount <= 2 && detection.Confidence < 0.68f)
        {
            rejectReason = $"IsolatedHandOrCupArtifact (Wrist/hand keypoints only without head or torso, Conf: {detection.Confidence:P0})";
        }
        // Rule D: Low YOLO confidence with zero or near-zero human keypoints on small background candidate (suppresses false positive ghosts)
        else if (validKeypointCount == 0 && !isForegroundPerson && (detection.Confidence < 0.58f || height < 80f || area < 3000f))
        {
            rejectReason = $"InsufficientHumanPoseEvidence (0 keypoints on small background candidate with YOLO conf {detection.Confidence:P0})";
        }
        // Rule E: Inverted or nonsensical skeleton geometry
        else if (validKeypointCount >= 3 && anatomicalScore < 0.20f && detection.Confidence < 0.60f)
        {
            rejectReason = $"InconsistentAnatomicalStructure (Anatomical score: {anatomicalScore:F2})";
        }
        else
        {
            // Candidate passes negative filters; classify into tiers
            if ((detection.Confidence >= 0.45f && (validKeypointCount >= 2 || hasUpperBodyEvidence || isForegroundPerson)) ||
                detection.Confidence >= 0.70f)
            {
                isValid = true;
                tier = PersonCandidateTier.StrongPerson;
                validationScore = (detection.Confidence * 0.50f) + (Math.Min(1.0f, validKeypointCount / 4.0f) * 0.25f) + (anatomicalScore * 0.25f);
            }
            else if ((detection.Confidence >= 0.38f || (validKeypointCount >= 2 && (hasTorso || hasHead || hasShoulders)) || isForegroundPerson) &&
                     aspectRatio >= 0.08f && aspectRatio <= 1.85f &&
                     (validKeypointCount >= 1 || height >= 70f || isForegroundPerson))
            {
                isValid = true;
                tier = PersonCandidateTier.ProbablePerson;
                validationScore = (detection.Confidence * 0.45f) + (Math.Min(1.0f, validKeypointCount / 3.0f) * 0.30f) + (anatomicalScore * 0.25f);
            }
            else
            {
                rejectReason = $"WeakMultiSignalConfidence (YOLO: {detection.Confidence:P0}, Keypoints: {validKeypointCount}, Score: {validationScore:F2})";
            }
        }

        var result = new PersonCandidateValidationResult
        {
            CandidateIndex = candidateIndex,
            IsValidPerson = isValid,
            ValidationScore = validationScore,
            Tier = tier,
            RejectReason = rejectReason,
            Width = width,
            Height = height,
            Area = area,
            AspectRatio = aspectRatio,
            FrameAreaFraction = frameAreaFraction,
            ValidKeypointCount = validKeypointCount,
            UpperBodyKeypointCount = upperKps.Count,
            LowerBodyKeypointCount = lowerKps.Count,
            AnatomicalScore = anatomicalScore,
            HasHeadEvidence = hasHead,
            HasShoulderEvidence = hasShoulders,
            HasTorsoEvidence = hasTorso,
            HasLimbEvidence = hasLimbs,
            YoloConfidence = detection.Confidence,
            BoundingBox = detection.Box
        };

        if (isValid)
        {
            _logger.LogDebug(
                "Candidate #{Idx} VALIDATED as {Tier} (Score: {Score:F2}, YOLO: {Yolo:P0}, KPs: {KPs}, Box: {W}x{H})",
                candidateIndex, tier, validationScore, detection.Confidence, validKeypointCount, (int)width, (int)height);
        }
        else
        {
            _logger.LogInformation(
                "Candidate #{Idx} REJECTED: {Reason} (YOLO: {Yolo:P0}, KPs: {KPs}, Box: {W}x{H})",
                candidateIndex, rejectReason, detection.Confidence, validKeypointCount, (int)width, (int)height);
        }

        return result;
    }

    public IReadOnlyList<ValidatedPersonCandidate> FilterValidCandidates(
        IReadOnlyList<DetectionResult> detections,
        IReadOnlyList<PersonPoseResult>? poses,
        int frameWidth,
        int frameHeight)
    {
        if (detections == null || detections.Count == 0)
        {
            return Array.Empty<ValidatedPersonCandidate>();
        }

        // Pair each detection with its best spatial pose
        var detectionToPoseMap = new Dictionary<int, PersonPoseResult>();
        if (poses != null && poses.Count > 0)
        {
            var matchedPoseIndices = new HashSet<int>();
            for (int d = 0; d < detections.Count; d++)
            {
                var det = detections[d];
                int bestPoseIdx = -1;
                float bestIoU = 0f;

                for (int p = 0; p < poses.Count; p++)
                {
                    if (matchedPoseIndices.Contains(p)) continue;
                    float iou = CalculateIoU(det.Box, poses[p].BoundingBox);
                    if (iou > bestIoU)
                    {
                        bestIoU = iou;
                        bestPoseIdx = p;
                    }
                }

                if (bestPoseIdx >= 0 && bestIoU >= 0.15f)
                {
                    detectionToPoseMap[d] = poses[bestPoseIdx];
                    matchedPoseIndices.Add(bestPoseIdx);
                }
            }
        }

        var validatedList = new List<ValidatedPersonCandidate>();

        for (int d = 0; d < detections.Count; d++)
        {
            var det = detections[d];
            var pose = detectionToPoseMap.GetValueOrDefault(d) ?? new PersonPoseResult
            {
                BoundingBox = det.Box,
                DetectionConfidence = det.Confidence,
                Visibility = new BodyVisibilityResult { HeadVisible = true, UpperBodyVisible = true, Orientation = PoseOrientation.Uncertain }
            };

            var validationResult = ValidateCandidate(det, pose, frameWidth, frameHeight, candidateIndex: d + 1);

            if (validationResult.IsValidPerson)
            {
                validatedList.Add(new ValidatedPersonCandidate
                {
                    OriginalCandidateIndex = d,
                    Detection = det,
                    Pose = pose,
                    ValidationResult = validationResult
                });
            }
        }

        return validatedList;
    }

    private static float ComputeAnatomicalConsistency(List<Keypoint> keypoints, BoundingBox box)
    {
        if (keypoints == null || keypoints.Count < 2) return 0.50f; // Neutral default for distant/partial

        int checks = 0;
        int passed = 0;

        var nose = keypoints.FirstOrDefault(k => k.Index == 0);
        var ls = keypoints.FirstOrDefault(k => k.Index == 5);
        var rs = keypoints.FirstOrDefault(k => k.Index == 6);
        var lh = keypoints.FirstOrDefault(k => k.Index == 11);
        var rh = keypoints.FirstOrDefault(k => k.Index == 12);
        var lk = keypoints.FirstOrDefault(k => k.Index == 13);
        var rk = keypoints.FirstOrDefault(k => k.Index == 14);
        var la = keypoints.FirstOrDefault(k => k.Index == 15);
        var ra = keypoints.FirstOrDefault(k => k.Index == 16);

        // Check 1: Head above shoulders
        float shoulderY = (ls != null && rs != null) ? (ls.Y + rs.Y) / 2f : (ls?.Y ?? rs?.Y ?? -1f);
        if (nose != null && shoulderY > 0)
        {
            checks++;
            if (nose.Y <= shoulderY + (box.Height * 0.15f)) passed++;
        }

        // Check 2: Shoulders above hips
        float hipY = (lh != null && rh != null) ? (lh.Y + rh.Y) / 2f : (lh?.Y ?? rh?.Y ?? -1f);
        if (shoulderY > 0 && hipY > 0)
        {
            checks++;
            if (shoulderY <= hipY + (box.Height * 0.15f)) passed++;
        }

        // Check 3: Hips above knees
        float kneeY = (lk != null && rk != null) ? (lk.Y + rk.Y) / 2f : (lk?.Y ?? rk?.Y ?? -1f);
        if (hipY > 0 && kneeY > 0)
        {
            checks++;
            if (hipY <= kneeY + (box.Height * 0.15f)) passed++;
        }

        // Check 4: Knees above ankles
        float ankleY = (la != null && ra != null) ? (la.Y + ra.Y) / 2f : (la?.Y ?? ra?.Y ?? -1f);
        if (kneeY > 0 && ankleY > 0)
        {
            checks++;
            if (kneeY <= ankleY + (box.Height * 0.15f)) passed++;
        }

        // Check 5: Shoulder separation width is sane
        if (ls != null && rs != null)
        {
            checks++;
            float shoulderDist = Math.Abs(ls.X - rs.X);
            if (shoulderDist >= (box.Width * 0.10f) && shoulderDist <= (box.Width * 1.20f)) passed++;
        }

        return checks > 0 ? (float)passed / checks : 0.60f;
    }

    private static float CalculateIoU(BoundingBox boxA, BoundingBox boxB)
    {
        float xA = Math.Max(boxA.X, boxB.X);
        float yA = Math.Max(boxA.Y, boxB.Y);
        float xB = Math.Min(boxA.X + boxA.Width, boxB.X + boxB.Width);
        float yB = Math.Min(boxA.Y + boxA.Height, boxB.Y + boxB.Height);

        float interArea = Math.Max(0, xB - xA) * Math.Max(0, yB - yA);
        if (interArea <= 0) return 0.0f;

        float boxAArea = boxA.Width * boxA.Height;
        float boxBArea = boxB.Width * boxB.Height;
        float unionArea = boxAArea + boxBArea - interArea;

        return unionArea > 0 ? interArea / unionArea : 0.0f;
    }
}
