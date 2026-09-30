using Microsoft.Extensions.Options;
using OpenCvSharp;
using VisionAttributeAI.Models.AI;
using VisionAttributeAI.Models.Pose;
using VisionAttributeAI.Models.Tracking;

namespace VisionAttributeAI.Services.Quality;

/// <summary>
/// Computes independent, attribute-specific visual quality scores [0.0 .. 1.0] and enforces resolution & pose evidence gating.
/// Ensures YOLO confidence does NOT dominate attribute scores.
/// </summary>
public class AttributeQualityEvaluator : IAttributeQualityEvaluator
{
    private readonly PersonAnalysisOptions _options;

    public AttributeQualityEvaluator(IOptions<PersonAnalysisOptions> options)
    {
        _options = options.Value;
    }

    public AttributeQualityScores EvaluateQuality(
        Mat frameMat,
        BoundingBox personBox,
        float yoloConfidence,
        PersonPoseResult pose,
        int frameWidth,
        int frameHeight)
    {
        var vis = pose.Visibility;
        var reg = pose.Regions;
        var kps = pose.Keypoints ?? new List<Keypoint>();

        // Keypoint lookups: 0:Nose, 1:LEye, 2:REye, 3:LEar, 4:REar, 5:LShoulder, 6:RShoulder, 7:LElbow, 8:RElbow, 9:LWrist, 10:RWrist, 11:LHip, 12:RHip, 13:LKnee, 14:RKnee, 15:LAnkle, 16:RAnkle
        float noseConf = GetKeypointConf(kps, 0);
        float lEyeConf = GetKeypointConf(kps, 1);
        float rEyeConf = GetKeypointConf(kps, 2);
        float lEarConf = GetKeypointConf(kps, 3);
        float rEarConf = GetKeypointConf(kps, 4);

        float lShoulderConf = GetKeypointConf(kps, 5);
        float rShoulderConf = GetKeypointConf(kps, 6);
        float lElbowConf = GetKeypointConf(kps, 7);
        float rElbowConf = GetKeypointConf(kps, 8);
        float lWristConf = GetKeypointConf(kps, 9);
        float rWristConf = GetKeypointConf(kps, 10);

        float lHipConf = GetKeypointConf(kps, 11);
        float rHipConf = GetKeypointConf(kps, 12);
        float lKneeConf = GetKeypointConf(kps, 13);
        float rKneeConf = GetKeypointConf(kps, 14);
        float lAnkleConf = GetKeypointConf(kps, 15);
        float rAnkleConf = GetKeypointConf(kps, 16);

        // -------------------------------------------------------------
        // 1. Appearance Quality (Head / Face / Orientation)
        // -------------------------------------------------------------
        float appQuality = 0f;
        bool hasAppEvidence = false;
        string appRejection = string.Empty;

        if (vis.HeadVisible && reg.HeadRegion != null)
        {
            var headBox = reg.HeadRegion;
            if (headBox.Width < _options.MinHeadWidth || headBox.Height < _options.MinHeadHeight)
            {
                appRejection = $"Head resolution too small ({Math.Round(headBox.Width)}x{Math.Round(headBox.Height)} < {_options.MinHeadWidth}x{_options.MinHeadHeight})";
            }
            else if (vis.Orientation == PoseOrientation.Back)
            {
                appRejection = "Rear-facing pose (insufficient facial/identity cues)";
            }
            else
            {
                // Orientation weight factor: Front=1.0, NearFrontal=0.90, Side=0.55, Uncertain=0.45
                float orientFactor = vis.Orientation switch
                {
                    PoseOrientation.Front => 1.0f,
                    PoseOrientation.NearFrontal => 0.90f,
                    PoseOrientation.Side => 0.55f,
                    _ => 0.45f
                };

                // Head keypoint confidence factor
                float headKpAvg = (noseConf * 0.4f) + ((lEyeConf + rEyeConf) / 2f * 0.35f) + ((lEarConf + rEarConf) / 2f * 0.25f);
                headKpAvg = Math.Clamp(headKpAvg, 0.2f, 1.0f);

                // Resolution factor [0.4 .. 1.0]
                float headResFactor = Math.Clamp((headBox.Width * headBox.Height) / 5000f, 0.4f, 1.0f);

                // Sharpness factor
                double headSharpness = MeasureSharpness(frameMat, headBox);
                float headSharpFactor = Math.Clamp((float)(headSharpness / 120.0), 0.3f, 1.0f);

                // Clipping factor
                float clipFactor = vis.HeadClipped ? 0.65f : 1.0f;

                // Composite Appearance Quality Score (0..1)
                appQuality = (orientFactor * 0.40f) + (headKpAvg * 0.25f) + (headSharpFactor * 0.20f) + (headResFactor * 0.15f);
                appQuality = Math.Clamp(appQuality * clipFactor, 0.05f, 1.0f);

                if (appQuality >= 0.30f)
                {
                    hasAppEvidence = true;
                }
                else
                {
                    appRejection = $"Low appearance visual quality ({appQuality:F2} < 0.30)";
                }
            }
        }
        else
        {
            appRejection = "Head region not visible";
        }

        // -------------------------------------------------------------
        // 2. Upper Torso Quality
        // -------------------------------------------------------------
        float upperQuality = 0f;
        bool hasUpperEvidence = false;
        string upperRejection = string.Empty;

        if (vis.UpperBodyVisible && reg.UpperTorsoRegion != null)
        {
            var torsoBox = reg.UpperTorsoRegion;
            if (torsoBox.Width < _options.MinTorsoWidth || torsoBox.Height < _options.MinTorsoHeight)
            {
                upperRejection = $"Torso resolution too small ({Math.Round(torsoBox.Width)}x{Math.Round(torsoBox.Height)} < {_options.MinTorsoWidth}x{_options.MinTorsoHeight})";
            }
            else
            {
                // Shoulder & Hip keypoint evidence
                float shoulderAvg = (lShoulderConf + rShoulderConf) / 2f;
                float hipAvg = (lHipConf + rHipConf) / 2f;
                float torsoKpFactor = (shoulderAvg * 0.65f) + (Math.Max(0.2f, hipAvg) * 0.35f);
                torsoKpFactor = Math.Clamp(torsoKpFactor, 0.2f, 1.0f);

                // Torso area & sharpness
                float torsoAreaFactor = Math.Clamp((torsoBox.Width * torsoBox.Height) / 15000f, 0.4f, 1.0f);
                double torsoSharpness = MeasureSharpness(frameMat, torsoBox);
                float torsoSharpFactor = Math.Clamp((float)(torsoSharpness / 120.0), 0.3f, 1.0f);

                // Clipping factor
                float clipFactor = vis.UpperClipped ? 0.70f : 1.0f;

                upperQuality = (torsoKpFactor * 0.40f) + (torsoSharpFactor * 0.35f) + (torsoAreaFactor * 0.25f);
                upperQuality = Math.Clamp(upperQuality * clipFactor, 0.05f, 1.0f);

                if (upperQuality >= 0.30f)
                {
                    hasUpperEvidence = true;
                }
                else
                {
                    upperRejection = $"Low upper torso quality ({upperQuality:F2} < 0.30)";
                }
            }
        }
        else
        {
            upperRejection = "Upper torso not visible";
        }

        // -------------------------------------------------------------
        // 3. Lower Body Quality
        // -------------------------------------------------------------
        float lowerQuality = 0f;
        bool hasLowerEvidence = false;
        string lowerRejection = string.Empty;

        if (vis.LowerBodyVisible && reg.LowerBodyRegion != null)
        {
            var lowerBox = reg.LowerBodyRegion;
            if (lowerBox.Width < _options.MinLowerWidth || lowerBox.Height < _options.MinLowerHeight)
            {
                lowerRejection = $"Lower body resolution too small ({Math.Round(lowerBox.Width)}x{Math.Round(lowerBox.Height)} < {_options.MinLowerWidth}x{_options.MinLowerHeight})";
            }
            else
            {
                float hipAvg = (lHipConf + rHipConf) / 2f;
                float kneeAvg = (lKneeConf + rKneeConf) / 2f;
                float ankleAvg = (lAnkleConf + rAnkleConf) / 2f;
                float lowerKpFactor = (hipAvg * 0.35f) + (kneeAvg * 0.40f) + (ankleAvg * 0.25f);
                lowerKpFactor = Math.Clamp(lowerKpFactor, 0.2f, 1.0f);

                float lowerAreaFactor = Math.Clamp((lowerBox.Width * lowerBox.Height) / 18000f, 0.4f, 1.0f);
                double lowerSharpness = MeasureSharpness(frameMat, lowerBox);
                float lowerSharpFactor = Math.Clamp((float)(lowerSharpness / 120.0), 0.3f, 1.0f);
                float clipFactor = vis.LowerClipped ? 0.65f : 1.0f;

                lowerQuality = (lowerKpFactor * 0.45f) + (lowerSharpFactor * 0.30f) + (lowerAreaFactor * 0.25f);
                lowerQuality = Math.Clamp(lowerQuality * clipFactor, 0.05f, 1.0f);

                if (lowerQuality >= 0.30f)
                {
                    hasLowerEvidence = true;
                }
                else
                {
                    lowerRejection = $"Low lower body quality ({lowerQuality:F2} < 0.30)";
                }
            }
        }
        else
        {
            lowerRejection = "Lower body not visible (waist-up framing / occluded)";
        }

        // -------------------------------------------------------------
        // 4. Shoes / Feet Quality Gate
        // -------------------------------------------------------------
        float shoesQuality = 0f;
        bool hasShoesEvidence = false;
        string shoesRejection = string.Empty;
        float feetWidth = 0f;
        float feetHeight = 0f;
        double feetSharpness = 0.0;
        bool feetGatePassed = false;

        if (vis.FeetVisible && reg.FeetRegion != null)
        {
            var feetBox = reg.FeetRegion;
            feetWidth = feetBox.Width;
            feetHeight = feetBox.Height;
            feetSharpness = MeasureSharpness(frameMat, feetBox);
            float ankleAvg = Math.Max(lAnkleConf, rAnkleConf);

            if (feetWidth < _options.MinFeetWidth || feetHeight < _options.MinFeetHeight)
            {
                shoesRejection = $"Feet resolution too small ({Math.Round(feetWidth)}x{Math.Round(feetHeight)} < {_options.MinFeetWidth}x{_options.MinFeetHeight})";
            }
            else if (vis.FeetClipped)
            {
                shoesRejection = "Feet clipped by lower frame boundary";
            }
            else if (feetSharpness < _options.MinFeetSharpness)
            {
                shoesRejection = $"Feet sharpness too low ({feetSharpness:F1} < {_options.MinFeetSharpness:F1})";
            }
            else if (ankleAvg < _options.MinFeetPoseConfidence)
            {
                shoesRejection = $"Feet pose keypoint confidence too low ({ankleAvg:F2} < {_options.MinFeetPoseConfidence:F2})";
            }
            else
            {
                float feetKpFactor = Math.Clamp(ankleAvg, 0.2f, 1.0f);
                float feetSharpFactor = Math.Clamp((float)(feetSharpness / 100.0), 0.3f, 1.0f);

                shoesQuality = (feetKpFactor * 0.55f) + (feetSharpFactor * 0.45f);
                shoesQuality = Math.Clamp(shoesQuality, 0.05f, 1.0f);

                if (shoesQuality >= 0.30f)
                {
                    hasShoesEvidence = true;
                    feetGatePassed = true;
                }
                else
                {
                    shoesRejection = $"Low footwear composite quality ({shoesQuality:F2} < 0.30)";
                }
            }
        }
        else
        {
            shoesRejection = "Feet region not visible in frame";
        }

        // -------------------------------------------------------------
        // 5. Wrist Quality Gate (Left & Right independently)
        // -------------------------------------------------------------
        float leftWristQuality = 0f;
        float rightWristQuality = 0f;
        bool hasWristEvidence = false;
        string wristRejection = string.Empty;

        float lWristW = 0f, lWristH = 0f;
        double lWristSharp = 0.0;
        bool lWristGatePassed = false;

        float rWristW = 0f, rWristH = 0f;
        double rWristSharp = 0.0;
        bool rWristGatePassed = false;

        if (vis.LeftWristVisible && reg.LeftWristRegion != null && !vis.LeftWristClipped)
        {
            var lBox = reg.LeftWristRegion;
            lWristW = lBox.Width;
            lWristH = lBox.Height;
            lWristSharp = MeasureSharpness(frameMat, lBox);

            if (lWristW >= _options.WatchMinRoiWidth && lWristH >= _options.WatchMinRoiHeight &&
                lWristSharp >= _options.WatchMinSharpness && lWristConf >= _options.WatchMinPoseConfidence)
            {
                float kpFactor = (lWristConf * 0.70f) + (lElbowConf * 0.30f);
                leftWristQuality = Math.Clamp((kpFactor * 0.6f) + ((float)(lWristSharp / 100.0) * 0.4f), 0.1f, 1.0f);
                lWristGatePassed = true;
            }
        }

        if (vis.RightWristVisible && reg.RightWristRegion != null && !vis.RightWristClipped)
        {
            var rBox = reg.RightWristRegion;
            rWristW = rBox.Width;
            rWristH = rBox.Height;
            rWristSharp = MeasureSharpness(frameMat, rBox);

            if (rWristW >= _options.WatchMinRoiWidth && rWristH >= _options.WatchMinRoiHeight &&
                rWristSharp >= _options.WatchMinSharpness && rWristConf >= _options.WatchMinPoseConfidence)
            {
                float kpFactor = (rWristConf * 0.70f) + (rElbowConf * 0.30f);
                rightWristQuality = Math.Clamp((kpFactor * 0.6f) + ((float)(rWristSharp / 100.0) * 0.4f), 0.1f, 1.0f);
                rWristGatePassed = true;
            }
        }

        if (lWristGatePassed || rWristGatePassed)
        {
            hasWristEvidence = true;
        }
        else
        {
            if (!vis.LeftWristVisible && !vis.RightWristVisible)
            {
                wristRejection = "Wrists not visible in frame";
            }
            else
            {
                wristRejection = $"Wrist ROI below minimum resolution ({_options.WatchMinRoiWidth}x{_options.WatchMinRoiHeight}px) or sharpness ({_options.WatchMinSharpness}) gate";
            }
        }

        return new AttributeQualityScores
        {
            AppearanceQuality = appQuality,
            UpperQuality = upperQuality,
            LowerQuality = lowerQuality,
            ShoesQuality = shoesQuality,
            LeftWristQuality = leftWristQuality,
            RightWristQuality = rightWristQuality,
            FeetWidth = feetWidth,
            FeetHeight = feetHeight,
            FeetSharpness = feetSharpness,
            FeetGatePassed = feetGatePassed,
            LeftWristWidth = lWristW,
            LeftWristHeight = lWristH,
            LeftWristSharpness = lWristSharp,
            LeftWristGatePassed = lWristGatePassed,
            RightWristWidth = rWristW,
            RightWristHeight = rWristH,
            RightWristSharpness = rWristSharp,
            RightWristGatePassed = rWristGatePassed,
            HasAppearanceEvidence = hasAppEvidence,
            HasUpperEvidence = hasUpperEvidence,
            HasLowerEvidence = hasLowerEvidence,
            HasShoesEvidence = hasShoesEvidence,
            HasWristEvidence = hasWristEvidence,
            AppearanceRejectionReason = appRejection,
            UpperRejectionReason = upperRejection,
            LowerRejectionReason = lowerRejection,
            ShoesRejectionReason = shoesRejection,
            WristRejectionReason = wristRejection
        };
    }

    private static float GetKeypointConf(List<Keypoint> kps, int index)
    {
        if (index >= 0 && index < kps.Count)
        {
            return kps[index].Confidence;
        }
        return 0f;
    }

    private static double MeasureSharpness(Mat src, BoundingBox box)
    {
        try
        {
            int x = (int)Math.Max(0, Math.Floor(box.X));
            int y = (int)Math.Max(0, Math.Floor(box.Y));
            int w = (int)Math.Min(src.Width - x, Math.Ceiling(box.Width));
            int h = (int)Math.Min(src.Height - y, Math.Ceiling(box.Height));

            if (w <= 4 || h <= 4) return 0.0;

            using var roi = new Mat(src, new Rect(x, y, w, h));
            using var gray = new Mat();

            if (roi.Channels() == 1)
                roi.CopyTo(gray);
            else
                Cv2.CvtColor(roi, gray, ColorConversionCodes.BGR2GRAY);

            using var laplacian = new Mat();
            Cv2.Laplacian(gray, laplacian, MatType.CV_64F);

            Cv2.MeanStdDev(laplacian, out _, out Scalar stddev);
            return stddev.Val0 * stddev.Val0;
        }
        catch
        {
            return 0.0;
        }
    }
}
