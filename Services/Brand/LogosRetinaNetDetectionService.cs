using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using VisionAttributeAI.Models.AI;
using VisionAttributeAI.Models.Brand;

namespace VisionAttributeAI.Services.Brand;

/// <summary>
/// LOGOS RetinaNet Object Detection Service for brand logo candidate localization.
/// Operates on 384x384 inputs and decodes 27,621 multi-scale anchor representations.
/// </summary>
public class LogosRetinaNetDetectionService : IBrandLogoDetectionService, IDisposable
{
    private const int InputSize = 384;
    private const int TotalAnchors = 27621;

    private readonly BrandOptions _options;
    private readonly ILogger<LogosRetinaNetDetectionService> _logger;
    private InferenceSession? _session;
    private string _inputName = "images";
    private readonly float[] _anchors; // Flattened [TotalAnchors * 4]
    private bool _disposed;

    // Normalization constants (ImageNet standard)
    private static readonly float[] Mean = [0.485f, 0.456f, 0.406f];
    private static readonly float[] Std = [0.229f, 0.224f, 0.225f];

    public bool IsModelLoaded => _session != null;

    public LogosRetinaNetDetectionService(
        IOptions<BrandOptions> options,
        ILogger<LogosRetinaNetDetectionService> logger)
    {
        _options = options.Value;
        _logger = logger;

        _anchors = GenerateAllAnchors();
        InitializeModel();
    }

    private void InitializeModel()
    {
        try
        {
            var modelPath = _options.LogosRetinaNetModelPath;
            if (string.IsNullOrWhiteSpace(modelPath) || !File.Exists(modelPath))
            {
                _logger.LogWarning("LOGOS RetinaNet model not found at '{ModelPath}'. Logo detection disabled.", modelPath);
                return;
            }

            var sessionOptions = new Microsoft.ML.OnnxRuntime.SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL
            };

            _session = new InferenceSession(modelPath, sessionOptions);
            _inputName = _session.InputMetadata.Keys.FirstOrDefault() ?? "images";

            _logger.LogInformation("Successfully initialized LOGOS RetinaNet from '{ModelPath}'. Input='{Input}'", modelPath, _inputName);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to load LOGOS RetinaNet model from '{ModelPath}'.", _options.LogosRetinaNetModelPath);
        }
    }

    public Task<IReadOnlyList<LogoCandidate>> DetectLogoCandidatesAsync(Mat regionCrop, string regionName, CancellationToken cancellationToken = default)
    {
        if (!_options.Enabled || !IsModelLoaded || _session == null || regionCrop.Empty())
        {
            return Task.FromResult<IReadOnlyList<LogoCandidate>>(Array.Empty<LogoCandidate>());
        }

        int origW = regionCrop.Width;
        int origH = regionCrop.Height;

        if (origW < _options.MinRegionDimension || origH < _options.MinRegionDimension)
        {
            return Task.FromResult<IReadOnlyList<LogoCandidate>>(Array.Empty<LogoCandidate>());
        }

        try
        {
            // 1. Resize to 384x384
            using var resizedMat = new Mat();
            Cv2.Resize(regionCrop, resizedMat, new Size(InputSize, InputSize), 0, 0, InterpolationFlags.Linear);

            // Convert BGR to RGB
            using var rgbMat = new Mat();
            Cv2.CvtColor(resizedMat, rgbMat, ColorConversionCodes.BGR2RGB);

            // 2. Build input tensor [1, 3, 384, 384]
            var tensor = new DenseTensor<float>(new[] { 1, 3, InputSize, InputSize });

            for (int y = 0; y < InputSize; y++)
            {
                for (int x = 0; x < InputSize; x++)
                {
                    var pixel = rgbMat.At<Vec3b>(y, x);
                    // Channel 0: R, 1: G, 2: B
                    tensor[0, 0, y, x] = (pixel.Item0 / 255f - Mean[0]) / Std[0];
                    tensor[0, 1, y, x] = (pixel.Item1 / 255f - Mean[1]) / Std[1];
                    tensor[0, 2, y, x] = (pixel.Item2 / 255f - Mean[2]) / Std[2];
                }
            }

            // 3. Inference
            var inputs = new[] { NamedOnnxValue.CreateFromTensor(_inputName, tensor) };
            using var outputs = _session.Run(inputs);

            var scoreOutput = outputs.FirstOrDefault(o => o.Name == "scores" || o.Name.Contains("score", StringComparison.OrdinalIgnoreCase))?.AsTensor<float>();
            var deltaOutput = outputs.FirstOrDefault(o => o.Name == "deltas" || o.Name.Contains("delta", StringComparison.OrdinalIgnoreCase) || o.Name.Contains("box", StringComparison.OrdinalIgnoreCase))?.AsTensor<float>();

            if (scoreOutput == null || deltaOutput == null)
            {
                // Fallback to position 0 and 1
                var outList = outputs.ToList();
                scoreOutput ??= outList[0].AsTensor<float>();
                deltaOutput ??= outList[1].AsTensor<float>();
            }

            // 4. Decode boxes and scores
            var rawBoxes = new List<Rect2d>();
            var rawScores = new List<float>();
            var rawClasses = new List<int>();

            float confThreshold = _options.DetectionConfidenceThreshold;

            for (int i = 0; i < TotalAnchors; i++)
            {
                float s0 = scoreOutput[0, i, 0]; // Class 0
                float s1 = scoreOutput[0, i, 1]; // Class 1
                float maxScore = Math.Max(s0, s1);
                int classId = s0 >= s1 ? 0 : 1;

                if (maxScore < confThreshold)
                {
                    continue;
                }

                // Decode box
                int anchorOffset = i * 4;
                float ax1 = _anchors[anchorOffset + 0];
                float ay1 = _anchors[anchorOffset + 1];
                float ax2 = _anchors[anchorOffset + 2];
                float ay2 = _anchors[anchorOffset + 3];

                float pw = ax2 - ax1;
                float ph = ay2 - ay1;
                float px = ax1 + 0.5f * pw;
                float py = ay1 + 0.5f * ph;

                float dx = deltaOutput[0, i, 0];
                float dy = deltaOutput[0, i, 1];
                float dw = deltaOutput[0, i, 2];
                float dh = deltaOutput[0, i, 3];

                float gx = px + pw * dx;
                float gy = py + ph * dy;
                float gw = pw * MathF.Exp(Math.Clamp(dw, -10f, 10f));
                float gh = ph * MathF.Exp(Math.Clamp(dh, -10f, 10f));

                float x1 = Math.Clamp(gx - 0.5f * gw, 0f, (float)InputSize);
                float y1 = Math.Clamp(gy - 0.5f * gh, 0f, (float)InputSize);
                float x2 = Math.Clamp(gx + 0.5f * gw, 0f, (float)InputSize);
                float y2 = Math.Clamp(gy + 0.5f * gh, 0f, (float)InputSize);

                float bw = x2 - x1;
                float bh = y2 - y1;

                if (bw > 2 && bh > 2)
                {
                    rawBoxes.Add(new Rect2d(x1, y1, bw, bh));
                    rawScores.Add(maxScore);
                    rawClasses.Add(classId);
                }
            }

            if (rawBoxes.Count == 0)
            {
                return Task.FromResult<IReadOnlyList<LogoCandidate>>(Array.Empty<LogoCandidate>());
            }

            // 5. NMS
            var cvBoxes = rawBoxes.Select(b => new Rect((int)b.X, (int)b.Y, (int)b.Width, (int)b.Height)).ToArray();
            CvDnn.NMSBoxes(cvBoxes, rawScores.ToArray(), confThreshold, _options.DetectionNmsThreshold, out int[] indices);

            if (indices == null || indices.Length == 0)
            {
                return Task.FromResult<IReadOnlyList<LogoCandidate>>(Array.Empty<LogoCandidate>());
            }

            // 6. Scale back to original region coordinates
            float scaleX = (float)origW / InputSize;
            float scaleY = (float)origH / InputSize;

            var candidates = new List<LogoCandidate>();
            foreach (int idx in indices)
            {
                var box = rawBoxes[idx];
                float origBoxX = (float)box.X * scaleX;
                float origBoxY = (float)box.Y * scaleY;
                float origBoxW = (float)box.Width * scaleX;
                float origBoxH = (float)box.Height * scaleY;

                // Clamp to region bounds
                origBoxX = Math.Clamp(origBoxX, 0f, origW - 1);
                origBoxY = Math.Clamp(origBoxY, 0f, origH - 1);
                origBoxW = Math.Clamp(origBoxW, 1f, origW - origBoxX);
                origBoxH = Math.Clamp(origBoxH, 1f, origH - origBoxY);

                if (origBoxW >= _options.MinLogoCropSize || origBoxH >= _options.MinLogoCropSize)
                {
                    candidates.Add(new LogoCandidate
                    {
                        Box = new BoundingBox(origBoxX, origBoxY, origBoxW, origBoxH),
                        Confidence = rawScores[idx],
                        ClassId = rawClasses[idx],
                        RegionName = regionName
                    });
                }
            }

            // Sort by confidence descending
            candidates = candidates.OrderByDescending(c => c.Confidence).ToList();
            return Task.FromResult<IReadOnlyList<LogoCandidate>>(candidates);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error executing LOGOS RetinaNet detection on region '{Region}'.", regionName);
            return Task.FromResult<IReadOnlyList<LogoCandidate>>(Array.Empty<LogoCandidate>());
        }
    }

    private static float[] GenerateAllAnchors()
    {
        int[] strides = [8, 16, 32, 64, 128];
        int[] baseSizes = [32, 64, 128, 256, 512];
        float[] octaveScales = [1.0f, MathF.Pow(2f, 1f / 3f), MathF.Pow(2f, 2f / 3f)];
        float[] ratios = [0.5f, 1.0f, 2.0f];

        var anchors = new float[TotalAnchors * 4];
        int anchorIndex = 0;

        for (int l = 0; l < strides.Length; l++)
        {
            int stride = strides[l];
            int baseSize = baseSizes[l];
            int featH = InputSize / stride;
            int featW = InputSize / stride;

            // Generate 9 base anchors for this level
            var baseAnchors = new (float w, float h)[9];
            int bIdx = 0;
            foreach (float r in ratios)
            {
                float wRatio = MathF.Sqrt(r);
                float hRatio = 1.0f / wRatio;
                foreach (float s in octaveScales)
                {
                    float w = baseSize * s / wRatio;
                    float h = baseSize * s * hRatio;
                    baseAnchors[bIdx++] = (w, h);
                }
            }

            for (int y = 0; y < featH; y++)
            {
                float centerY = (y + 0.5f) * stride;
                for (int x = 0; x < featW; x++)
                {
                    float centerX = (x + 0.5f) * stride;

                    for (int k = 0; k < 9; k++)
                    {
                        var (w, h) = baseAnchors[k];
                        anchors[anchorIndex++] = centerX - 0.5f * w;
                        anchors[anchorIndex++] = centerY - 0.5f * h;
                        anchors[anchorIndex++] = centerX + 0.5f * w;
                        anchors[anchorIndex++] = centerY + 0.5f * h;
                    }
                }
            }
        }

        return anchors;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _session?.Dispose();
            _session = null;
            _disposed = true;
        }
    }
}
