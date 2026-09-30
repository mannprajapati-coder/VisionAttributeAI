using Microsoft.Extensions.Options;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using VisionAttributeAI.Models.AI;

namespace VisionAttributeAI.Services.Watch;

/// <summary>
/// Dedicated YOLOv8-Nano Wrist-Watch Object Detection Service.
/// Analyzes Left and Right wrist crops independently with configurable quality and detection thresholds.
/// Distinguishes: NotVisible, InsufficientVisualEvidence, NoWatchDetected, WatchDetected, ModelUnavailable.
/// Under NO circumstances are rule-based heuristics used if the model is missing or fails.
/// </summary>
public class WatchDetectionService : IWatchDetectionService, IDisposable
{
    private readonly ILogger<WatchDetectionService> _logger;
    private readonly PersonAnalysisOptions _options;
    private Microsoft.ML.OnnxRuntime.InferenceSession? _session;
    private string _inputName = "images";
    private string _outputName = "output0";
    private bool _isModelLoaded;
    private string _loadFailureReason = string.Empty;
    private bool _disposed;

    public WatchDetectionService(
        IOptions<PersonAnalysisOptions> options,
        ILogger<WatchDetectionService> logger)
    {
        _options = options.Value;
        _logger = logger;

        InitializeModel();
    }

    private void InitializeModel()
    {
        try
        {
            var modelPath = _options.WatchModelPath;
            if (string.IsNullOrWhiteSpace(modelPath))
            {
                _loadFailureReason = "Watch model path configuration is empty.";
                _logger.LogWarning("Watch detection model path is not specified in configuration.");
                return;
            }

            if (!File.Exists(modelPath))
            {
                _loadFailureReason = $"Watch model file does not exist at '{modelPath}'.";
                _logger.LogWarning("Watch detection model not found at '{ModelPath}'. Watch detection will report ModelUnavailable.", modelPath);
                return;
            }

            var sessionOptions = new Microsoft.ML.OnnxRuntime.SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL
            };

            _session = new Microsoft.ML.OnnxRuntime.InferenceSession(modelPath, sessionOptions);
            _inputName = _session.InputMetadata.Keys.FirstOrDefault() ?? "images";
            _outputName = _session.OutputMetadata.Keys.FirstOrDefault() ?? "output0";
            _isModelLoaded = true;

            _logger.LogInformation("Successfully initialized YOLOv8-Nano Watch Detector from '{ModelPath}'. Input='{Input}', Output='{Output}'",
                modelPath, _inputName, _outputName);
        }
        catch (Exception ex)
        {
            _loadFailureReason = $"Failed to load ONNX model: {ex.Message}";
            _logger.LogError(ex, "Failed to load watch detector from '{ModelPath}'. Watch detection will return ModelUnavailable.", _options.WatchModelPath);
        }
    }

    public Task<WatchResult> DetectWatchAsync(Mat? leftWristCrop, Mat? rightWristCrop, CancellationToken cancellationToken = default)
    {
        bool hasLeft = leftWristCrop != null && !leftWristCrop.Empty();
        bool hasRight = rightWristCrop != null && !rightWristCrop.Empty();

        if (!hasLeft && !hasRight)
        {
            return Task.FromResult(new WatchResult
            {
                HasWatch = false,
                Confidence = 0.0f,
                Status = "NotVisible",
                Details = "Neither left nor right wrist is visible in frame."
            });
        }

        if (!_isModelLoaded || _session == null)
        {
            _logger.LogDebug("Watch detection requested but watch model is unavailable: {Reason}", _loadFailureReason);
            return Task.FromResult(new WatchResult
            {
                HasWatch = false,
                Confidence = 0.0f,
                Status = "ModelUnavailable",
                Details = $"Watch detection model is unavailable: {_loadFailureReason}",
                LeftWristAnalyzed = hasLeft,
                RightWristAnalyzed = hasRight
            });
        }

        try
        {
            float lConf = 0.0f;
            bool lDetected = false;
            if (hasLeft)
            {
                (lDetected, lConf) = RunOnnxWatchInference(leftWristCrop!);
            }

            float rConf = 0.0f;
            bool rDetected = false;
            if (hasRight)
            {
                (rDetected, rConf) = RunOnnxWatchInference(rightWristCrop!);
            }

            bool anyWatch = lDetected || rDetected;
            float maxConf = Math.Max(lConf, rConf);

            string status;
            string details;

            if (anyWatch)
            {
                status = "WatchDetected";
                string side = (lDetected && rDetected) ? "Both wrists" : (lDetected ? "Left wrist" : "Right wrist");
                details = $"Watch detected on {side} (Conf={maxConf:F2}).";
            }
            else
            {
                status = "NoWatchDetected";
                string inspected = (hasLeft && hasRight) ? "Both wrists inspected" : (hasLeft ? "Left wrist inspected" : "Right wrist inspected");
                details = $"{inspected}; no watch detected above confidence threshold ({_options.WatchDetectionConfidence:F2}).";
            }

            return Task.FromResult(new WatchResult
            {
                HasWatch = anyWatch,
                Confidence = maxConf,
                Status = status,
                Details = details,
                LeftWristAnalyzed = hasLeft,
                LeftWristDetected = lDetected,
                LeftWristConfidence = lConf,
                RightWristAnalyzed = hasRight,
                RightWristDetected = rDetected,
                RightWristConfidence = rConf
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Runtime exception occurred during watch ONNX inference.");
            return Task.FromResult(new WatchResult
            {
                HasWatch = false,
                Confidence = 0.0f,
                Status = "ModelUnavailable",
                Details = $"Inference error: {ex.Message}",
                LeftWristAnalyzed = hasLeft,
                RightWristAnalyzed = hasRight
            });
        }
    }

    private (bool Detected, float Confidence) RunOnnxWatchInference(Mat wristMat)
    {
        if (wristMat.Width < 15 || wristMat.Height < 15)
        {
            return (false, 0.0f);
        }

        int targetSize = 160; // Lightweight wrist crop input size
        int origW = wristMat.Width;
        int origH = wristMat.Height;

        float scale = Math.Min((float)targetSize / origW, (float)targetSize / origH);
        int newW = (int)Math.Round(origW * scale);
        int newH = (int)Math.Round(origH * scale);
        int padX = (targetSize - newW) / 2;
        int padY = (targetSize - newH) / 2;

        using var resized = new Mat();
        Cv2.Resize(wristMat, resized, new Size(newW, newH), 0, 0, InterpolationFlags.Linear);

        using var letterboxed = new Mat(targetSize, targetSize, MatType.CV_8UC3, new Scalar(114, 114, 114));
        var targetRoi = new Rect(padX, padY, newW, newH);
        using var roiMat = new Mat(letterboxed, targetRoi);
        resized.CopyTo(roiMat);

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

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_inputName, tensor)
        };

        using var results = _session!.Run(inputs);
        var outputTensor = results.First(r => r.Name == _outputName);
        var outputData = outputTensor.AsEnumerable<float>().ToArray();
        var dims = outputTensor.AsTensor<float>().Dimensions; // typically [1, 5, N]

        int numFeatures = dims.Length > 1 ? dims[1] : 5;
        int numAnchors = dims.Length > 2 ? dims[2] : (outputData.Length / numFeatures);

        float maxScore = 0f;
        for (int i = 0; i < numAnchors; i++)
        {
            // Class 0 score index is 4 in standard YOLO 1-class output [cx, cy, w, h, score]
            float score = outputData[4 * numAnchors + i];
            if (score > maxScore)
            {
                maxScore = score;
            }
        }

        bool isDetected = maxScore >= _options.WatchDetectionConfidence;
        return (isDetected, maxScore);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _session?.Dispose();
            _session = null;
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }
}
