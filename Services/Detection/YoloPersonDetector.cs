using Microsoft.Extensions.Options;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using VisionAttributeAI.Models.AI;

namespace VisionAttributeAI.Services.Detection;

/// <summary>
/// Pretrained YOLO Object Detection Service for PERSON detection using Microsoft.ML.OnnxRuntime.
/// Automatically inspects model input/output metadata, performs aspect-ratio preserving letterboxing,
/// handles Float16/Float32 tensors, parses anchor/anchor-free outputs, and applies NMS.
/// </summary>
public class YoloPersonDetector : IPersonDetector, IObjectDetectionService
{
    private readonly ILogger<YoloPersonDetector> _logger;
    private readonly YoloModelOptions _options;
    private InferenceSession? _session;
    private bool _disposed;

    public bool IsModelLoaded => _session != null;
    public string InputName { get; private set; } = "images";
    public string OutputName { get; private set; } = "output0";
    public int[] InputDimensions { get; private set; } = [1, 3, 640, 640];
    public Type InputElementType { get; private set; } = typeof(float);
    public Type OutputElementType { get; private set; } = typeof(float);

    public YoloPersonDetector(IOptions<YoloModelOptions> options, ILogger<YoloPersonDetector> logger)
    {
        _logger = logger;
        _options = options.Value;

        InitializeModel();
    }

    private void InitializeModel()
    {
        try
        {
            if (string.IsNullOrWhiteSpace(_options.ModelPath) || !File.Exists(_options.ModelPath))
            {
                _logger.LogWarning("YOLO ONNX model not found at '{ModelPath}'. Detector is in offline/standby mode.", _options.ModelPath);
                return;
            }

            var sessionOptions = new Microsoft.ML.OnnxRuntime.SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL
            };

            _session = new InferenceSession(_options.ModelPath, sessionOptions);

            // Inspect and extract actual ONNX model metadata dynamically
            InspectModelMetadata();

            _logger.LogInformation(
                "Successfully loaded YOLO ONNX model '{ModelPath}'. Input='{Input}' ({InType}, [{InDim}]), Output='{Output}' ({OutType})",
                _options.ModelPath,
                InputName,
                InputElementType,
                string.Join(",", InputDimensions),
                OutputName,
                OutputElementType);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to initialize YOLO ONNX model session from '{ModelPath}'.", _options.ModelPath);
        }
    }

    private void InspectModelMetadata()
    {
        if (_session == null) return;

        // 1. Inspect Input Metadata
        var inputMeta = _session.InputMetadata;
        if (inputMeta.Count > 0)
        {
            var firstInput = inputMeta.First();
            InputName = firstInput.Key;
            InputElementType = firstInput.Value.ElementType;
            InputDimensions = firstInput.Value.Dimensions;

            // Update expected dimensions if defined in ONNX model
            if (InputDimensions.Length >= 4)
            {
                if (InputDimensions[2] > 0) _options.InputHeight = InputDimensions[2];
                if (InputDimensions[3] > 0) _options.InputWidth = InputDimensions[3];
            }
        }

        // 2. Inspect Output Metadata
        var outputMeta = _session.OutputMetadata;
        if (outputMeta.Count > 0)
        {
            var firstOutput = outputMeta.First();
            OutputName = firstOutput.Key;
            OutputElementType = firstOutput.Value.ElementType;
        }
    }

    public Task<IReadOnlyList<DetectionResult>> DetectPersonsAsync(Mat image, CancellationToken cancellationToken = default)
    {
        if (_session == null || image.Empty())
        {
            return Task.FromResult<IReadOnlyList<DetectionResult>>([]);
        }

        return Task.Run(() =>
        {
            cancellationToken.ThrowIfCancellationRequested();

            int origWidth = image.Width;
            int origHeight = image.Height;
            int targetWidth = _options.InputWidth;
            int targetHeight = _options.InputHeight;

            // 1. Letterbox Preprocessing (Aspect-Ratio Preserving with 114 Gray Padding)
            using var letterboxMat = Letterbox(image, targetWidth, targetHeight, out var letterboxInfo);

            // 2. Convert BGR to RGB
            using var rgbMat = new Mat();
            Cv2.CvtColor(letterboxMat, rgbMat, ColorConversionCodes.BGR2RGB);

            // 3. Build ONNX Tensor (Handling Float16 vs Float32 according to model requirements)
            NamedOnnxValue inputNamedValue = BuildInputTensor(rgbMat, targetWidth, targetHeight);

            // 4. Run ONNX Runtime Model Inference
            using var results = _session.Run([inputNamedValue]);
            using var outputValue = results.First(o => o.Name == OutputName);

            // 5. Parse Model Output & Filter for Person Detections
            var candidates = ParseModelOutput(outputValue, letterboxInfo, origWidth, origHeight);

            // 6. Apply Non-Maximum Suppression (NMS)
            var finalDetections = ApplyNonMaximumSuppression(candidates, origWidth, origHeight);

            _logger.LogDebug(
                "YOLO Detection on ({W}x{H}) -> {Count} candidate box(es), {FinalCount} person(s) after NMS.",
                origWidth,
                origHeight,
                candidates.Count,
                finalDetections.Count);

            return (IReadOnlyList<DetectionResult>)finalDetections;
        }, cancellationToken);
    }

    private NamedOnnxValue BuildInputTensor(Mat rgbMat, int width, int height)
    {
        if (InputElementType == typeof(Float16))
        {
            var tensor16 = new DenseTensor<Float16>([1, 3, height, width]);
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    var pixel = rgbMat.At<Vec3b>(y, x);
                    tensor16[0, 0, y, x] = (Float16)(pixel.Item0 / 255.0f); // R
                    tensor16[0, 1, y, x] = (Float16)(pixel.Item1 / 255.0f); // G
                    tensor16[0, 2, y, x] = (Float16)(pixel.Item2 / 255.0f); // B
                }
            }
            return NamedOnnxValue.CreateFromTensor(InputName, tensor16);
        }
        else
        {
            var tensor32 = new DenseTensor<float>([1, 3, height, width]);
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    var pixel = rgbMat.At<Vec3b>(y, x);
                    tensor32[0, 0, y, x] = pixel.Item0 / 255.0f; // R
                    tensor32[0, 1, y, x] = pixel.Item1 / 255.0f; // G
                    tensor32[0, 2, y, x] = pixel.Item2 / 255.0f; // B
                }
            }
            return NamedOnnxValue.CreateFromTensor(InputName, tensor32);
        }
    }

    private record LetterboxInfo(float Scale, float PadX, float PadY);

    /// <summary>
    /// Resizes an image preserving aspect ratio and pads remaining space with (114, 114, 114).
    /// </summary>
    private static Mat Letterbox(Mat src, int targetWidth, int targetHeight, out LetterboxInfo info)
    {
        float scale = Math.Min((float)targetWidth / src.Width, (float)targetHeight / src.Height);
        int unpaddedWidth = (int)Math.Round(src.Width * scale);
        int unpaddedHeight = (int)Math.Round(src.Height * scale);

        int padX = (targetWidth - unpaddedWidth) / 2;
        int padY = (targetHeight - unpaddedHeight) / 2;

        info = new LetterboxInfo(scale, padX, padY);

        using var resized = new Mat();
        Cv2.Resize(src, resized, new Size(unpaddedWidth, unpaddedHeight), 0, 0, InterpolationFlags.Linear);

        var letterboxed = new Mat(new Size(targetWidth, targetHeight), MatType.CV_8UC3, new Scalar(114, 114, 114));
        var roi = new Rect(padX, padY, unpaddedWidth, unpaddedHeight);
        using var destinationRoi = new Mat(letterboxed, roi);
        resized.CopyTo(destinationRoi);

        return letterboxed;
    }

    private record CandidateBox(Rect2d Rect, float Confidence);

    /// <summary>
    /// Parses model outputs dynamically supporting both YOLOv5 ([1, N, 85]) and YOLOv8 ([1, 84, N]) formats.
    /// </summary>
    private List<CandidateBox> ParseModelOutput(DisposableNamedOnnxValue outputValue, LetterboxInfo info, int origWidth, int origHeight)
    {
        var candidates = new List<CandidateBox>();

        if (OutputElementType == typeof(Float16))
        {
            var tensor = outputValue.AsTensor<Float16>();
            ParseTensor(tensor.Dimensions, (i0, i1, i2) => (float)tensor[i0, i1, i2], candidates, info, origWidth, origHeight);
        }
        else
        {
            var tensor = outputValue.AsTensor<float>();
            ParseTensor(tensor.Dimensions, (i0, i1, i2) => tensor[i0, i1, i2], candidates, info, origWidth, origHeight);
        }

        return candidates;
    }

    private void ParseTensor(
        ReadOnlySpan<int> dims,
        Func<int, int, int, float> getValue,
        List<CandidateBox> candidates,
        LetterboxInfo info,
        int origWidth,
        int origHeight)
    {
        if (dims.Length != 3) return;

        int dim1 = dims[1];
        int dim2 = dims[2];

        // Format 1: YOLOv5 / Anchor-Based [1, NumBoxes, 85] (dim2 >= 80)
        if (dim2 >= 80)
        {
            int numPredictions = dim1;
            int personClassIdx = 5 + _options.PersonClassId;

            for (int i = 0; i < numPredictions; i++)
            {
                float objectness = getValue(0, i, 4);
                if (objectness < _options.ConfidenceThreshold) continue;

                float classProb = getValue(0, i, personClassIdx);
                float confidence = objectness * classProb;

                if (confidence >= _options.ConfidenceThreshold)
                {
                    float cx = getValue(0, i, 0);
                    float cy = getValue(0, i, 1);
                    float w = getValue(0, i, 2);
                    float h = getValue(0, i, 3);

                    // Un-letterbox back to original image space
                    float xOrig = (cx - (w / 2.0f) - info.PadX) / info.Scale;
                    float yOrig = (cy - (h / 2.0f) - info.PadY) / info.Scale;
                    float wOrig = w / info.Scale;
                    float hOrig = h / info.Scale;

                    candidates.Add(new CandidateBox(new Rect2d(xOrig, yOrig, wOrig, hOrig), confidence));
                }
            }
        }
        // Format 2: YOLOv8 / Anchor-Free [1, 84, NumBoxes] (dim1 <= 90 and dim2 > 500)
        else if (dim1 <= 90 && dim2 > 500)
        {
            int numPredictions = dim2;
            int personClassRow = 4 + _options.PersonClassId;

            for (int i = 0; i < numPredictions; i++)
            {
                float personScore = getValue(0, personClassRow, i);

                if (personScore >= _options.ConfidenceThreshold)
                {
                    float cx = getValue(0, 0, i);
                    float cy = getValue(0, 1, i);
                    float w = getValue(0, 2, i);
                    float h = getValue(0, 3, i);

                    // Un-letterbox back to original image space
                    float xOrig = (cx - (w / 2.0f) - info.PadX) / info.Scale;
                    float yOrig = (cy - (h / 2.0f) - info.PadY) / info.Scale;
                    float wOrig = w / info.Scale;
                    float hOrig = h / info.Scale;

                    candidates.Add(new CandidateBox(new Rect2d(xOrig, yOrig, wOrig, hOrig), personScore));
                }
            }
        }
    }

    private List<DetectionResult> ApplyNonMaximumSuppression(List<CandidateBox> candidates, int origWidth, int origHeight)
    {
        var finalDetections = new List<DetectionResult>();
        if (candidates.Count == 0) return finalDetections;

        var boxes = new List<Rect2d>(candidates.Count);
        var confidences = new List<float>(candidates.Count);

        foreach (var c in candidates)
        {
            boxes.Add(c.Rect);
            confidences.Add(c.Confidence);
        }

        // OpenCV NMS calculation
        CvDnn.NMSBoxes(boxes, confidences, _options.ConfidenceThreshold, _options.NmsThreshold, out var indices);

        foreach (var idx in indices)
        {
            var r = boxes[idx];
            var conf = confidences[idx];

            var box = new BoundingBox
            {
                X = (float)r.X,
                Y = (float)r.Y,
                Width = (float)r.Width,
                Height = (float)r.Height
            }.Clamp(origWidth, origHeight);

            finalDetections.Add(new DetectionResult
            {
                Box = box,
                Confidence = conf,
                ClassId = _options.PersonClassId,
                Label = "person"
            });
        }

        return finalDetections;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _session?.Dispose();
        _disposed = true;
        GC.SuppressFinalize(this);
    }
}
