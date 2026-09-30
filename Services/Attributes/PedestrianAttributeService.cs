using Microsoft.Extensions.Options;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using VisionAttributeAI.DTOs;
using VisionAttributeAI.Models.AI;

namespace VisionAttributeAI.Services.Attributes;

/// <summary>
/// Pretrained PULC Pedestrian Attribute Recognition (PAR) Service using Microsoft.ML.OnnxRuntime.
/// Processes individual cropped person images and classifies 26 PULC pedestrian visual attributes.
/// </summary>
public class PedestrianAttributeService : IPedestrianAttributeService
{
    private readonly ILogger<PedestrianAttributeService> _logger;
    private readonly ParModelOptions _options;
    private InferenceSession? _session;
    private InferenceSession? _clipSession;
    private string _clipInputName = "pixel_values";
    private string _clipOutputName = "image_embeds";
    private bool _disposed;

    // Normalization constants for PULC / ImageNet-based backbones
    private static readonly float[] Mean = [0.485f, 0.456f, 0.406f];
    private static readonly float[] Std = [0.229f, 0.224f, 0.225f];

    // Normalization constants for OpenAI CLIP / Fashion-CLIP ViT backbones
    private static readonly float[] ClipMean = [0.48145466f, 0.4578275f, 0.40821073f];
    private static readonly float[] ClipStd = [0.26862954f, 0.26130258f, 0.27577711f];

    // PULC Person Attribute 26 heads in exact model export order
    public static readonly string[] Pa100kAttributeNames =
    [
        "Hat",                // 0
        "Glasses",            // 1
        "ShortSleeve",        // 2
        "LongSleeve",         // 3
        "UpperStripe",        // 4
        "UpperLogo",          // 5
        "UpperPlaid",         // 6
        "UpperSplice",        // 7
        "LowerStripe",        // 8
        "LowerPattern",       // 9
        "LongCoat",           // 10
        "Trousers",           // 11
        "Shorts",             // 12
        "Skirt&Dress",        // 13
        "Boots",              // 14
        "HandBag",            // 15
        "ShoulderBag",        // 16
        "Backpack",           // 17
        "HoldObjectsInFront", // 18
        "AgeLess18",          // 19 (PA-100K / PaddleClas PULC export index 19)
        "Age18-60",           // 20 (PA-100K / PaddleClas PULC export index 20)
        "AgeOver60",          // 21 (PA-100K / PaddleClas PULC export index 21)
        "Female",             // 22
        "Front",              // 23
        "Side",               // 24
        "Back"                // 25
    ];

    public bool IsModelLoaded => _session != null || _clipSession != null;
    public string InputName { get; private set; } = "x";
    public string OutputName { get; private set; } = "sigmoid_2.tmp_0";
    public int[] InputDimensions { get; private set; } = [1, 3, 256, 192];
    public Type InputElementType { get; private set; } = typeof(float);
    public Type OutputElementType { get; private set; } = typeof(float);

    public PedestrianAttributeService(
        IOptions<ParModelOptions> options,
        ILogger<PedestrianAttributeService> logger)
    {
        _logger = logger;
        _options = options.Value;

        InitializeModel();
    }

    private void InitializeModel()
    {
        // 1. Initialize Primary PULC Model
        try
        {
            var modelPath = _options.ModelPath;
            if (string.IsNullOrWhiteSpace(modelPath) || !File.Exists(modelPath))
            {
                _logger.LogWarning("PAR ONNX model not found at path '{ModelPath}'. Pedestrian attribute recognition fallback mode.", modelPath);
            }
            else
            {
                var sessionOptions = new Microsoft.ML.OnnxRuntime.SessionOptions
                {
                    GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                    ExecutionMode = ExecutionMode.ORT_SEQUENTIAL
                };

                _session = new InferenceSession(modelPath, sessionOptions);
                InspectModelMetadata();

                _logger.LogInformation(
                    "Successfully loaded PAR ONNX model '{ModelPath}'. Input='{Input}' ({InType}, [{InDim}]), Output='{Output}' ({OutType})",
                    modelPath,
                    InputName,
                    InputElementType,
                    string.Join(",", InputDimensions),
                    OutputName,
                    OutputElementType);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to initialize PAR ONNX model session from '{ModelPath}'.", _options.ModelPath);
        }

        // 2. Initialize Fashion-CLIP Vision Model for Zero-Shot Classification (Appearance Sex, etc.)
        try
        {
            var clipPath = _options.ClipModelPath;
            if (string.IsNullOrWhiteSpace(clipPath) || !File.Exists(clipPath))
            {
                _logger.LogWarning("Fashion-CLIP ONNX model not found at path '{ClipModelPath}'. Zero-shot appearance sex recognition disabled.", clipPath);
            }
            else
            {
                var clipSessionOptions = new Microsoft.ML.OnnxRuntime.SessionOptions
                {
                    GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                    ExecutionMode = ExecutionMode.ORT_SEQUENTIAL
                };

                _clipSession = new InferenceSession(clipPath, clipSessionOptions);

                if (_clipSession.InputMetadata.Count > 0)
                {
                    _clipInputName = _clipSession.InputMetadata.First().Key;
                }
                if (_clipSession.OutputMetadata.Count > 0)
                {
                    _clipOutputName = _clipSession.OutputMetadata.ContainsKey("image_embeds")
                        ? "image_embeds"
                        : _clipSession.OutputMetadata.First().Key;
                }

                _logger.LogInformation(
                    "Successfully loaded Fashion-CLIP Vision model '{ClipPath}'. Input='{ClipIn}', Output='{ClipOut}'",
                    clipPath, _clipInputName, _clipOutputName);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to initialize Fashion-CLIP Vision model session from '{ClipPath}'.", _options.ClipModelPath);
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

            _logger.LogInformation("Discovered PAR Input: Name='{Name}', Type={Type}, Dimensions=[{Dimensions}]",
                InputName, InputElementType.Name, string.Join(",", InputDimensions));
        }

        // 2. Inspect Output Metadata
        var outputMeta = _session.OutputMetadata;
        if (outputMeta.Count > 0)
        {
            var firstOutput = outputMeta.First();
            OutputName = firstOutput.Key;
            OutputElementType = firstOutput.Value.ElementType;

            _logger.LogInformation("Discovered PAR Output: Name='{Name}', Type={Type}, Dimensions=[{Dimensions}]",
                OutputName, OutputElementType.Name, string.Join(",", firstOutput.Value.Dimensions));
        }
    }

    public Task<PersonAttributes> RecognizeAttributesAsync(byte[] imageBytes, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(imageBytes);
        if (imageBytes.Length == 0)
        {
            throw new ArgumentException("Provided image bytes cannot be empty.", nameof(imageBytes));
        }

        using var mat = Cv2.ImDecode(imageBytes, ImreadModes.Color);
        if (mat.Empty())
        {
            throw new InvalidOperationException("Failed to decode image bytes into a valid OpenCV Mat.");
        }

        return RecognizeAttributesAsync(mat, cancellationToken);
    }

    public Task<PersonAttributes> RecognizeAttributesAsync(Mat personCrop, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(personCrop);

        if (personCrop.Empty() || personCrop.Width <= 0 || personCrop.Height <= 0)
        {
            throw new ArgumentException("Person crop image is empty or invalid.", nameof(personCrop));
        }

        if (_session == null && _clipSession == null)
        {
            throw new InvalidOperationException(
                $"No attribute recognition models are loaded. Ensure '{_options.ModelPath}' or '{_options.ClipModelPath}' exists.");
        }

        cancellationToken.ThrowIfCancellationRequested();

        PersonAttributes attributes;

        // 1. Run PULC Model for general pedestrian attributes (if loaded)
        if (_session != null)
        {
            int targetHeight = _options.InputHeight > 0 ? _options.InputHeight : 256;
            int targetWidth = _options.InputWidth > 0 ? _options.InputWidth : 192;

            if (InputDimensions.Length >= 4)
            {
                if (InputDimensions[2] > 0) targetHeight = InputDimensions[2];
                if (InputDimensions[3] > 0) targetWidth = InputDimensions[3];
            }

            var tensor = Preprocess(personCrop, targetWidth, targetHeight);
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(InputName, tensor)
            };

            using var results = _session.Run(inputs);
            var outputTensor = results.First(r => r.Name == OutputName);

            var (rawOutputs, probabilities) = ParseOutputProbabilities(outputTensor);
            attributes = MapProbabilitiesToDto(rawOutputs, probabilities, _options.Thresholds);
        }
        else
        {
            attributes = new PersonAttributes();
        }

        // 2. High-Accuracy Zero-Shot Appearance Sex Classification via Fashion-CLIP
        if (_clipSession != null && _options.UseClipForAppearanceSex)
        {
            var (clipSex, clipConfidence, femaleProb, maleProb) = ClassifyAppearanceSexWithClip(personCrop);
            attributes.AppearanceSex = clipSex;
            attributes.AppearanceSexConfidence = clipConfidence;
            attributes.RawAttributes["Female"] = femaleProb;
            attributes.RawAttributes["Male"] = maleProb;

            // Update diagnostics table for Appearance Sex
            var femaleDiag = attributes.Diagnostics.FirstOrDefault(d => d.Label == "Female" || d.Index == 22);
            float sexThreshold = _options.Thresholds.AppearanceSex > 0 ? _options.Thresholds.AppearanceSex : 0.60f;

            if (femaleDiag != null)
            {
                femaleDiag.PostActivation = femaleProb;
                femaleDiag.Threshold = sexThreshold;
                femaleDiag.IsActive = (clipSex == "Female");
                femaleDiag.Result = clipSex == "Female"
                    ? $"Female (CLIP: {femaleProb:P1})"
                    : (clipSex == "Male" ? $"Male (CLIP: {maleProb:P1})" : "Unknown (CLIP: Ambiguous)");
            }
            else
            {
                attributes.Diagnostics.Add(new AttributeDiagnosticItem
                {
                    Index = 22,
                    Label = "Female",
                    RawOutput = femaleProb,
                    PostActivation = femaleProb,
                    Threshold = sexThreshold,
                    Result = clipSex == "Female"
                        ? $"Female (CLIP: {femaleProb:P1})"
                        : (clipSex == "Male" ? $"Male (CLIP: {maleProb:P1})" : "Unknown (CLIP: Ambiguous)"),
                    IsActive = (clipSex == "Female")
                });
            }
        }

        return Task.FromResult(attributes);
    }

    private (string Sex, float Confidence, float FemaleProb, float MaleProb) ClassifyAppearanceSexWithClip(Mat personCrop)
    {
        if (_clipSession == null) return ("Unknown", 0.0f, 0.5f, 0.5f);

        var clipTensor = PreprocessClip(personCrop);
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_clipInputName, clipTensor)
        };

        using var results = _clipSession.Run(inputs);
        var outputTensor = results.First(r => r.Name == _clipOutputName);
        var rawEmbeddings = outputTensor.AsEnumerable<float>().ToArray();

        // L2 normalize 512-dim visual embedding
        float norm = 0f;
        for (int i = 0; i < rawEmbeddings.Length; i++)
        {
            norm += rawEmbeddings[i] * rawEmbeddings[i];
        }
        norm = MathF.Sqrt(norm);
        if (norm > 1e-6f)
        {
            for (int i = 0; i < rawEmbeddings.Length; i++)
            {
                rawEmbeddings[i] /= norm;
            }
        }

        // Cosine similarity against precomputed L2-normalized text embeddings
        float manDot = 0f;
        float womanDot = 0f;
        for (int i = 0; i < rawEmbeddings.Length && i < ClipTextEmbeddings.ManEmbedding.Length; i++)
        {
            manDot += rawEmbeddings[i] * ClipTextEmbeddings.ManEmbedding[i];
            womanDot += rawEmbeddings[i] * ClipTextEmbeddings.WomanEmbedding[i];
        }

        // Temperature scaling (CLIP standard logit scale = 100.0)
        float logitMan = manDot * 100.0f;
        float logitWoman = womanDot * 100.0f;

        float maxLogit = Math.Max(logitMan, logitWoman);
        float expMan = MathF.Exp(logitMan - maxLogit);
        float expWoman = MathF.Exp(logitWoman - maxLogit);
        float sumExp = expMan + expWoman;

        float probMan = expMan / sumExp;
        float probWoman = expWoman / sumExp;

        float threshold = _options.Thresholds.AppearanceSex > 0 ? _options.Thresholds.AppearanceSex : 0.60f;

        if (probWoman >= threshold)
        {
            return ("Female", probWoman, probWoman, probMan);
        }
        else if (probMan >= threshold)
        {
            return ("Male", probMan, probWoman, probMan);
        }
        else
        {
            return ("Unknown", Math.Max(probWoman, probMan), probWoman, probMan);
        }
    }

    private static DenseTensor<float> PreprocessClip(Mat src)
    {
        const int clipWidth = 224;
        const int clipHeight = 224;

        using var resized = new Mat();
        Cv2.Resize(src, resized, new Size(clipWidth, clipHeight), interpolation: InterpolationFlags.Cubic);

        using var rgb = new Mat();
        Cv2.CvtColor(resized, rgb, ColorConversionCodes.BGR2RGB);

        var tensor = new DenseTensor<float>([1, 3, clipHeight, clipWidth]);
        unsafe
        {
            byte* ptr = rgb.DataPointer;
            int step = (int)rgb.Step();

            for (int y = 0; y < clipHeight; y++)
            {
                byte* rowPtr = ptr + (y * step);
                for (int x = 0; x < clipWidth; x++)
                {
                    int pixelIndex = x * 3;
                    float r = rowPtr[pixelIndex] / 255.0f;
                    float g = rowPtr[pixelIndex + 1] / 255.0f;
                    float b = rowPtr[pixelIndex + 2] / 255.0f;

                    tensor[0, 0, y, x] = (r - ClipMean[0]) / ClipStd[0];
                    tensor[0, 1, y, x] = (g - ClipMean[1]) / ClipStd[1];
                    tensor[0, 2, y, x] = (b - ClipMean[2]) / ClipStd[2];
                }
            }
        }

        return tensor;
    }

    private static DenseTensor<float> Preprocess(Mat src, int targetWidth, int targetHeight)
    {
        // Resize to target dimensions (Linear interpolation)
        using var resized = new Mat();
        Cv2.Resize(src, resized, new Size(targetWidth, targetHeight), interpolation: InterpolationFlags.Linear);

        // Convert OpenCV BGR to RGB
        using var rgb = new Mat();
        Cv2.CvtColor(resized, rgb, ColorConversionCodes.BGR2RGB);

        // Allocate NCHW tensor [1, 3, targetHeight, targetWidth]
        var tensor = new DenseTensor<float>([1, 3, targetHeight, targetWidth]);

        // Normalize: (pixel/255.0 - mean) / std
        unsafe
        {
            byte* ptr = rgb.DataPointer;
            int step = (int)rgb.Step();

            for (int y = 0; y < targetHeight; y++)
            {
                byte* rowPtr = ptr + (y * step);
                for (int x = 0; x < targetWidth; x++)
                {
                    int pixelIndex = x * 3;
                    float r = rowPtr[pixelIndex] / 255.0f;
                    float g = rowPtr[pixelIndex + 1] / 255.0f;
                    float b = rowPtr[pixelIndex + 2] / 255.0f;

                    tensor[0, 0, y, x] = (r - Mean[0]) / Std[0];
                    tensor[0, 1, y, x] = (g - Mean[1]) / Std[1];
                    tensor[0, 2, y, x] = (b - Mean[2]) / Std[2];
                }
            }
        }

        return tensor;
    }

    private static (float[] Raw, float[] Probabilities) ParseOutputProbabilities(DisposableNamedOnnxValue outputTensor)
    {
        var rawData = outputTensor.AsEnumerable<float>().ToArray();

        // If the model output is already sigmoid-activated (values in [0, 1]), don't apply double-sigmoid.
        bool requiresSigmoid = rawData.Any(v => v < -0.01f || v > 1.01f);

        var probabilities = new float[rawData.Length];
        for (int i = 0; i < rawData.Length; i++)
        {
            probabilities[i] = requiresSigmoid ? Sigmoid(rawData[i]) : Math.Clamp(rawData[i], 0.0f, 1.0f);
        }

        return (rawData, probabilities);
    }

    private static float Sigmoid(float x)
    {
        return 1.0f / (1.0f + MathF.Exp(-x));
    }

    private static PersonAttributes MapProbabilitiesToDto(float[] rawData, float[] probs, ParThresholdOptions thresholds)
    {
        var dto = new PersonAttributes();

        // Populate raw attributes map for complete debugging visibility (using exact 26 PULC index mapping)
        for (int i = 0; i < probs.Length && i < Pa100kAttributeNames.Length; i++)
        {
            dto.RawAttributes[Pa100kAttributeNames[i]] = probs[i];
        }

        float GetRaw(int index) => (index >= 0 && index < rawData.Length) ? rawData[index] : 0.0f;
        float GetProb(int index) => (index >= 0 && index < probs.Length) ? probs[index] : 0.0f;

        // 1. Hat and Glasses (Indices 0: Hat, 1: Glasses)
        float hatThreshold = thresholds.Hat > 0 ? thresholds.Hat : 0.50f;
        dto.HasHat = GetProb(0) >= hatThreshold;
        dto.HatConfidence = GetProb(0);

        float glassesThreshold = thresholds.Glasses > 0 ? thresholds.Glasses : 0.40f;
        dto.HasGlasses = GetProb(1) >= glassesThreshold;
        dto.GlassesConfidence = GetProb(1);

        // 2. Sleeve Type (Indices 2: ShortSleeve, 3: LongSleeve)
        float shortSleeve = GetProb(2);
        float longSleeve = GetProb(3);
        float maxSleeve = Math.Max(shortSleeve, longSleeve);
        float sleeveThreshold = thresholds.SleeveType > 0 ? thresholds.SleeveType : 0.50f;

        if (maxSleeve < sleeveThreshold)
        {
            dto.SleeveType = "Unknown";
            dto.SleeveTypeConfidence = maxSleeve;
        }
        else if (shortSleeve > longSleeve)
        {
            dto.SleeveType = "ShortSleeve";
            dto.SleeveTypeConfidence = shortSleeve;
        }
        else
        {
            dto.SleeveType = "LongSleeve";
            dto.SleeveTypeConfidence = longSleeve;
        }

        // 3. Clothing Patterns (Indices: 4: UpperStripe, 5: UpperLogo, 6: UpperPlaid, 7: UpperSplice, 8: LowerStripe, 9: LowerPattern)
        float patternThreshold = thresholds.ClothingPatterns > 0 ? thresholds.ClothingPatterns : 0.50f;
        int[] patternIndices = [4, 5, 6, 7, 8, 9];
        foreach (int pIdx in patternIndices)
        {
            if (GetProb(pIdx) >= patternThreshold)
            {
                dto.ClothingPatterns.Add(Pa100kAttributeNames[pIdx]);
            }
        }

        // 4. Lower Body Attire (Indices: 10: LongCoat, 11: Trousers, 12: Shorts, 13: Skirt&Dress)
        float lowerThreshold = thresholds.LowerAttire > 0 ? thresholds.LowerAttire : 0.50f;
        dto.LongCoat = GetProb(10) >= lowerThreshold;
        dto.LongCoatConfidence = GetProb(10);

        dto.Trousers = GetProb(11) >= lowerThreshold;
        dto.TrousersConfidence = GetProb(11);

        dto.Shorts = GetProb(12) >= lowerThreshold;
        dto.ShortsConfidence = GetProb(12);

        dto.SkirtOrDress = GetProb(13) >= lowerThreshold;
        dto.SkirtOrDressConfidence = GetProb(13);

        // 5. Footwear (Index 14: Boots)
        float bootsThreshold = thresholds.Boots > 0 ? thresholds.Boots : 0.50f;
        dto.HasBoots = GetProb(14) >= bootsThreshold;
        dto.BootsConfidence = GetProb(14);

        // 6. Accessories & Bags (Indices: 15: HandBag, 16: ShoulderBag, 17: Backpack, 18: HoldObjectsInFront)
        float bagThreshold = thresholds.Bags > 0 ? thresholds.Bags : 0.50f;
        dto.HasHandBag = GetProb(15) >= bagThreshold;
        dto.HandBagConfidence = GetProb(15);

        dto.HasShoulderBag = GetProb(16) >= bagThreshold;
        dto.ShoulderBagConfidence = GetProb(16);

        dto.HasBackpack = GetProb(17) >= bagThreshold;
        dto.BackpackConfidence = GetProb(17);

        dto.HoldObjectsInFront = GetProb(18) >= bagThreshold;
        dto.HoldObjectsInFrontConfidence = GetProb(18);

        // 7. Age Group (Indices: 19: AgeLess18, 20: Age18-60, 21: AgeOver60)
        float ageLess18 = GetProb(19);
        float age18To60 = GetProb(20);
        float ageOver60 = GetProb(21);
        float maxAgeProb = Math.Max(ageLess18, Math.Max(age18To60, ageOver60));
        float ageThreshold = thresholds.AgeGroup > 0 ? thresholds.AgeGroup : 0.50f;

        if (maxAgeProb < ageThreshold)
        {
            dto.AgeGroup = "Unknown";
            dto.AgeGroupConfidence = maxAgeProb;
        }
        else if (maxAgeProb == ageLess18)
        {
            dto.AgeGroup = "Age < 18";
            dto.AgeGroupConfidence = ageLess18;
        }
        else if (maxAgeProb == ageOver60)
        {
            dto.AgeGroup = "Age > 60";
            dto.AgeGroupConfidence = ageOver60;
        }
        else
        {
            dto.AgeGroup = "Age 18-60";
            dto.AgeGroupConfidence = age18To60;
        }

        // 8. Appearance Sex (Index 22: Female)
        // Explicitly framed as visual appearance estimation rather than verified identity
        float femaleProb = GetProb(22);
        float maleProb = 1.0f - femaleProb;
        float sexThreshold = thresholds.AppearanceSex > 0 ? thresholds.AppearanceSex : 0.60f;

        if (femaleProb >= sexThreshold)
        {
            dto.AppearanceSex = "Female";
            dto.AppearanceSexConfidence = femaleProb;
        }
        else if (maleProb >= sexThreshold)
        {
            dto.AppearanceSex = "Male";
            dto.AppearanceSexConfidence = maleProb;
        }
        else
        {
            dto.AppearanceSex = "Unknown";
            dto.AppearanceSexConfidence = Math.Max(femaleProb, maleProb);
        }

        // 9. Orientation (Indices: 23: Front, 24: Side, 25: Back)
        float front = GetProb(23);
        float side = GetProb(24);
        float back = GetProb(25);
        float maxOrientation = Math.Max(front, Math.Max(side, back));
        float orientationThreshold = thresholds.Orientation > 0 ? thresholds.Orientation : 0.50f;

        if (maxOrientation < orientationThreshold)
        {
            dto.Orientation = "Unknown";
            dto.OrientationConfidence = maxOrientation;
        }
        else if (maxOrientation == front)
        {
            dto.Orientation = "Front";
            dto.OrientationConfidence = front;
        }
        else if (maxOrientation == side)
        {
            dto.Orientation = "Side";
            dto.OrientationConfidence = side;
        }
        else
        {
            dto.Orientation = "Back";
            dto.OrientationConfidence = back;
        }

        // 10. Generate full 26-row Diagnostic Breakdown
        for (int i = 0; i < Pa100kAttributeNames.Length; i++)
        {
            string label = Pa100kAttributeNames[i];
            float raw = GetRaw(i);
            float post = GetProb(i);
            float threshold = 0.50f;
            string result;
            bool isActive = false;

            switch (i)
            {
                case 0: // Hat
                    threshold = hatThreshold;
                    isActive = post >= threshold;
                    result = isActive ? "Yes (Detected)" : "No";
                    break;
                case 1: // Glasses
                    threshold = glassesThreshold;
                    isActive = post >= threshold;
                    result = isActive ? "Yes (Detected)" : "No";
                    break;
                case 2: // ShortSleeve
                case 3: // LongSleeve
                    threshold = sleeveThreshold;
                    if (dto.SleeveType == "Unknown")
                    {
                        result = "Unknown (Low Confidence)";
                    }
                    else if (dto.SleeveType == label)
                    {
                        isActive = true;
                        result = $"Selected ({dto.SleeveType})";
                    }
                    else
                    {
                        result = "Not Selected";
                    }
                    break;
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9: // Patterns
                    threshold = patternThreshold;
                    isActive = post >= threshold;
                    result = isActive ? "Yes (Detected)" : "No";
                    break;
                case 10:
                case 11:
                case 12:
                case 13: // Lower Attire
                    threshold = lowerThreshold;
                    isActive = post >= threshold;
                    result = isActive ? "Yes (Detected)" : "No";
                    break;
                case 14: // Boots
                    threshold = bootsThreshold;
                    isActive = post >= threshold;
                    result = isActive ? "Yes (Detected)" : "No";
                    break;
                case 15:
                case 16:
                case 17:
                case 18: // Bags & Hand-held
                    threshold = bagThreshold;
                    isActive = post >= threshold;
                    result = isActive ? "Yes (Detected)" : "No";
                    break;
                case 19: // AgeLess18
                case 20: // Age18-60
                case 21: // AgeOver60
                    threshold = ageThreshold;
                    if (dto.AgeGroup == "Unknown")
                    {
                        result = "Unknown (Low Confidence)";
                    }
                    else if ((i == 19 && dto.AgeGroup == "Age < 18") ||
                             (i == 20 && dto.AgeGroup == "Age 18-60") ||
                             (i == 21 && dto.AgeGroup == "Age > 60"))
                    {
                        isActive = true;
                        result = $"Selected ({dto.AgeGroup})";
                    }
                    else
                    {
                        result = "Not Selected";
                    }
                    break;
                case 22: // Female
                    threshold = sexThreshold;
                    if (post >= sexThreshold)
                    {
                        isActive = true;
                        result = $"Female (>= {sexThreshold:F2})";
                    }
                    else if ((1.0f - post) >= sexThreshold)
                    {
                        isActive = false;
                        result = $"Male (Inferred: P(Male)={(1.0f - post):F2})";
                    }
                    else
                    {
                        result = "Unknown (Ambiguous)";
                    }
                    break;
                case 23: // Front
                case 24: // Side
                case 25: // Back
                    threshold = orientationThreshold;
                    if (dto.Orientation == "Unknown")
                    {
                        result = "Unknown (Low Confidence)";
                    }
                    else if (dto.Orientation == label)
                    {
                        isActive = true;
                        result = $"Selected ({dto.Orientation})";
                    }
                    else
                    {
                        result = "Not Selected";
                    }
                    break;
                default:
                    result = post >= threshold ? "Yes" : "No";
                    break;
            }

            dto.Diagnostics.Add(new AttributeDiagnosticItem
            {
                Index = i,
                Label = label,
                RawOutput = raw,
                PostActivation = post,
                Threshold = threshold,
                Result = result,
                IsActive = isActive
            });
        }

        return dto;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _session?.Dispose();
        _clipSession?.Dispose();
        _disposed = true;
        GC.SuppressFinalize(this);
    }
}
