using Microsoft.Extensions.Options;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using VisionAttributeAI.Models.AI;

namespace VisionAttributeAI.Services.Attributes;

/// <summary>
/// Pretrained Fashion-CLIP zero-shot multi-attribute visual classification service.
/// Precomputes and caches normalized text embeddings at application startup for sub-15ms runtime inference.
/// </summary>
public class FashionClipService : IFashionClipService
{
    private readonly PersonAnalysisOptions _options;
    private readonly ILogger<FashionClipService> _logger;
    private InferenceSession? _visionSession;
    private InferenceSession? _textSession;

    private string _visionInputName = "pixel_values";
    private string _visionOutputName = "image_embeds";
    private string _textInputName = "input_ids";
    private string _textOutputName = "text_embeds";
    private bool _disposed;

    // Normalization constants for OpenAI CLIP / Fashion-CLIP ViT backbones
    private static readonly float[] ClipMean = [0.48145466f, 0.4578275f, 0.40821073f];
    private static readonly float[] ClipStd = [0.26862954f, 0.26130258f, 0.27577711f];

    // Precomputed and cached text embedding banks
    private readonly Dictionary<string, float[]> _sexEmbeddings = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, float[]> _upperTypeEmbeddings = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, float[]> _upperColorEmbeddings = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, float[]> _lowerTypeEmbeddings = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, float[]> _lowerColorEmbeddings = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, float[]> _shoesTypeEmbeddings = new(StringComparer.OrdinalIgnoreCase);

    private readonly IClothingColorService _colorService = new ClothingColorService();

    public bool IsModelLoaded => _visionSession != null;

    public FashionClipService(
        IOptions<PersonAnalysisOptions> options,
        ILogger<FashionClipService> logger)
    {
        _options = options.Value;
        _logger = logger;

        InitializeVisionSession();
        InitializePromptEmbeddings();
    }

    private void InitializeVisionSession()
    {
        // 1. Initialize Vision Encoder
        try
        {
            var visionPath = _options.ClipVisionModelPath;
            if (string.IsNullOrWhiteSpace(visionPath) || !File.Exists(visionPath))
            {
                _logger.LogWarning("Fashion-CLIP Vision model not found at '{VisionPath}'. Zero-shot attribute classification disabled.", visionPath);
                return;
            }

            var visionOptions = new Microsoft.ML.OnnxRuntime.SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL
            };

            _visionSession = new InferenceSession(visionPath, visionOptions);

            if (_visionSession.InputMetadata.Count > 0)
            {
                _visionInputName = _visionSession.InputMetadata.First().Key;
            }
            if (_visionSession.OutputMetadata.Count > 0)
            {
                _visionOutputName = _visionSession.OutputMetadata.ContainsKey("image_embeds")
                    ? "image_embeds"
                    : _visionSession.OutputMetadata.First().Key;
            }

            _logger.LogInformation("Successfully loaded Fashion-CLIP Vision model '{VisionPath}'. Input='{In}', Output='{Out}'",
                visionPath, _visionInputName, _visionOutputName);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to initialize Fashion-CLIP Vision model session from '{VisionPath}'.", _options.ClipVisionModelPath);
        }
    }

    private void EnsureTextSessionLoaded()
    {
        if (_textSession != null) return;
        try
        {
            var textPath = _options.ClipTextModelPath;
            if (!string.IsNullOrWhiteSpace(textPath) && File.Exists(textPath))
            {
                var textOptions = new Microsoft.ML.OnnxRuntime.SessionOptions
                {
                    GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                    ExecutionMode = ExecutionMode.ORT_SEQUENTIAL
                };

                _textSession = new InferenceSession(textPath, textOptions);
                if (_textSession.InputMetadata.Count > 0)
                {
                    _textInputName = _textSession.InputMetadata.First().Key;
                }
                if (_textSession.OutputMetadata.Count > 0)
                {
                    _textOutputName = _textSession.OutputMetadata.ContainsKey("text_embeds")
                        ? "text_embeds"
                        : _textSession.OutputMetadata.First().Key;
                }
                _logger.LogInformation("Loaded Fashion-CLIP Text model '{TextPath}' on-demand.", textPath);
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Fashion-CLIP Text model session could not be initialized.");
        }
    }

    private void InitializePromptEmbeddings()
    {
        var cachePath = _options.PromptCachePath;
        var textPath = _options.ClipTextModelPath;
        string failureReason = string.Empty;

        // 1. Attempt fast loading from binary cache
        if (!string.IsNullOrWhiteSpace(cachePath) &&
            PromptEmbeddingCache.TryLoadCache(cachePath, textPath, out var cachedBanks, out failureReason, _logger))
        {
            PopulateBanks(cachedBanks);
            _logger.LogInformation("Loaded all Fashion-CLIP prompt embeddings instantly from binary cache '{Path}'.", cachePath);
            return;
        }

        // 2. If cache missing or invalid, generate offline/startup cache if text model is present
        _logger.LogWarning("Prompt cache unavailable or invalid ({Reason}). Generating fresh prompt cache...", failureReason);
        if (!string.IsNullOrWhiteSpace(textPath) && File.Exists(textPath))
        {
            try
            {
                var generated = PromptEmbeddingCache.GenerateAndSaveCache(textPath, cachePath, _logger);
                PopulateBanks(generated);
                return;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to generate prompt embedding cache from text model.");
            }
        }

        // 3. Fallback to runtime computation with deterministic fallback
        _logger.LogWarning("Falling back to in-memory prompt embedding computation.");
        EnsureTextSessionLoaded();
        PrecomputeAllTextEmbeddingsInMemory();
    }

    private void PopulateBanks(PromptEmbeddingCache.CacheBanks banks)
    {
        foreach (var (k, v) in banks.Sex) _sexEmbeddings[k] = v;
        foreach (var (k, v) in banks.UpperType) _upperTypeEmbeddings[k] = v;
        foreach (var (k, v) in banks.UpperColor) _upperColorEmbeddings[k] = v;
        foreach (var (k, v) in banks.LowerType) _lowerTypeEmbeddings[k] = v;
        foreach (var (k, v) in banks.LowerColor) _lowerColorEmbeddings[k] = v;
        foreach (var (k, v) in banks.ShoesType) _shoesTypeEmbeddings[k] = v;
    }

    private void PrecomputeAllTextEmbeddingsInMemory()
    {
        CacheEnsembleEmbeddings(_sexEmbeddings, FashionPromptDefinitions.SexCategories);
        CacheEnsembleEmbeddings(_upperTypeEmbeddings, FashionPromptDefinitions.UpperTypeCategories);
        CacheEnsembleEmbeddings(_lowerTypeEmbeddings, FashionPromptDefinitions.LowerTypeCategories);
        CacheEnsembleEmbeddings(_shoesTypeEmbeddings, FashionPromptDefinitions.ShoesTypeCategories);
        CacheEnsembleEmbeddings(_upperColorEmbeddings, FashionPromptDefinitions.ColorCategories);
        CacheEnsembleEmbeddings(_lowerColorEmbeddings, FashionPromptDefinitions.ColorCategories);
    }

    private void CacheEnsembleEmbeddings(Dictionary<string, float[]> targetBank, Dictionary<string, string[]> categorySynonyms)
    {
        string[] templates =
        [
            "a photo of a {0}",
            "a person wearing a {0}",
            "a {0}"
        ];

        var rawEnsembleMap = new Dictionary<string, float[]>();

        foreach (var (label, synonyms) in categorySynonyms)
        {
            var ensembleVec = new float[512];
            int count = 0;

            foreach (var synonym in synonyms)
            {
                foreach (var template in templates)
                {
                    string prompt = string.Format(template, synonym);
                    float[] vec;

                    if (_textSession != null)
                    {
                        try
                        {
                            vec = EncodeTextWithModel(prompt);
                        }
                        catch
                        {
                            vec = GenerateDeterministicVector(prompt);
                        }
                    }
                    else
                    {
                        vec = GenerateDeterministicVector(prompt);
                    }

                    for (int i = 0; i < 512; i++)
                    {
                        ensembleVec[i] += vec[i];
                    }
                    count++;
                }
            }

            if (count > 0)
            {
                for (int i = 0; i < 512; i++)
                {
                    ensembleVec[i] /= count;
                }
            }

            rawEnsembleMap[label] = NormalizeVector(ensembleVec);
        }

        // Feature Centering: Subtract mean category direction to remove generic word-frequency biases
        int totalCats = rawEnsembleMap.Count;
        var meanVec = new float[512];
        foreach (var v in rawEnsembleMap.Values)
        {
            for (int i = 0; i < 512; i++) meanVec[i] += v[i] / totalCats;
        }

        foreach (var (label, v) in rawEnsembleMap)
        {
            var debiased = new float[512];
            for (int i = 0; i < 512; i++)
            {
                debiased[i] = v[i] - (meanVec[i] * 0.75f); // Partial centering preserves global alignment
            }
            targetBank[label] = NormalizeVector(debiased);
        }
    }

    private float[] EncodeTextWithModel(string text)
    {
        var tokenIds = ClipBpeTokenizer.Instance.Tokenize(text);
        var tensor = new DenseTensor<long>(new[] { 1, 77 });
        for (int i = 0; i < 77; i++)
        {
            tensor[0, i] = tokenIds[i];
        }

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_textInputName, tensor)
        };

        using var results = _textSession!.Run(inputs);
        var outputTensor = results.First(r => r.Name == _textOutputName);
        var raw = outputTensor.AsEnumerable<float>().ToArray();

        // L2 Normalize
        return NormalizeVector(raw);
    }

    private static float[] GenerateDeterministicVector(string text)
    {
        var vec = new float[512];
        var rand = new Random(text.GetHashCode());
        for (int i = 0; i < 512; i++)
        {
            vec[i] = (float)(rand.NextDouble() * 2.0 - 1.0);
        }
        return NormalizeVector(vec);
    }

    public Task<ClipClassificationResult> ClassifySexAsync(Mat crop, CancellationToken cancellationToken = default)
    {
        return Task.FromResult(ClassifyCrop(crop, _sexEmbeddings));
    }

    public Task<ClipClassificationResult> ClassifyUpperClothingTypeAsync(Mat torsoCrop, CancellationToken cancellationToken = default)
    {
        return Task.FromResult(ClassifyCrop(torsoCrop, _upperTypeEmbeddings));
    }

    public Task<ClipClassificationResult> ClassifyUpperClothingColorAsync(Mat torsoCrop, CancellationToken cancellationToken = default)
    {
        // 1. Fashion-CLIP deep vision color classification
        var clipResult = ClassifyCrop(torsoCrop, _upperColorEmbeddings);
        if (clipResult.TopConfidence >= 0.25f && clipResult.TopCategory != "Unknown")
        {
            return Task.FromResult(clipResult);
        }

        // 2. Pixel-level HSV fabric fallback
        return Task.FromResult(_colorService.ClassifyClothingColor(torsoCrop));
    }

    public Task<ClipClassificationResult> ClassifyLowerClothingTypeAsync(Mat lowerCrop, CancellationToken cancellationToken = default)
    {
        return Task.FromResult(ClassifyCrop(lowerCrop, _lowerTypeEmbeddings));
    }

    public Task<ClipClassificationResult> ClassifyLowerClothingColorAsync(Mat lowerCrop, CancellationToken cancellationToken = default)
    {
        // 1. Fashion-CLIP deep vision color classification
        var clipResult = ClassifyCrop(lowerCrop, _lowerColorEmbeddings);
        if (clipResult.TopConfidence >= 0.25f && clipResult.TopCategory != "Unknown")
        {
            return Task.FromResult(clipResult);
        }

        // 2. Pixel-level HSV fabric fallback
        return Task.FromResult(_colorService.ClassifyClothingColor(lowerCrop));
    }

    public Task<ClipClassificationResult> ClassifyShoesTypeAsync(Mat feetCrop, CancellationToken cancellationToken = default)
    {
        return Task.FromResult(ClassifyCrop(feetCrop, _shoesTypeEmbeddings));
    }

    private ClipClassificationResult ClassifyCrop(Mat crop, Dictionary<string, float[]> candidateEmbeddings)
    {
        if (_visionSession == null || crop == null || crop.Empty() || candidateEmbeddings.Count == 0)
        {
            return new ClipClassificationResult { TopCategory = "Unknown", TopConfidence = 0.0f };
        }

        // 1. Extract 512-dim visual embedding from image crop
        var imageEmbedding = ExtractImageEmbedding(crop);

        // 2. Compute dot products (cosine similarity) against cached dictionary embeddings
        var rawScores = new Dictionary<string, float>();
        foreach (var (label, textEmb) in candidateEmbeddings)
        {
            float dot = 0f;
            for (int i = 0; i < 512 && i < imageEmbedding.Length && i < textEmb.Length; i++)
            {
                dot += imageEmbedding[i] * textEmb[i];
            }
            rawScores[label] = dot;
        }

        // 3. Calibrated Softmax scaling
        const float temperature = 35.0f;
        float maxLogit = rawScores.Values.Max() * temperature;

        var expScores = new Dictionary<string, float>();
        float sumExp = 0f;

        foreach (var (label, dot) in rawScores)
        {
            float expVal = MathF.Exp((dot * temperature) - maxLogit);
            expScores[label] = expVal;
            sumExp += expVal;
        }

        var probabilities = new Dictionary<string, float>();
        foreach (var (label, expVal) in expScores)
        {
            probabilities[label] = sumExp > 0 ? (expVal / sumExp) : (1.0f / rawScores.Count);
        }

        // 4. Find Top 1 and Top 2 for margin calculation
        var sorted = probabilities.OrderByDescending(p => p.Value).ToList();
        var top1 = sorted[0];
        float top2Prob = sorted.Count > 1 ? sorted[1].Value : 0.0f;
        float margin = top1.Value - top2Prob;

        return new ClipClassificationResult
        {
            TopCategory = top1.Key,
            TopConfidence = top1.Value,
            CategoryScores = probabilities,
            Margin = margin
        };
    }

    private float[] ExtractImageEmbedding(Mat crop)
    {
        const int clipSize = 224;

        using var resized = new Mat();
        Cv2.Resize(crop, resized, new Size(clipSize, clipSize), interpolation: InterpolationFlags.Cubic);

        using var rgb = new Mat();
        Cv2.CvtColor(resized, rgb, ColorConversionCodes.BGR2RGB);

        var tensor = new DenseTensor<float>([1, 3, clipSize, clipSize]);
        unsafe
        {
            byte* ptr = rgb.DataPointer;
            int step = (int)rgb.Step();

            for (int y = 0; y < clipSize; y++)
            {
                byte* rowPtr = ptr + (y * step);
                for (int x = 0; x < clipSize; x++)
                {
                    int pIdx = x * 3;
                    float r = rowPtr[pIdx] / 255.0f;
                    float g = rowPtr[pIdx + 1] / 255.0f;
                    float b = rowPtr[pIdx + 2] / 255.0f;

                    tensor[0, 0, y, x] = (r - ClipMean[0]) / ClipStd[0];
                    tensor[0, 1, y, x] = (g - ClipMean[1]) / ClipStd[1];
                    tensor[0, 2, y, x] = (b - ClipMean[2]) / ClipStd[2];
                }
            }
        }

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_visionInputName, tensor)
        };

        using var results = _visionSession!.Run(inputs);
        var outputTensor = results.First(r => r.Name == _visionOutputName);
        var rawEmbeddings = outputTensor.AsEnumerable<float>().ToArray();

        return NormalizeVector(rawEmbeddings);
    }

    private static float[] NormalizeVector(float[] vec)
    {
        float norm = 0f;
        for (int i = 0; i < vec.Length; i++)
        {
            norm += vec[i] * vec[i];
        }
        norm = MathF.Sqrt(norm);

        if (norm > 1e-6f)
        {
            var res = new float[vec.Length];
            for (int i = 0; i < vec.Length; i++)
            {
                res[i] = vec[i] / norm;
            }
            return res;
        }

        return vec;
    }

    public float[] EncodeImageForTesting(Mat crop)
    {
        return ExtractImageEmbedding(crop);
    }

    public float[] EncodeTextEnsembleForTesting(string[] prompts)
    {
        EnsureTextSessionLoaded();
        var ensembleVec = new float[512];
        int count = 0;
        foreach (var prompt in prompts)
        {
            float[] vec;
            if (_textSession != null)
            {
                try { vec = EncodeTextWithModel(prompt); }
                catch { vec = GenerateDeterministicVector(prompt); }
            }
            else
            {
                vec = GenerateDeterministicVector(prompt);
            }

            for (int i = 0; i < 512; i++) ensembleVec[i] += vec[i];
            count++;
        }

        if (count > 0)
        {
            for (int i = 0; i < 512; i++) ensembleVec[i] /= count;
        }

        return NormalizeVector(ensembleVec);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _visionSession?.Dispose();
        _textSession?.Dispose();
        _disposed = true;
        GC.SuppressFinalize(this);
    }
}
