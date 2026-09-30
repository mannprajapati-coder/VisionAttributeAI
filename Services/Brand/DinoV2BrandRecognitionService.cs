using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using VisionAttributeAI.Models.Brand;

namespace VisionAttributeAI.Services.Brand;

/// <summary>
/// Pretrained DINOv2 (ViT-S/14) Feature Embedding and Multi-Exemplar Cosine Similarity Matching Service.
/// Implements dual-threshold decision logic: Cosine Similarity >= 0.70 and Runner-Up Margin >= 0.04.
/// </summary>
public class DinoV2BrandRecognitionService : IBrandRecognitionService, IDisposable
{
    private const int InputSize = 224;
    private const int EmbeddingDim = 384;

    private readonly BrandOptions _options;
    private readonly ILogger<DinoV2BrandRecognitionService> _logger;
    private InferenceSession? _session;
    private string _inputName = "images";
    private bool _disposed;

    private static readonly float[] Mean = [0.485f, 0.456f, 0.406f];
    private static readonly float[] Std = [0.229f, 0.224f, 0.225f];

    // In-memory normalized gallery embeddings
    private readonly List<GalleryExemplar> _gallery = new();

    public bool IsModelLoaded => _session != null;
    public int GalleryExemplarCount => _gallery.Count;

    public DinoV2BrandRecognitionService(
        IOptions<BrandOptions> options,
        ILogger<DinoV2BrandRecognitionService> logger)
    {
        _options = options.Value;
        _logger = logger;

        InitializeModelAndGallery();
    }

    private void InitializeModelAndGallery()
    {
        try
        {
            var modelPath = _options.DinoV2ModelPath;
            if (string.IsNullOrWhiteSpace(modelPath) || !File.Exists(modelPath))
            {
                _logger.LogWarning("DINOv2 ONNX model not found at '{ModelPath}'. Brand recognition disabled.", modelPath);
                return;
            }

            var sessionOptions = new Microsoft.ML.OnnxRuntime.SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL
            };

            _session = new InferenceSession(modelPath, sessionOptions);
            _inputName = _session.InputMetadata.Keys.FirstOrDefault() ?? "images";

            _logger.LogInformation("Successfully initialized DINOv2 ONNX from '{ModelPath}'. Input='{Input}'", modelPath, _inputName);

            // Load and pre-embed gallery
            LoadGallery();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to initialize DINOv2 Brand Recognition model from '{ModelPath}'.", _options.DinoV2ModelPath);
        }
    }

    private void LoadGallery()
    {
        var cachePath = _options.GalleryCachePath;
        var modelPath = _options.DinoV2ModelPath;
        var galleryDir = _options.GalleryPath;
        string failureReason = string.Empty;

        // 1. Attempt fast loading from binary cache (< 2ms)
        if (!string.IsNullOrWhiteSpace(cachePath) &&
            BrandGalleryCache.TryLoadCache(cachePath, modelPath, galleryDir, out var cachedExemplars, out failureReason, _logger))
        {
            _gallery.Clear();
            foreach (var ce in cachedExemplars)
            {
                _gallery.Add(new GalleryExemplar
                {
                    FileName = ce.FileName,
                    BrandName = ce.BrandName,
                    Embedding = ce.Embedding
                });
            }
            _logger.LogInformation("Loaded {Count} DINOv2 brand gallery exemplars instantly from binary cache '{CachePath}'.",
                _gallery.Count, cachePath);
            return;
        }

        // 2. If cache missing or invalid, generate offline/startup cache
        _logger.LogWarning("Brand gallery cache unavailable or invalid ({Reason}). Generating fresh brand gallery cache...", failureReason);
        if (!string.IsNullOrWhiteSpace(modelPath) && File.Exists(modelPath) &&
            !string.IsNullOrWhiteSpace(galleryDir) && Directory.Exists(galleryDir))
        {
            try
            {
                var generated = BrandGalleryCache.GenerateAndSaveCache(modelPath, galleryDir, cachePath, _logger);
                _gallery.Clear();
                foreach (var ge in generated)
                {
                    _gallery.Add(new GalleryExemplar
                    {
                        FileName = ge.FileName,
                        BrandName = ge.BrandName,
                        Embedding = ge.Embedding
                    });
                }
                _logger.LogInformation("Successfully initialized Brand Gallery with {Count} exemplars across {BrandCount} brands.",
                    _gallery.Count, _gallery.Select(g => g.BrandName).Distinct().Count());
                return;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to generate brand gallery cache.");
            }
        }

        // 3. Fallback to direct reading
        if (string.IsNullOrWhiteSpace(galleryDir) || !Directory.Exists(galleryDir))
        {
            _logger.LogWarning("Brand Gallery directory not found at '{GalleryDir}'.", galleryDir);
            return;
        }

        var imageFiles = Directory.GetFiles(galleryDir, "*.png");
        _logger.LogInformation("Loading and embedding {Count} brand gallery exemplars from '{GalleryDir}'...", imageFiles.Length, galleryDir);

        foreach (var file in imageFiles)
        {
            try
            {
                var fileName = Path.GetFileName(file);
                var brandName = ExtractBrandName(fileName);

                using var mat = Cv2.ImRead(file, ImreadModes.Color);
                if (mat.Empty())
                {
                    _logger.LogWarning("Failed to read gallery image '{File}'.", file);
                    continue;
                }

                var embedding = ExtractEmbedding(mat);
                _gallery.Add(new GalleryExemplar
                {
                    FileName = fileName,
                    BrandName = brandName,
                    Embedding = embedding
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to embed gallery exemplar '{File}'.", file);
            }
        }

        _logger.LogInformation("Successfully initialized Brand Gallery with {Count} exemplars across {BrandCount} brands.",
            _gallery.Count, _gallery.Select(g => g.BrandName).Distinct().Count());
    }

    public Task<BrandResult> RecognizeBrandAsync(Mat logoCrop, string regionName, CancellationToken cancellationToken = default)
    {
        if (!_options.Enabled || !IsModelLoaded || _session == null || _gallery.Count == 0 || logoCrop.Empty())
        {
            return Task.FromResult(BrandResult.Unknown(regionName));
        }

        try
        {
            var queryEmbedding = ExtractEmbedding(logoCrop);

            // Compute cosine similarity against all gallery exemplars
            var brandMaxSims = new Dictionary<string, (float MaxSim, string BestFile)>(StringComparer.OrdinalIgnoreCase);

            foreach (var brand in BrandOptions.SupportedBrands)
            {
                brandMaxSims[brand] = (0f, string.Empty);
            }

            foreach (var exemplar in _gallery)
            {
                float sim = CosineSimilarity(queryEmbedding, exemplar.Embedding);
                if (!brandMaxSims.TryGetValue(exemplar.BrandName, out var current) || sim > current.MaxSim)
                {
                    brandMaxSims[exemplar.BrandName] = (sim, exemplar.FileName);
                }
            }

            // Sort brands descending by max similarity
            var sortedBrands = brandMaxSims
                .OrderByDescending(kv => kv.Value.MaxSim)
                .ToList();

            if (sortedBrands.Count == 0)
            {
                return Task.FromResult(BrandResult.Unknown(regionName));
            }

            var top1 = sortedBrands[0];
            var top2 = sortedBrands.Count > 1 ? sortedBrands[1] : default;

            string top1Brand = top1.Key;
            float top1Sim = top1.Value.MaxSim;
            string bestFile = top1.Value.BestFile;

            string top2Brand = top2.Key ?? "None";
            float top2Sim = top2.Value.MaxSim;

            float margin = top1Sim - top2Sim;

            // Strict Dual Gate: Top1 Cosine Sim >= MinCosineSimilarity AND Runner-Up Margin >= MinMargin
            bool passesDualGate = top1Sim >= _options.MinCosineSimilarity && margin >= _options.MinRunnerUpMargin;

            if (passesDualGate)
            {
                return Task.FromResult(BrandResult.Candidate(
                    top1Brand,
                    regionName,
                    top1Sim,
                    top2Brand,
                    top2Sim,
                    margin,
                    bestFile));
            }
            else
            {
                return Task.FromResult(BrandResult.Unknown(
                    regionName,
                    top1Sim,
                    top2Brand,
                    top2Sim,
                    margin));
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error performing DINOv2 brand recognition on region '{Region}'.", regionName);
            return Task.FromResult(BrandResult.Unknown(regionName));
        }
    }

    private float[] ExtractEmbedding(Mat imageMat)
    {
        if (_session == null) throw new InvalidOperationException("DINOv2 model session is not initialized.");

        // Aspect-preserving resize with white padding to 224x224
        int origW = imageMat.Width;
        int origH = imageMat.Height;
        float scale = (float)InputSize / Math.Max(origW, origH);
        int nw = Math.Max(1, (int)MathF.Round(origW * scale));
        int nh = Math.Max(1, (int)MathF.Round(origH * scale));

        using var resizedMat = new Mat();
        Cv2.Resize(imageMat, resizedMat, new Size(nw, nh), 0, 0, InterpolationFlags.Linear);

        using var rgbResized = new Mat();
        Cv2.CvtColor(resizedMat, rgbResized, ColorConversionCodes.BGR2RGB);

        // Create 224x224 white canvas (255, 255, 255)
        using var paddedMat = new Mat(new Size(InputSize, InputSize), MatType.CV_8UC3, new Scalar(255, 255, 255));
        int offsetX = (InputSize - nw) / 2;
        int offsetY = (InputSize - nh) / 2;
        var roiRect = new Rect(offsetX, offsetY, nw, nh);
        rgbResized.CopyTo(new Mat(paddedMat, roiRect));

        // Create Tensor [1, 3, 224, 224]
        var tensor = new DenseTensor<float>(new[] { 1, 3, InputSize, InputSize });

        for (int y = 0; y < InputSize; y++)
        {
            for (int x = 0; x < InputSize; x++)
            {
                var pixel = paddedMat.At<Vec3b>(y, x);
                tensor[0, 0, y, x] = (pixel.Item0 / 255f - Mean[0]) / Std[0];
                tensor[0, 1, y, x] = (pixel.Item1 / 255f - Mean[1]) / Std[1];
                tensor[0, 2, y, x] = (pixel.Item2 / 255f - Mean[2]) / Std[2];
            }
        }

        var inputs = new[] { NamedOnnxValue.CreateFromTensor(_inputName, tensor) };
        using var outputs = _session.Run(inputs);
        var rawEmbTensor = outputs.First().AsTensor<float>();

        // Extract and L2 Normalize
        var embedding = new float[EmbeddingDim];
        float sumSq = 0f;
        for (int i = 0; i < EmbeddingDim; i++)
        {
            float val = rawEmbTensor[0, i];
            embedding[i] = val;
            sumSq += val * val;
        }

        float norm = MathF.Sqrt(sumSq) + 1e-8f;
        for (int i = 0; i < EmbeddingDim; i++)
        {
            embedding[i] /= norm;
        }

        return embedding;
    }

    private static float CosineSimilarity(float[] a, float[] b)
    {
        float dot = 0f;
        int len = Math.Min(a.Length, b.Length);
        for (int i = 0; i < len; i++)
        {
            dot += a[i] * b[i];
        }
        return dot;
    }

    private static string ExtractBrandName(string fileName)
    {
        var prefix = fileName.Split('_')[0];
        return prefix.ToLowerInvariant() switch
        {
            "nike" => "Nike",
            "adidas" => "Adidas",
            "puma" => "Puma",
            "underarmour" or "under_armour" => "Under Armour",
            "newbalance" or "new_balance" => "New Balance",
            "gucci" => "Gucci",
            "louisvuitton" or "louis_vuitton" or "lv" => "Louis Vuitton",
            _ => prefix
        };
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

    private class GalleryExemplar
    {
        public string FileName { get; set; } = string.Empty;
        public string BrandName { get; set; } = string.Empty;
        public float[] Embedding { get; set; } = Array.Empty<float>();
    }
}
