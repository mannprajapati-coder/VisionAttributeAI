using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.DependencyInjection;
using VisionAttributeAI.Services.Detection;
using VisionAttributeAI.Services.Pose;
using VisionAttributeAI.Services.Attributes;
using VisionAttributeAI.Services.Brand;
using VisionAttributeAI.Services.Watch;
using VisionAttributeAI.Services.LiveCamera;
using VisionAttributeAI.Services.PersonAnalysis;

namespace VisionAttributeAI.Services.Warmup;

public enum ModelWarmupStatus
{
    NotStarted,
    Initializing,
    Ready,
    Failed
}

public interface IModelWarmupService
{
    ModelWarmupStatus Status { get; }
    bool IsReady => Status == ModelWarmupStatus.Ready;
    double TotalWarmupMs { get; }
    string? ErrorMessage { get; }
    Task EnsureReadyAsync(CancellationToken cancellationToken = default);
}

/// <summary>
/// Background Hosted Service that eagerly pre-warms all singleton AI models and loads binary embedding caches
/// on application boot. Ensures live camera requests never incur first-request initialization stalls.
/// </summary>
public class ModelWarmupService : IModelWarmupService, IHostedService
{
    private readonly IServiceProvider _serviceProvider;
    private readonly ILogger<ModelWarmupService> _logger;
    private readonly TaskCompletionSource _readyTcs = new(TaskCreationOptions.RunContinuationsAsynchronously);

    public ModelWarmupStatus Status { get; private set; } = ModelWarmupStatus.NotStarted;
    public double TotalWarmupMs { get; private set; }
    public string? ErrorMessage { get; private set; }

    public ModelWarmupService(IServiceProvider serviceProvider, ILogger<ModelWarmupService> logger)
    {
        _serviceProvider = serviceProvider;
        _logger = logger;
    }

    public Task EnsureReadyAsync(CancellationToken cancellationToken = default)
    {
        if (Status == ModelWarmupStatus.Ready) return Task.CompletedTask;
        if (Status == ModelWarmupStatus.Failed) throw new InvalidOperationException($"AI pipeline warmup failed: {ErrorMessage}");
        return _readyTcs.Task.WaitAsync(cancellationToken);
    }

    public Task StartAsync(CancellationToken cancellationToken)
    {
        // Run warmup in background so application web server starts immediately
        _ = Task.Run(() => PerformWarmupAsync(cancellationToken), cancellationToken);
        return Task.CompletedTask;
    }

    public Task StopAsync(CancellationToken cancellationToken) => Task.CompletedTask;

    private Task PerformWarmupAsync(CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        Status = ModelWarmupStatus.Initializing;
        var sw = Stopwatch.StartNew();
        _logger.LogInformation("[PreWarm] Beginning eager application AI pipeline pre-warming...");

        try
        {
            using var scope = _serviceProvider.CreateScope();
            var sp = scope.ServiceProvider;

            // 1. YOLO Person Detector
            var yoloSw = Stopwatch.StartNew();
            var yolo = sp.GetRequiredService<IPersonDetector>();
            yoloSw.Stop();
            _logger.LogInformation("[PreWarm] • YOLO Person Detector ready in {Ms:F1} ms", yoloSw.Elapsed.TotalMilliseconds);

            // 2. YOLO Pose Estimator
            var poseSw = Stopwatch.StartNew();
            var pose = sp.GetRequiredService<IPoseEstimationService>();
            poseSw.Stop();
            _logger.LogInformation("[PreWarm] • YOLO Pose Estimator ready in {Ms:F1} ms", poseSw.Elapsed.TotalMilliseconds);

            // 3. Fashion-CLIP & Cached Prompt Embeddings
            var clipSw = Stopwatch.StartNew();
            var clip = sp.GetRequiredService<IFashionClipService>();
            clipSw.Stop();
            _logger.LogInformation("[PreWarm] • Fashion-CLIP & Prompt Cache ready in {Ms:F1} ms", clipSw.Elapsed.TotalMilliseconds);

            // 4. Brand Subsystem (RetinaNet + DINOv2 + Gallery Cache)
            var brandSw = Stopwatch.StartNew();
            var brandPipeline = sp.GetRequiredService<IBrandPipelineService>();
            brandSw.Stop();
            _logger.LogInformation("[PreWarm] • Brand Recognition Subsystem ready in {Ms:F1} ms", brandSw.Elapsed.TotalMilliseconds);

            // 5. Watch Detector (fails fast if model file unavailable)
            var watchSw = Stopwatch.StartNew();
            var watch = sp.GetRequiredService<IWatchDetectionService>();
            watchSw.Stop();
            _logger.LogInformation("[PreWarm] • Watch Detection Service ready in {Ms:F1} ms", watchSw.Elapsed.TotalMilliseconds);

            // 6. Unified Person Analysis & Live Camera Engine
            var liveSw = Stopwatch.StartNew();
            var liveService = sp.GetRequiredService<ILiveCameraAnalysisService>();
            liveSw.Stop();
            _logger.LogInformation("[PreWarm] • Live Camera Analysis Engine ready in {Ms:F1} ms", liveSw.Elapsed.TotalMilliseconds);

            sw.Stop();
            TotalWarmupMs = sw.Elapsed.TotalMilliseconds;
            Status = ModelWarmupStatus.Ready;
            _readyTcs.TrySetResult();

            _logger.LogInformation("[PreWarm] All AI models and embedding caches pre-warmed and ready in {Ms:F1} ms total! Live camera frames will process instantly.",
                TotalWarmupMs);
        }
        catch (Exception ex)
        {
            sw.Stop();
            TotalWarmupMs = sw.Elapsed.TotalMilliseconds;
            Status = ModelWarmupStatus.Failed;
            ErrorMessage = ex.Message;
            _readyTcs.TrySetException(ex);
            _logger.LogError(ex, "[PreWarm] AI pipeline pre-warming encountered an error: {Message}", ex.Message);
        }

        return Task.CompletedTask;
    }
}
