using VisionAttributeAI.Models.AI;
using VisionAttributeAI.Services.Aggregation;
using VisionAttributeAI.Services.Attributes;
using VisionAttributeAI.Services.Benchmark;
using VisionAttributeAI.Services.Cropping;
using VisionAttributeAI.Services.Detection;
using VisionAttributeAI.Services.LiveCamera;
using VisionAttributeAI.Services.PersonAnalysis;
using VisionAttributeAI.Services.Pipeline;
using VisionAttributeAI.Services.Pose;
using VisionAttributeAI.Services.Quality;
using VisionAttributeAI.Services.Tracking;
using VisionAttributeAI.Services.Validation;
using VisionAttributeAI.Services.Video;
using VisionAttributeAI.Services.Watch;

namespace VisionAttributeAI.Extensions;

public static class ServiceCollectionExtensions
{
    /// <summary>
    /// Registers Vision AI pipeline services, detectors, croppers, attribute recognizers, quality evaluators, video analysis, and configuration options.
    /// </summary>
    public static IServiceCollection AddVisionAiPipeline(this IServiceCollection services, IConfiguration configuration)
    {
        // Bind configuration options
        services.Configure<YoloModelOptions>(configuration.GetSection(YoloModelOptions.SectionName));
        services.Configure<ParModelOptions>(configuration.GetSection(ParModelOptions.SectionName));
        services.Configure<PersonAnalysisOptions>(configuration.GetSection(PersonAnalysisOptions.SectionName));
        services.Configure<VisionAttributeAI.Models.Tracking.LiveTrackingOptions>(configuration.GetSection(VisionAttributeAI.Models.Tracking.LiveTrackingOptions.SectionName));
        services.Configure<VisionAttributeAI.Models.Brand.BrandOptions>(configuration.GetSection(VisionAttributeAI.Models.Brand.BrandOptions.SectionName));
        services.Configure<ColorAnalysisOptions>(configuration.GetSection(ColorAnalysisOptions.SectionName));

        // Validators for incoming images and videos
        services.AddSingleton<IImageValidator, ImageValidator>();
        services.AddSingleton<IVideoValidator, VideoValidator>();
        services.AddSingleton<IPersonCandidateValidator, PersonCandidateValidator>();

        // Person ROI Cropper
        services.AddSingleton<IPersonCropper, PersonCropper>();

        // Model inference sessions (Singletons to avoid expensive reloading)
        services.AddSingleton<YoloPersonDetector>();
        services.AddSingleton<IPersonDetector>(sp => sp.GetRequiredService<YoloPersonDetector>());
        
        // Pose Estimation & Body Region Keypoint Extractor
        services.AddSingleton<IPoseEstimationService, YoloPoseEstimationService>();

        // Fashion-CLIP Sub-Region Feature & Attribute Classifier
        services.AddSingleton<IFashionClipService, FashionClipService>();

        // Fabric Color Classifier
        services.AddSingleton<IClothingColorService, ClothingColorService>();

        // Watch Detection Service
        services.AddSingleton<IWatchDetectionService, WatchDetectionService>();

        // Brand & Logo Recognition Subsystem (LOGOS RetinaNet + DINOv2 Gallery Matcher)
        services.AddSingleton<VisionAttributeAI.Services.Brand.IBrandLogoDetectionService, VisionAttributeAI.Services.Brand.LogosRetinaNetDetectionService>();
        services.AddSingleton<VisionAttributeAI.Services.Brand.IBrandRecognitionService, VisionAttributeAI.Services.Brand.DinoV2BrandRecognitionService>();
        services.AddSingleton<VisionAttributeAI.Services.Brand.IBrandPipelineService, VisionAttributeAI.Services.Brand.BrandPipelineService>();

        // Pedestrian Attribute Recognition (PAR) Service (Legacy Diagnostic)
        services.AddSingleton<PedestrianAttributeService>();
        services.AddSingleton<IPedestrianAttributeService>(sp => sp.GetRequiredService<PedestrianAttributeService>());
        services.AddSingleton<IPedestrianAttributeRecognizer, PedestrianAttributeRecognizer>();

        // Attribute-Specific Quality Evaluator, Tracking, Temporal Aggregation
        services.AddSingleton<IFrameQualityFilter, FrameQualityFilter>();
        services.AddSingleton<IAttributeQualityEvaluator, AttributeQualityEvaluator>();
        services.AddSingleton<IPersonTracker, IoUPersonTracker>();
        services.AddSingleton<ITemporalAppearanceAggregator, TemporalAppearanceAggregator>();

        // Shared Unified Person Analysis Engine (Used across Image, Video, and Camera)
        services.AddSingleton<IUnifiedPersonAnalysisService, UnifiedPersonAnalysisService>();

        // Live Camera Analysis Service
        services.AddSingleton<ILiveCameraAnalysisService, LiveCameraAnalysisService>();

        // Application AI Model Pre-warming & Readiness Hosted Service
        services.AddSingleton<VisionAttributeAI.Services.Warmup.ModelWarmupService>();
        services.AddSingleton<VisionAttributeAI.Services.Warmup.IModelWarmupService>(sp => sp.GetRequiredService<VisionAttributeAI.Services.Warmup.ModelWarmupService>());
        services.AddHostedService(sp => sp.GetRequiredService<VisionAttributeAI.Services.Warmup.ModelWarmupService>());

        // Accuracy Benchmark Service (Development Ground Truth Evaluation)
        services.AddSingleton<IBenchmarkService, BenchmarkService>();

        // Image & Video Pipelines
        services.AddScoped<IImageAnalysisPipeline, ImageAnalysisPipeline>();
        services.AddScoped<IVideoAnalysisService, VideoAnalysisService>();

        return services;
    }
}
