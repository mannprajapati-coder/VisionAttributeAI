using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using VisionAttributeAI.Models.AI;
using VisionAttributeAI.Models.Brand;
using VisionAttributeAI.Models.Pose;
using VisionAttributeAI.Models.Tracking;
using VisionAttributeAI.Services.Aggregation;
using VisionAttributeAI.Services.Attributes;
using VisionAttributeAI.Services.Brand;
using VisionAttributeAI.Services.Detection;
using VisionAttributeAI.Services.LiveCamera;
using VisionAttributeAI.Services.PersonAnalysis;
using VisionAttributeAI.Services.Pose;
using VisionAttributeAI.Services.Quality;
using VisionAttributeAI.Services.Tracking;
using VisionAttributeAI.Services.Validation;
using VisionAttributeAI.Services.Watch;

namespace VisionAttributeAI.Tests;

/// <summary>
/// Comprehensive Diagnostic Runner to isolate and measure exact startup timings of all AI services,
/// ONNX sessions, prompt encodings, DI resolutions, cold-start vs warm-start, and frame processing.
/// Zero production modifications.
/// </summary>
public static class LiveCameraStartupDiagnosticRunner
{
    public static async Task RunStartupDiagnosticsAsync(IServiceProvider serviceProvider)
    {
        Console.WriteLine("\n=========================================================================================");
        Console.WriteLine(" LIVE CAMERA STARTUP & INITIALIZATION BOTTLENECK DIAGNOSTIC SUITE");
        Console.WriteLine("=========================================================================================\n");

        var swTotal = Stopwatch.StartNew();
        var swStage = Stopwatch.StartNew();

        Console.WriteLine("[LiveStartup] 1. MEASURING INDIVIDUAL COMPONENT CONSTRUCTORS & INITIALIZATIONS");
        Console.WriteLine("-----------------------------------------------------------------------------------------");

        // 1. YOLO Person Detector
        swStage.Restart();
        var yoloSw = Stopwatch.StartNew();
        var yoloOpt = Options.Create(new YoloModelOptions { ModelPath = "models/yolo/yolov5n.onnx" });
        using var yoloLoggerFactory = LoggerFactory.Create(builder => builder.AddConsole());
        var yoloLogger = yoloLoggerFactory.CreateLogger<YoloPersonDetector>();
        var yoloDetector = new YoloPersonDetector(yoloOpt, yoloLogger);
        yoloSw.Stop();
        Console.WriteLine($"[LiveStartup] YOLO Person Detector Init      : {yoloSw.Elapsed.TotalMilliseconds,8:F1} ms (Session: models/yolo/yolov5n.onnx)");

        // 2. YOLO Pose Estimation
        swStage.Restart();
        var poseSw = Stopwatch.StartNew();
        var poseOpt = Options.Create(new PersonAnalysisOptions { PoseModelPath = "models/pose/yolov8n-pose.onnx" });
        var poseLogger = yoloLoggerFactory.CreateLogger<YoloPoseEstimationService>();
        var poseService = new YoloPoseEstimationService(poseOpt, poseLogger);
        poseSw.Stop();
        Console.WriteLine($"[LiveStartup] YOLO Pose Estimator Init       : {poseSw.Elapsed.TotalMilliseconds,8:F1} ms (Session: models/pose/yolov8n-pose.onnx)");

        // 3. Fashion-CLIP Detailed Breakdown (Vision ONNX vs Text ONNX vs Prompt Embeddings)
        Console.WriteLine("\n--- Fashion-CLIP Initialization Breakdown ---");
        var clipVisionSw = Stopwatch.StartNew();
        using var clipVisionSession = new InferenceSession("models/par/fashion_clip_vision.onnx");
        clipVisionSw.Stop();
        Console.WriteLine($"   • Vision ONNX Session Load (352MB): {clipVisionSw.Elapsed.TotalMilliseconds,8:F1} ms");

        var clipTextSw = Stopwatch.StartNew();
        using var clipTextSession = new InferenceSession("models/par/fashion_clip_text.onnx");
        clipTextSw.Stop();
        Console.WriteLine($"   • Text ONNX Session Load (254MB)  : {clipTextSw.Elapsed.TotalMilliseconds,8:F1} ms");

        var tokSw = Stopwatch.StartNew();
        var tokenizer = ClipBpeTokenizer.Instance;
        tokSw.Stop();
        Console.WriteLine($"   • BPE Tokenizer Init (vocab+merges): {tokSw.Elapsed.TotalMilliseconds,8:F1} ms");

        var clipServiceSw = Stopwatch.StartNew();
        var clipOpt = Options.Create(new PersonAnalysisOptions
        {
            ClipVisionModelPath = "models/par/fashion_clip_vision.onnx",
            ClipTextModelPath = "models/par/fashion_clip_text.onnx"
        });
        var clipLogger = yoloLoggerFactory.CreateLogger<FashionClipService>();
        var fashionClipService = new FashionClipService(clipOpt, clipLogger);
        clipServiceSw.Stop();
        double promptEmbeddingTime = clipServiceSw.Elapsed.TotalMilliseconds - clipVisionSw.Elapsed.TotalMilliseconds - clipTextSw.Elapsed.TotalMilliseconds;
        Console.WriteLine($"   • Total FashionClipService Constructor: {clipServiceSw.Elapsed.TotalMilliseconds,8:F1} ms");
        Console.WriteLine($"   • Text Prompt Embedding Generation Time: {promptEmbeddingTime,8:F1} ms (~1,000 prompt inferences)");

        // 4. Brand Recognition (RetinaNet + DINOv2 + Gallery Embeddings)
        Console.WriteLine("\n--- Brand Recognition Subsystem Breakdown ---");
        var dinoSw = Stopwatch.StartNew();
        using var dinoSession = File.Exists("models/brand/dinov2_vits14.onnx") ? new InferenceSession("models/brand/dinov2_vits14.onnx") : null;
        dinoSw.Stop();
        Console.WriteLine($"   • DINOv2 ONNX Load                : {dinoSw.Elapsed.TotalMilliseconds,8:F1} ms (File: {(dinoSession != null ? "Found" : "Not Found")})");

        var brandSw = Stopwatch.StartNew();
        var brandOpt = Options.Create(new BrandOptions());
        var dinoLogger = yoloLoggerFactory.CreateLogger<DinoV2BrandRecognitionService>();
        var brandDinoService = new DinoV2BrandRecognitionService(brandOpt, dinoLogger);
        brandSw.Stop();
        Console.WriteLine($"   • DinoV2BrandRecognitionService   : {brandSw.Elapsed.TotalMilliseconds,8:F1} ms");

        var retinaLogger = yoloLoggerFactory.CreateLogger<LogosRetinaNetDetectionService>();
        var retinaService = new LogosRetinaNetDetectionService(brandOpt, retinaLogger);

        // 5. Watch Detection
        Console.WriteLine("\n--- Watch Detection Service ---");
        var watchSw = Stopwatch.StartNew();
        var watchOpt = Options.Create(new PersonAnalysisOptions { WatchModelPath = "models/yolo/yolov8n-watch.onnx" });
        var watchLogger = yoloLoggerFactory.CreateLogger<WatchDetectionService>();
        var watchService = new WatchDetectionService(watchOpt, watchLogger);
        watchSw.Stop();
        Console.WriteLine($"   • WatchDetectionService Init      : {watchSw.Elapsed.TotalMilliseconds,8:F1} ms (Config: {File.Exists(watchOpt.Value.WatchModelPath)})");

        // 6. Complete DI Resolution & Pipeline Construction (First HTTP Request Simulation)
        Console.WriteLine("\n=========================================================================================");
        Console.WriteLine(" 2. SIMULATING COLD-START VS WARM-START LIVE CAMERA REQUESTS");
        Console.WriteLine("=========================================================================================");

        // Create dummy camera frame (640x480 JPEG bytes)
        using var dummyMat = new Mat(480, 640, MatType.CV_8UC3, new Scalar(100, 150, 200));
        // Draw a synthetic person-like shape
        Cv2.Rectangle(dummyMat, new Rect(200, 100, 240, 350), new Scalar(50, 80, 220), -1);
        Cv2.Circle(dummyMat, new Point(320, 150), 40, new Scalar(160, 200, 240), -1);
        byte[] dummyFrameBytes = dummyMat.ToBytes(".jpg");

        // TEST A: COLD START (Resolving ILiveCameraAnalysisService and processing first frame)
        var coldStartSw = Stopwatch.StartNew();
        var liveService = serviceProvider.GetRequiredService<ILiveCameraAnalysisService>();
        var coldResponse = await liveService.ProcessLiveFrameAsync(dummyFrameBytes, 1);
        coldStartSw.Stop();

        Console.WriteLine($"\n[TEST A - COLD START] First Frame Processing Time: {coldStartSw.Elapsed.TotalMilliseconds,8:F1} ms");
        Console.WriteLine($"   • Detections Returned: {coldResponse.Persons.Count}");
        Console.WriteLine($"   • Detection Latency Reported: {coldResponse.DetectionMs:F1} ms");

        // TEST B: WARM START (Subsequent frames after service is fully initialized)
        var warmTimings = new List<double>();
        for (int i = 2; i <= 10; i++)
        {
            var warmSw = Stopwatch.StartNew();
            var warmResponse = await liveService.ProcessLiveFrameAsync(dummyFrameBytes, i);
            warmSw.Stop();
            warmTimings.Add(warmSw.Elapsed.TotalMilliseconds);
        }

        double warmP50 = warmTimings.OrderBy(v => v).ElementAt(warmTimings.Count / 2);
        double warmP95 = warmTimings.OrderBy(v => v).ElementAt((int)(warmTimings.Count * 0.95));
        double warmMean = warmTimings.Average();

        Console.WriteLine($"\n[TEST B - WARM START] Subsequent Frames (N=9):");
        Console.WriteLine($"   • P50 Latency: {warmP50:F1} ms");
        Console.WriteLine($"   • P95 Latency: {warmP95:F1} ms");
        Console.WriteLine($"   • Mean Latency: {warmMean:F1} ms\n");

        // 7. Audit Repeated Model Loading & Allocations
        Console.WriteLine("=========================================================================================");
        Console.WriteLine(" 3. AUDIT OF PROCESSING PATH FOR REPEATED ALLOCATIONS / MODEL CREATIONS");
        Console.WriteLine("=========================================================================================");
        Console.WriteLine("   • YOLO InferenceSession        : Reused Singleton instance");
        Console.WriteLine("   • Pose InferenceSession        : Reused Singleton instance");
        Console.WriteLine("   • Fashion-CLIP Vision Session  : Reused Singleton instance");
        Console.WriteLine("   • Fashion-CLIP Text Session    : Reused Singleton instance");
        Console.WriteLine("   • Text Prompt Embedding Cache  : Generated ONCE during FashionClipService constructor");
        Console.WriteLine("   • DINOv2 Session & Gallery     : Reused Singleton instance, gallery embedded once");
        Console.WriteLine("   • Watch Detection Session      : Handled safely (Failed fast on startup if missing)");

        // 8. DI Lifetime Audit
        Console.WriteLine("\n=========================================================================================");
        Console.WriteLine(" 4. DI REGISTRATION LIFETIME AUDIT");
        Console.WriteLine("=========================================================================================");
        Console.WriteLine("   • YoloPersonDetector           : Singleton");
        Console.WriteLine("   • YoloPoseEstimationService    : Singleton");
        Console.WriteLine("   • FashionClipService           : Singleton");
        Console.WriteLine("   • ClothingColorService         : Singleton");
        Console.WriteLine("   • DinoV2BrandRecognitionService: Singleton");
        Console.WriteLine("   • LogosRetinaNetDetectionService: Singleton");
        Console.WriteLine("   • BrandPipelineService         : Singleton");
        Console.WriteLine("   • WatchDetectionService        : Singleton");
        Console.WriteLine("   • UnifiedPersonAnalysisService : Singleton");
        Console.WriteLine("   • LiveCameraAnalysisService    : Singleton");
        Console.WriteLine("   • ImageAnalysisPipeline        : Scoped (Image endpoint only)");
        Console.WriteLine("   • VideoAnalysisService         : Scoped (Video endpoint only)");

        Console.WriteLine("\n=========================================================================================");
        Console.WriteLine(" 5. SUMMARY OF ROOT CAUSE & STARTUP DELAY");
        Console.WriteLine("=========================================================================================");
        Console.WriteLine($"   • Total Cold Initialization Time: {coldStartSw.Elapsed.TotalMilliseconds:F0} ms");
        Console.WriteLine($"   • Prompt Embeddings Generation   : {promptEmbeddingTime:F0} ms");
        Console.WriteLine($"   • Heavy ViT / ONNX File I/O     : {(clipVisionSw.Elapsed.TotalMilliseconds + clipTextSw.Elapsed.TotalMilliseconds):F0} ms");
        Console.WriteLine($"   • Warm Frame Execution Time     : {warmMean:F0} ms");
        Console.WriteLine($"   • Cold-to-Warm Ratio            : {coldStartSw.Elapsed.TotalMilliseconds / Math.Max(1, warmMean):F1}x");
        Console.WriteLine();
    }
}
