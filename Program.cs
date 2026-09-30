using VisionAttributeAI.Extensions;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddControllers();

// Add OpenAPI / Swagger endpoint exploration
builder.Services.AddOpenApi();

// Register Vision AI pipeline, detectors, and recognizers
builder.Services.AddVisionAiPipeline(builder.Configuration);

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
}

app.UseHttpsRedirection();

app.UseDefaultFiles();
app.UseStaticFiles();
app.MapControllers();

// Optional verification CLI commands
if (args.Length > 0 && args[0].Equals("--run-color-rootcause-diagnostic", StringComparison.OrdinalIgnoreCase))
{
    using var scope = app.Services.CreateScope();
    var detector = scope.ServiceProvider.GetRequiredService<VisionAttributeAI.Services.Detection.IPersonDetector>();
    var pose = scope.ServiceProvider.GetRequiredService<VisionAttributeAI.Services.Pose.IPoseEstimationService>();

    await VisionAttributeAI.Tests.ColorRootCauseDiagnosticSuite.RunAsync(
        detector, pose, "debug_color_rootcause");
    return;
}

if (args.Length > 0 && args[0].Equals("--scan-all-downloads", StringComparison.OrdinalIgnoreCase))
{
    using var scope = app.Services.CreateScope();
    var detector = scope.ServiceProvider.GetRequiredService<VisionAttributeAI.Services.Detection.IPersonDetector>();
    var pose = scope.ServiceProvider.GetRequiredService<VisionAttributeAI.Services.Pose.IPoseEstimationService>();
    var unified = scope.ServiceProvider.GetRequiredService<VisionAttributeAI.Services.PersonAnalysis.IUnifiedPersonAnalysisService>();
    var clip = scope.ServiceProvider.GetRequiredService<VisionAttributeAI.Services.Attributes.IFashionClipService>();

    await VisionAttributeAI.Tests.DownloadsColorAudit.RunAuditAsync(
        detector, pose, unified, clip, "debug_downloads_scan");
    return;
}

if (args.Length > 0 && args[0].Equals("--run-cmc-benchmark", StringComparison.OrdinalIgnoreCase))
{
    VisionAttributeAI.Tests.CameraMotionEstimatorBenchmark.RunBenchmark();
    return;
}

if (args.Length > 0 && args[0].Equals("--run-camera-motion-diagnostic", StringComparison.OrdinalIgnoreCase))
{
    VisionAttributeAI.Tests.CameraMotionDiagnosticRunner.RunFullInvestigation();
    return;
}

if (args.Length > 0 && args[0].Equals("--run-tracking-tests", StringComparison.OrdinalIgnoreCase))
{
    var testResults = VisionAttributeAI.Tests.TrackingTestSuite.RunAllTests();
    Console.WriteLine("=== TRACKING & VALIDATION TEST SUITE RESULTS ===");
    int passed = 0;
    foreach (var r in testResults)
    {
        string status = r.Passed ? "PASS" : "FAIL";
        Console.WriteLine($"[{status}] {r.Name} -> {r.Details}");
        if (r.Passed) passed++;
    }
    Console.WriteLine($"Total: {testResults.Count}, Passed: {passed}, Failed: {testResults.Count - passed}");
    return;
}

if (args.Length > 0 && args[0].Equals("--run-color-audit", StringComparison.OrdinalIgnoreCase))
{
    using var scope = app.Services.CreateScope();
    var detector = scope.ServiceProvider.GetRequiredService<VisionAttributeAI.Services.Detection.IPersonDetector>();
    var pose = scope.ServiceProvider.GetRequiredService<VisionAttributeAI.Services.Pose.IPoseEstimationService>();

    await VisionAttributeAI.Tests.ColorArchitectureAuditRunner.RunFullAuditAsync(detector, pose);
    return;
}

if (args.Length > 0 && args[0].Equals("--run-rigorous-benchmark", StringComparison.OrdinalIgnoreCase))
{
    using var scope = app.Services.CreateScope();
    var detector = scope.ServiceProvider.GetRequiredService<VisionAttributeAI.Services.Detection.IPersonDetector>();
    var pose = scope.ServiceProvider.GetRequiredService<VisionAttributeAI.Services.Pose.IPoseEstimationService>();

    await VisionAttributeAI.Tests.RigorousColorBenchmarkRunner.RunRigorousBenchmarkAsync(detector, pose);
    return;
}

if (args.Length > 0 && args[0].Equals("--run-fashion-clip-benchmark", StringComparison.OrdinalIgnoreCase))
{
    using var scope = app.Services.CreateScope();
    var detector = scope.ServiceProvider.GetRequiredService<VisionAttributeAI.Services.Detection.IPersonDetector>();
    var pose = scope.ServiceProvider.GetRequiredService<VisionAttributeAI.Services.Pose.IPoseEstimationService>();

    await VisionAttributeAI.Tests.FashionClipColorBenchmarkRunner.RunBenchmarkAsync(detector, pose);
    return;
}

if (args.Length > 0 && args[0].Equals("--run-quality-gating-benchmark", StringComparison.OrdinalIgnoreCase))
{
    using var scope = app.Services.CreateScope();
    var detector = scope.ServiceProvider.GetRequiredService<VisionAttributeAI.Services.Detection.IPersonDetector>();
    var pose = scope.ServiceProvider.GetRequiredService<VisionAttributeAI.Services.Pose.IPoseEstimationService>();

    await VisionAttributeAI.Tests.ColorQualityGatingBenchmarkRunner.RunQualityBenchmarkAsync(detector, pose);
    return;
}

if (args.Length > 0 && args[0].Equals("--run-camera-startup-diagnostic", StringComparison.OrdinalIgnoreCase))
{
    using var scope = app.Services.CreateScope();
    await VisionAttributeAI.Tests.LiveCameraStartupDiagnosticRunner.RunStartupDiagnosticsAsync(scope.ServiceProvider);
    return;
}

if (args.Length > 0 && args[0].Equals("--run-fashion-product-benchmark", StringComparison.OrdinalIgnoreCase))
{
    using var scope = app.Services.CreateScope();
    var detector = scope.ServiceProvider.GetRequiredService<VisionAttributeAI.Services.Detection.IPersonDetector>();
    var pose = scope.ServiceProvider.GetRequiredService<VisionAttributeAI.Services.Pose.IPoseEstimationService>();

    await VisionAttributeAI.Tests.FashionProductBaseColourBenchmarkRunner.RunBenchmarkAsync(detector, pose);
    return;
}

if (args.Length > 0 && args[0].Equals("--run-ground-truth-audit", StringComparison.OrdinalIgnoreCase))
{
    using var scope = app.Services.CreateScope();
    var detector = scope.ServiceProvider.GetRequiredService<VisionAttributeAI.Services.Detection.IPersonDetector>();
    var pose = scope.ServiceProvider.GetRequiredService<VisionAttributeAI.Services.Pose.IPoseEstimationService>();

    await VisionAttributeAI.Tests.GroundTruthAuditRunner.RunAuditAsync(detector, pose);
    return;
}

if (args.Length > 0 && args[0].Equals("--run-fabric-extraction-benchmark", StringComparison.OrdinalIgnoreCase))
{
    using var scope = app.Services.CreateScope();
    var detector = scope.ServiceProvider.GetRequiredService<VisionAttributeAI.Services.Detection.IPersonDetector>();
    var pose = scope.ServiceProvider.GetRequiredService<VisionAttributeAI.Services.Pose.IPoseEstimationService>();

    await VisionAttributeAI.Tests.FabricPixelExtractionBenchmarkRunner.RunExtractionBenchmarkAsync(detector, pose);
    return;
}

if (args.Length > 0 && (args[0].Equals("--run-cache-parity-and-benchmark", StringComparison.OrdinalIgnoreCase) ||
                        args[0].Equals("--generate-offline-caches", StringComparison.OrdinalIgnoreCase)))
{
    using var scope = app.Services.CreateScope();
    await VisionAttributeAI.Tests.StartupCacheParityTestRunner.RunFullParityAndBenchmarkAsync(scope.ServiceProvider);
    return;
}

if (args.Length > 0 && args[0].Equals("--run-pose-orientation-diagnostic", StringComparison.OrdinalIgnoreCase))
{
    using var scope = app.Services.CreateScope();
    var detector = scope.ServiceProvider.GetRequiredService<VisionAttributeAI.Services.Detection.IPersonDetector>();
    var pose = scope.ServiceProvider.GetRequiredService<VisionAttributeAI.Services.Pose.IPoseEstimationService>();

    await VisionAttributeAI.Tests.PoseOrientationDiagnosticRunner.RunDiagnosticsAsync(detector, pose);
    return;
}

app.Run();


