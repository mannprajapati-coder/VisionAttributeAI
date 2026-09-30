using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Microsoft.Extensions.Logging.Abstractions;
using OpenCvSharp;
using VisionAttributeAI.Models.Tracking;
using VisionAttributeAI.Services.Tracking;

namespace VisionAttributeAI.Tests;

public class CmcBenchmarkResult
{
    public string Scenario { get; set; } = string.Empty;
    public int Frame { get; set; }
    public float ExpectedDx { get; set; }
    public float EstimatedDx { get; set; }
    public float ErrorX { get; set; }
    public float ExpectedDy { get; set; }
    public float EstimatedDy { get; set; }
    public float ErrorY { get; set; }
    public double MotionScore { get; set; }
    public CameraMotionState MotionState { get; set; }
    public bool IsReliable { get; set; }
    public float Confidence { get; set; }
    public double ElapsedMs { get; set; }
}

public static class CameraMotionEstimatorBenchmark
{
    public static void RunBenchmark()
    {
        var estimator = new CameraMotionEstimator(NullLogger<CameraMotionEstimator>.Instance);
        var allResults = new List<CmcBenchmarkResult>();

        Console.WriteLine("================================================================================");
        Console.WriteLine("          ISOLATED CAMERA MOTION COMPENSATION (CMC) BENCHMARK                   ");
        Console.WriteLine("================================================================================");

        // Helper to draw textured background (brick wall / texture pattern so optical flow has realistic corners)
        Mat CreateScene(float bgOffsetX, float bgOffsetY, List<(float X, float Y, float W, float H)> persons)
        {
            var mat = new Mat(480, 640, MatType.CV_8UC3, new Scalar(220, 220, 220));

            // Draw rich background texture (grid/boxes/lines representing indoor office/street features)
            for (int r = -100; r < 580; r += 40)
            {
                for (int c = -100; c < 740; c += 50)
                {
                    int px = (int)(c + bgOffsetX);
                    int py = (int)(r + bgOffsetY);
                    if (px >= 0 && px < 620 && py >= 0 && py < 460)
                    {
                        Cv2.Rectangle(mat, new Rect(px, py, 20, 20), new Scalar(160, 160, 170), -1);
                        Cv2.Circle(mat, new Point(px + 30, py + 15), 5, new Scalar(120, 130, 140), -1);
                    }
                }
            }

            // Draw moving foreground person(s)
            foreach (var p in persons)
            {
                int px = (int)Math.Clamp(p.X, 0, 640 - p.W);
                int py = (int)Math.Clamp(p.Y, 0, 480 - p.H);
                Cv2.Rectangle(mat, new Rect(px, py + 40, (int)p.W, (int)p.H - 40), new Scalar(40, 50, 180), -1);
                Cv2.Circle(mat, new Point(px + (int)(p.W / 2), py + 25), (int)(p.W / 4), new Scalar(170, 140, 120), -1);
            }

            return mat;
        }

        void TestScenario(string name, int frames, Func<int, (float ExpDx, float ExpDy, Mat Prev, Mat Curr)> step)
        {
            Console.WriteLine($"\n--- TESTING: {name} ---");
            Console.WriteLine("Frame | Exp ΔX | Est ΔX | ErrX  | Exp ΔY | Est ΔY | ErrY  | Score | State         | Rel? | Conf  | Time(ms)");
            Console.WriteLine("---------------------------------------------------------------------------------------------------------");

            var scenarioResults = new List<CmcBenchmarkResult>();

            for (int f = 1; f <= frames; f++)
            {
                var (expDx, expDy, prev, curr) = step(f);

                var sw = Stopwatch.StartNew();
                var result = estimator.EstimateMotion(prev, curr);
                sw.Stop();

                float errX = Math.Abs(result.DeltaX - expDx);
                float errY = Math.Abs(result.DeltaY - expDy);

                var row = new CmcBenchmarkResult
                {
                    Scenario = name,
                    Frame = f,
                    ExpectedDx = expDx,
                    EstimatedDx = result.DeltaX,
                    ErrorX = errX,
                    ExpectedDy = expDy,
                    EstimatedDy = result.DeltaY,
                    ErrorY = errY,
                    MotionScore = result.MotionScore,
                    MotionState = result.MotionState,
                    IsReliable = result.IsReliable,
                    Confidence = result.Confidence,
                    ElapsedMs = sw.Elapsed.TotalMilliseconds
                };

                scenarioResults.Add(row);
                allResults.Add(row);

                Console.WriteLine($"{f,5} | {expDx,6:F1} | {result.DeltaX,6:F1} | {errX,5:F1} | {expDy,6:F1} | {result.DeltaY,6:F1} | {errY,5:F1} | {result.MotionScore,5:F2} | {result.MotionState,-13} | {result.IsReliable,-4} | {result.Confidence,5:P0} | {row.ElapsedMs,6:F2}ms");

                prev.Dispose();
                curr.Dispose();
            }

            double avgErrX = scenarioResults.Average(r => r.ErrorX);
            double maxErrX = scenarioResults.Max(r => r.ErrorX);
            Console.WriteLine($"Scenario Summary: Avg ErrX = {avgErrX:F2}px, Max ErrX = {maxErrX:F2}px");
        }

        // Scenario A: Stationary camera, 1 person
        TestScenario("SCENARIO A: Stationary Camera", 5, f =>
        {
            var p = new List<(float, float, float, float)> { (250f, 100f, 100f, 280f) };
            var prev = CreateScene(0, 0, p);
            var curr = CreateScene(0, 0, p);
            return (0f, 0f, prev, curr);
        });

        // Scenario B: Slow Pan (8px/frame)
        TestScenario("SCENARIO B: Slow Pan (8px/frame)", 5, f =>
        {
            float prevOffset = (f - 1) * 8f;
            float currOffset = f * 8f;
            var pPrev = new List<(float, float, float, float)> { (250f + prevOffset, 100f, 100f, 280f) };
            var pCurr = new List<(float, float, float, float)> { (250f + currOffset, 100f, 100f, 280f) };
            var prev = CreateScene(prevOffset, 0, pPrev);
            var curr = CreateScene(currOffset, 0, pCurr);
            return (8f, 0f, prev, curr);
        });

        // Scenario C: Fast Pan (75px/frame)
        TestScenario("SCENARIO C: Fast Pan (75px/frame)", 5, f =>
        {
            float prevOffset = (f - 1) * 75f;
            float currOffset = f * 75f;
            var pPrev = new List<(float, float, float, float)> { (100f + prevOffset, 100f, 100f, 280f) };
            var pCurr = new List<(float, float, float, float)> { (100f + currOffset, 100f, 100f, 280f) };
            var prev = CreateScene(prevOffset, 0, pPrev);
            var curr = CreateScene(currOffset, 0, pCurr);
            return (75f, 0f, prev, curr);
        });

        // Scenario D: Fast 120px Jump
        TestScenario("SCENARIO D: Fast 120px Shift", 3, f =>
        {
            float prevOffset = 0f;
            float currOffset = 120f;
            var pPrev = new List<(float, float, float, float)> { (150f, 100f, 100f, 280f) };
            var pCurr = new List<(float, float, float, float)> { (270f, 100f, 100f, 280f) };
            var prev = CreateScene(prevOffset, 0, pPrev);
            var curr = CreateScene(currOffset, 0, pCurr);
            return (120f, 0f, prev, curr);
        });

        // Scenario E: Person Moves (25px/frame), Camera Stationary (Expected Dx = 0!)
        TestScenario("SCENARIO E: Person Moves (25px), Camera Still (Exp: 0px)", 5, f =>
        {
            var pPrev = new List<(float, float, float, float)> { (150f + (f - 1) * 25f, 100f, 100f, 280f) };
            var pCurr = new List<(float, float, float, float)> { (150f + f * 25f, 100f, 100f, 280f) };
            var prev = CreateScene(0, 0, pPrev);
            var curr = CreateScene(0, 0, pCurr);
            return (0f, 0f, prev, curr); // Crucial: Camera is NOT moving!
        });

        // Scenario F: Person & Camera Both Move (Cam = +50px, Person moves opposite)
        TestScenario("SCENARIO F: Both Move (Cam +50px, Person -20px)", 5, f =>
        {
            float prevOffset = (f - 1) * 50f;
            float currOffset = f * 50f;
            var pPrev = new List<(float, float, float, float)> { (300f + prevOffset - (f - 1) * 20f, 100f, 100f, 280f) };
            var pCurr = new List<(float, float, float, float)> { (300f + currOffset - f * 20f, 100f, 100f, 280f) };
            var prev = CreateScene(prevOffset, 0, pPrev);
            var curr = CreateScene(currOffset, 0, pCurr);
            return (50f, 0f, prev, curr);
        });

        // Multi-Person Scene: 3 people moving independently while camera pans 40px
        TestScenario("MULTI-PERSON: 3 People + 40px Camera Pan", 5, f =>
        {
            float prevOffset = (f - 1) * 40f;
            float currOffset = f * 40f;
            var pPrev = new List<(float, float, float, float)>
            {
                (80f + prevOffset + (f-1)*5f, 100f, 70f, 260f),
                (250f + prevOffset - (f-1)*10f, 110f, 80f, 270f),
                (420f + prevOffset, 120f, 75f, 250f)
            };
            var pCurr = new List<(float, float, float, float)>
            {
                (80f + currOffset + f*5f, 100f, 70f, 260f),
                (250f + currOffset - f*10f, 110f, 80f, 270f),
                (420f + currOffset, 120f, 75f, 250f)
            };
            var prev = CreateScene(prevOffset, 0, pPrev);
            var curr = CreateScene(currOffset, 0, pCurr);
            return (40f, 0f, prev, curr);
        });

        // Performance Metrics (P50, P95)
        var latencies = allResults.Select(r => r.ElapsedMs).OrderBy(t => t).ToList();
        double p50 = latencies[(int)(latencies.Count * 0.50)];
        double p95 = latencies[(int)(latencies.Count * 0.95)];
        double avg = latencies.Average();

        Console.WriteLine("\n================================================================================");
        Console.WriteLine($"CMC LATENCY PERFORMANCE OVERHEAD (on 160x120 downscaled):");
        Console.WriteLine($"  Average CPU Time: {avg:F3} ms / frame");
        Console.WriteLine($"  P50 CPU Time:     {p50:F3} ms / frame");
        Console.WriteLine($"  P95 CPU Time:     {p95:F3} ms / frame");
        Console.WriteLine("================================================================================");
    }
}
