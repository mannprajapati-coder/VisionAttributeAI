using System.Collections.Concurrent;
using VisionAttributeAI.Models.Benchmark;
using VisionAttributeAI.Models.Tracking;

namespace VisionAttributeAI.Services.Benchmark;

/// <summary>
/// Development-only in-memory accuracy benchmark service.
/// Evaluates ground truth annotations against aggregated predictions without saving images to disk.
/// </summary>
public class BenchmarkService : IBenchmarkService
{
    private readonly ConcurrentDictionary<int, PersonGroundTruth> _groundTruths = new();
    private readonly ILogger<BenchmarkService> _logger;

    public BenchmarkService(ILogger<BenchmarkService> logger)
    {
        _logger = logger;
    }

    public void SetGroundTruth(PersonGroundTruth groundTruth)
    {
        ArgumentNullException.ThrowIfNull(groundTruth);
        _groundTruths[groundTruth.PersonId] = groundTruth;
        _logger.LogInformation("Set benchmark ground truth for Person #{PersonId}: Sex={Sex}, Upper={U}, Color={UC}, Lower={L}, Shoes={S}",
            groundTruth.PersonId, groundTruth.ExpectedSex, groundTruth.ExpectedUpperType, groundTruth.ExpectedUpperColor, groundTruth.ExpectedLowerType, groundTruth.ExpectedShoes);
    }

    public PersonGroundTruth? GetGroundTruth(int personId)
    {
        _groundTruths.TryGetValue(personId, out var gt);
        return gt;
    }

    public IReadOnlyDictionary<int, PersonGroundTruth> GetAllGroundTruths()
    {
        return _groundTruths;
    }

    public void RemoveGroundTruth(int personId)
    {
        _groundTruths.TryRemove(personId, out _);
    }

    public void Reset()
    {
        _groundTruths.Clear();
        _logger.LogInformation("Reset all benchmark ground truth records.");
    }

    public BenchmarkEvaluationReport EvaluateTracks(IReadOnlyList<TrackedPersonState> tracks)
    {
        var report = new BenchmarkEvaluationReport
        {
            GeneratedAtUtc = DateTime.UtcNow,
            TotalTracksBenchmarked = 0
        };

        if (tracks == null || tracks.Count == 0 || _groundTruths.IsEmpty)
        {
            return report;
        }

        var attributeMetricsMap = new Dictionary<string, AttributeCounter>(StringComparer.OrdinalIgnoreCase)
        {
            { "AppearanceSex", new AttributeCounter("AppearanceSex") },
            { "UpperType", new AttributeCounter("UpperType") },
            { "UpperColor", new AttributeCounter("UpperColor") },
            { "LowerType", new AttributeCounter("LowerType") },
            { "LowerColor", new AttributeCounter("LowerColor") },
            { "Shoes", new AttributeCounter("Shoes") }
        };

        int tracksEvaluated = 0;

        foreach (var track in tracks)
        {
            if (!_groundTruths.TryGetValue(track.PersonId, out var gt))
            {
                continue;
            }

            tracksEvaluated++;

            // 1. Appearance Sex
            EvaluateField("AppearanceSex", gt.ExpectedSex, track.AppearanceSex, track.AppearanceConfidence, track.AppearanceState.ToString(), attributeMetricsMap["AppearanceSex"], report.TrackDetails, track.PersonId);

            // 2. Upper Type
            EvaluateField("UpperType", gt.ExpectedUpperType, track.UpperType, track.UpperTypeConfidence, track.UpperTypeState.ToString(), attributeMetricsMap["UpperType"], report.TrackDetails, track.PersonId);

            // 3. Upper Color
            EvaluateField("UpperColor", gt.ExpectedUpperColor, track.UpperColor, track.UpperColorConfidence, track.UpperColorState.ToString(), attributeMetricsMap["UpperColor"], report.TrackDetails, track.PersonId);

            // 4. Lower Type
            EvaluateField("LowerType", gt.ExpectedLowerType, track.LowerType, track.LowerTypeConfidence, track.LowerTypeState.ToString(), attributeMetricsMap["LowerType"], report.TrackDetails, track.PersonId);

            // 5. Lower Color
            EvaluateField("LowerColor", gt.ExpectedLowerColor, track.LowerColor, track.LowerColorConfidence, track.LowerColorState.ToString(), attributeMetricsMap["LowerColor"], report.TrackDetails, track.PersonId);

            // 6. Shoes
            EvaluateField("Shoes", gt.ExpectedShoes, track.ShoesType, track.ShoesConfidence, track.ShoesState.ToString(), attributeMetricsMap["Shoes"], report.TrackDetails, track.PersonId);
        }

        report.TotalTracksBenchmarked = tracksEvaluated;

        int grandTotalEvaluated = 0;
        int grandTotalCorrect = 0;

        foreach (var counter in attributeMetricsMap.Values)
        {
            report.AttributeMetrics.Add(new AttributeAccuracyMetric
            {
                AttributeName = counter.Name,
                TotalEvaluated = counter.TotalEvaluated,
                CorrectCount = counter.CorrectCount,
                IncorrectCount = counter.IncorrectCount,
                SkippedCount = counter.SkippedCount,
                UnknownCount = counter.UnknownCount,
                InsufficientEvidenceCount = counter.InsufficientEvidenceCount
            });

            grandTotalEvaluated += counter.TotalEvaluated;
            grandTotalCorrect += counter.CorrectCount;
        }

        report.OverallAccuracyPercent = grandTotalEvaluated > 0 ? ((float)grandTotalCorrect / grandTotalEvaluated) * 100f : 0f;
        return report;
    }

    private static void EvaluateField(
        string fieldName,
        string expected,
        string predicted,
        float confidence,
        string state,
        AttributeCounter counter,
        List<TrackComparisonDetail> detailsList,
        int personId)
    {
        if (string.IsNullOrWhiteSpace(expected) || expected.Equals("Skip", StringComparison.OrdinalIgnoreCase))
        {
            counter.SkippedCount++;
            return;
        }

        counter.TotalEvaluated++;

        // Clean up predicted string if it contains "Analyzing... (x/y)" or "(Score%)"
        string cleanPred = CleanPredictionString(predicted);

        bool isUnknown = cleanPred.Contains("Unknown", StringComparison.OrdinalIgnoreCase) || cleanPred.Contains("Uncertain", StringComparison.OrdinalIgnoreCase);
        bool isInsufficient = cleanPred.Contains("Insufficient", StringComparison.OrdinalIgnoreCase) || state.Contains("Insufficient", StringComparison.OrdinalIgnoreCase);
        bool isNotVisible = cleanPred.Contains("Not Visible", StringComparison.OrdinalIgnoreCase);

        if (isInsufficient)
        {
            counter.InsufficientEvidenceCount++;
        }
        else if (isUnknown)
        {
            counter.UnknownCount++;
        }

        // Strict comparison: Unknown/NotVisible is NOT considered correct unless expected explicitly is "Unknown" or "Not Visible"
        bool isMatch = expected.Equals(cleanPred, StringComparison.OrdinalIgnoreCase);

        if (isMatch)
        {
            counter.CorrectCount++;
        }
        else
        {
            counter.IncorrectCount++;
        }

        detailsList.Add(new TrackComparisonDetail
        {
            PersonId = personId,
            AttributeName = fieldName,
            Expected = expected,
            Predicted = predicted,
            Confidence = confidence,
            IsMatch = isMatch,
            State = state
        });
    }

    private static string CleanPredictionString(string val)
    {
        if (string.IsNullOrWhiteSpace(val)) return "Unknown";
        int idx = val.IndexOf('(');
        if (idx > 0)
        {
            return val[..idx].Trim();
        }
        return val.Trim();
    }

    private class AttributeCounter
    {
        public string Name { get; }
        public int TotalEvaluated { get; set; }
        public int CorrectCount { get; set; }
        public int IncorrectCount { get; set; }
        public int SkippedCount { get; set; }
        public int UnknownCount { get; set; }
        public int InsufficientEvidenceCount { get; set; }

        public AttributeCounter(string name)
        {
            Name = name;
        }
    }
}
