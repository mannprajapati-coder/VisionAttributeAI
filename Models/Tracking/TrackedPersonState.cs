using VisionAttributeAI.Models.AI;
using VisionAttributeAI.Models.Pose;

namespace VisionAttributeAI.Models.Tracking;

public enum TrackConfirmationState
{
    Tentative = 0,
    Confirmed = 1,
    TemporarilyLost = 2,
    Retired = 3
}

/// <summary>
/// Maintains active tracking state, tracking diagnostics, best-frame quality pools,
/// and multi-attribute consensus & hysteresis state for a single PersonId.
/// </summary>
public class TrackedPersonState
{
    public int PersonId { get; set; }
    public Guid TrackGenerationId { get; init; } = Guid.NewGuid();
    public TrackConfirmationState ConfirmationState { get; set; } = TrackConfirmationState.Tentative;
    public int ConfirmationHits { get; set; } = 1;
    public bool IsConfirmed => ConfirmationState == TrackConfirmationState.Confirmed || ConfirmationState == TrackConfirmationState.TemporarilyLost;
    public bool IsTemporarilyLost => ConfirmationState == TrackConfirmationState.TemporarilyLost;

    public BoundingBox CurrentBox { get; set; }
    public BoundingBox? PreviousBox { get; set; }
    public float DetectionConfidence { get; set; }
    public DateTime FirstSeenTimestampUtc { get; init; } = DateTime.UtcNow;
    public DateTime LastSeenTimestampUtc { get; set; } = DateTime.UtcNow;
    public int TotalFramesObserved { get; set; }
    public int MissedFrames { get; set; }
    public bool IsActive { get; set; } = true;
    public bool IsFinalized { get; set; } = false;

    // Tracking Diagnostics & IoU Switch Detection
    public int TrackAgeFrames => TotalFramesObserved;
    public float LastMatchedIoU { get; set; } = 1.0f;
    public float CenterDisplacement { get; set; }
    public float NormalizedCenterDistance { get; set; }
    public float AreaChangeRatio { get; set; } = 1.0f;
    public bool PossibleIdSwitch { get; set; }
    public string PossibleIdSwitchReason { get; set; } = string.Empty;

    // Latest Frame Pose & Region Geometry
    public BodyVisibilityResult LatestVisibility { get; set; } = new();
    public BodyRegions LatestRegions { get; set; } = new();
    public List<Keypoint> LatestKeypoints { get; set; } = new();
    public AttributeQualityScores LatestQualityScores { get; set; } = new();

    // Rolling history of valid numerical observations (maximum window size, e.g. 10)
    public List<PersonObservation> Observations { get; } = new();

    // Top-K Best-Frame Pools per Attribute (Stored in-memory metadata only, 0 disk storage)
    public List<BestFrameRecord> BestAppearanceFrames { get; } = new();
    public List<BestFrameRecord> BestUpperFrames { get; } = new();
    public List<BestFrameRecord> BestLowerFrames { get; } = new();
    public List<BestFrameRecord> BestShoesFrames { get; } = new();
    public List<BestFrameRecord> BestWristFrames { get; } = new();

    // 1. Appearance Sex Aggregated State & Consensus
    public AttributeState AppearanceState { get; set; } = AttributeState.Analyzing;
    public string AppearanceSex { get; set; } = "Analyzing... (0/5)";
    public float AppearanceConfidence { get; set; }
    public float AppearanceConsensus { get; set; }
    public float AggregatedMaleScore { get; set; }
    public float AggregatedFemaleScore { get; set; }
    public bool IsAppearanceStable => AppearanceState == AttributeState.Stable;
    public int SexValidCount => Observations.Count(o => o.HasAppearanceObservation && o.SexConfidence > 0);
    public int AppearanceContradictionStreak { get; set; }

    // 2. Upper Clothing (Type & Color) Aggregated State & Consensus
    public AttributeState UpperTypeState { get; set; } = AttributeState.Analyzing;
    public string UpperType { get; set; } = "Analyzing... (0/5)";
    public float UpperTypeConfidence { get; set; }
    public float UpperTypeConsensus { get; set; }
    public Dictionary<string, float> TopUpperTypeScores { get; set; } = new();
    public bool IsUpperStable => UpperTypeState == AttributeState.Stable;
    public int UpperValidCount => Observations.Count(o => o.HasUpperObservation);
    public int UpperTypeContradictionStreak { get; set; }

    public AttributeState UpperColorState { get; set; } = AttributeState.Analyzing;
    public string UpperColor { get; set; } = "Analyzing... (0/5)";
    public float UpperColorConfidence { get; set; }
    public float UpperColorConsensus { get; set; }
    public ColorClassificationResult? LatestUpperColorDetails { get; set; }
    public int UpperColorContradictionStreak { get; set; }

    // 3. Lower Clothing (Type & Color) Aggregated State & Consensus
    public AttributeState LowerTypeState { get; set; } = AttributeState.NotVisible;
    public string LowerType { get; set; } = "Not Visible";
    public float LowerTypeConfidence { get; set; }
    public float LowerTypeConsensus { get; set; }
    public Dictionary<string, float> TopLowerTypeScores { get; set; } = new();
    public bool IsLowerStable => LowerTypeState == AttributeState.Stable;
    public int LowerValidCount => Observations.Count(o => o.HasLowerObservation);
    public int LowerTypeContradictionStreak { get; set; }

    public AttributeState LowerColorState { get; set; } = AttributeState.NotVisible;
    public string LowerColor { get; set; } = "Not Visible";
    public float LowerColorConfidence { get; set; }
    public float LowerColorConsensus { get; set; }
    public ColorClassificationResult? LatestLowerColorDetails { get; set; }
    public int LowerColorContradictionStreak { get; set; }

    // 4. Shoes Aggregated State & Consensus
    public AttributeState ShoesState { get; set; } = AttributeState.NotVisible;
    public string ShoesType { get; set; } = "Not Visible";
    public float ShoesConfidence { get; set; }
    public float ShoesConsensus { get; set; }
    public Dictionary<string, float> TopShoesScores { get; set; } = new();
    public bool IsShoesStable => ShoesState == AttributeState.Stable;
    public int ShoesValidCount => Observations.Count(o => o.HasShoesObservation);
    public int ShoesContradictionStreak { get; set; }

    // 5. Watch Aggregated State & Temporal Evidence
    public AttributeState WatchState { get; set; } = AttributeState.NotVisible;
    public string WatchStatus { get; set; } = "Not Visible";
    public float WatchConfidence { get; set; }
    public string WatchDetails { get; set; } = string.Empty;
    public int WatchPositiveObservations { get; set; }
    public int WatchNegativeObservations { get; set; }
    public bool IsWatchStable => WatchState == AttributeState.Stable;

    // 6. Brand & Logo Recognition Aggregated State
    public VisionAttributeAI.Models.Brand.TrackedBrandState BrandState { get; set; } = new();
    public VisionAttributeAI.Models.Brand.PersonBrandResult LatestBrands => BrandState.ToPersonBrandResult();

    public int TotalValidObservations => Observations.Count;
    public bool IsFullyStable => IsAppearanceStable && (IsUpperStable || !LatestVisibility.UpperBodyVisible);

    public TrackedPersonState(int personId, BoundingBox initialBox, float confidence)
    {
        PersonId = personId;
        CurrentBox = initialBox;
        DetectionConfidence = confidence;
        TotalFramesObserved = 1;
    }

    public void AddObservation(PersonObservation observation, int maxWindowSize, int maxBestFrames = 5)
    {
        Observations.Add(observation);
        while (Observations.Count > maxWindowSize)
        {
            Observations.RemoveAt(0);
        }

        LatestVisibility = observation.Visibility;
        LatestQualityScores = observation.QualityScores;

        // Update Best-Frame Pools
        if (observation.HasAppearanceObservation && observation.QualityScores.AppearanceQuality >= 0.25f)
        {
            AddToBestPool(BestAppearanceFrames, new BestFrameRecord
            {
                FrameIndex = observation.FrameIndex,
                QualityScore = observation.QualityScores.AppearanceQuality,
                Prediction = observation.SexPrediction,
                Confidence = observation.SexConfidence,
                Details = $"{observation.Orientation} view (Margin: {observation.SexMargin:P0})"
            }, maxBestFrames);
        }

        if (observation.HasUpperObservation && observation.QualityScores.UpperQuality >= 0.25f)
        {
            AddToBestPool(BestUpperFrames, new BestFrameRecord
            {
                FrameIndex = observation.FrameIndex,
                QualityScore = observation.QualityScores.UpperQuality,
                Prediction = $"{observation.UpperColor} {observation.UpperType}",
                Confidence = observation.UpperTypeConfidence,
                Details = $"Torso Area: {observation.CropWidth * observation.CropHeight} px"
            }, maxBestFrames);
        }

        if (observation.HasLowerObservation && observation.QualityScores.LowerQuality >= 0.25f)
        {
            AddToBestPool(BestLowerFrames, new BestFrameRecord
            {
                FrameIndex = observation.FrameIndex,
                QualityScore = observation.QualityScores.LowerQuality,
                Prediction = $"{observation.LowerColor} {observation.LowerType}",
                Confidence = observation.LowerTypeConfidence,
                Details = $"Lower Quality: {observation.QualityScores.LowerQuality:F2}"
            }, maxBestFrames);
        }

        if (observation.HasShoesObservation && observation.QualityScores.ShoesQuality >= 0.25f)
        {
            AddToBestPool(BestShoesFrames, new BestFrameRecord
            {
                FrameIndex = observation.FrameIndex,
                QualityScore = observation.QualityScores.ShoesQuality,
                Prediction = observation.ShoesType,
                Confidence = observation.ShoesConfidence,
                Details = $"Footwear Quality: {observation.QualityScores.ShoesQuality:F2}"
            }, maxBestFrames);
        }

        if (observation.QualityScores.LeftWristQuality >= 0.30f || observation.QualityScores.RightWristQuality >= 0.30f)
        {
            float wristQ = Math.Max(observation.QualityScores.LeftWristQuality, observation.QualityScores.RightWristQuality);
            string wristSide = observation.QualityScores.RightWristQuality >= observation.QualityScores.LeftWristQuality ? "Right" : "Left";
            AddToBestPool(BestWristFrames, new BestFrameRecord
            {
                FrameIndex = observation.FrameIndex,
                QualityScore = wristQ,
                Prediction = observation.WatchStatus,
                Confidence = observation.WatchConfidence,
                Details = $"{wristSide} Wrist (Q: {wristQ:F2})"
            }, maxBestFrames);
        }
    }

    private static void AddToBestPool(List<BestFrameRecord> pool, BestFrameRecord item, int maxItems)
    {
        // Add item, order descending by quality score, and truncate to maxItems
        pool.Add(item);
        var sorted = pool.OrderByDescending(x => x.QualityScore).Take(maxItems).ToList();
        pool.Clear();
        pool.AddRange(sorted);
    }
}
