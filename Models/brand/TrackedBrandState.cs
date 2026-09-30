namespace VisionAttributeAI.Models.Brand;

/// <summary>
/// Maintains temporal multi-frame brand evidence pools and state machines per region for a tracked person.
/// </summary>
public class TrackedBrandState
{
    private const int MaxPoolSize = 3;

    public BrandResult UpperBrand { get; set; } = BrandResult.NotVisible("Upper");
    public BrandResult LowerBrand { get; set; } = BrandResult.NotVisible("Lower");
    public BrandResult WatchBrand { get; set; } = BrandResult.NotVisible("Watch");
    public BrandResult ShoeBrand { get; set; } = BrandResult.NotVisible("Shoes");
    public BrandResult BagBrand { get; set; } = BrandResult.NotVisible("Bag");

    public List<BrandObservation> UpperObservations { get; } = new();
    public List<BrandObservation> LowerObservations { get; } = new();
    public List<BrandObservation> WatchObservations { get; } = new();
    public List<BrandObservation> ShoeObservations { get; } = new();
    public List<BrandObservation> BagObservations { get; } = new();

    public PersonBrandResult ToPersonBrandResult() => new()
    {
        UpperBrand = UpperBrand,
        LowerBrand = LowerBrand,
        WatchBrand = WatchBrand,
        ShoeBrand = ShoeBrand,
        BagBrand = BagBrand
    };

    public bool IsRegionStable(string region) => region.ToLowerInvariant() switch
    {
        "upper" or "uppertorso" => UpperBrand.IsStable,
        "lower" or "lowerbody" => LowerBrand.IsStable,
        "watch" or "wrist" => WatchBrand.IsStable,
        "shoe" or "shoes" or "feet" => ShoeBrand.IsStable,
        "bag" or "backpack" => BagBrand.IsStable,
        _ => false
    };

    public void UpdateWithObservation(
        string region,
        BrandObservation obs,
        bool isVideoMode,
        bool possibleIdSwitch,
        int requiredStableCount = 2)
    {
        if (possibleIdSwitch)
        {
            // Freeze updates during ambiguous track transitions / possible ID switches
            return;
        }

        var normalizedRegion = NormalizeRegion(region);
        var pool = GetPoolForRegion(normalizedRegion);
        var currentResult = GetResultForRegion(normalizedRegion);

        if (isVideoMode && currentResult.IsStable)
        {
            // Already reached stable state; preserve and stop inference
            return;
        }

        // Single Image / Offline Snapshot Mode
        if (!isVideoMode)
        {
            if (obs.PassedDualGate && obs.BrandName != "BrandUnknown")
            {
                SetResultForRegion(normalizedRegion, BrandResult.Candidate(
                    obs.BrandName,
                    normalizedRegion,
                    obs.Similarity,
                    obs.RunnerUpBrand,
                    obs.RunnerUpSimilarity,
                    obs.Margin,
                    obs.MatchedExemplarFile));
            }
            else
            {
                SetResultForRegion(normalizedRegion, BrandResult.Unknown(
                    normalizedRegion,
                    obs.Similarity,
                    obs.RunnerUpBrand,
                    obs.RunnerUpSimilarity,
                    obs.Margin));
            }
            return;
        }

        // Multi-frame Video Mode
        pool.Add(obs);
        while (pool.Count > MaxPoolSize)
        {
            pool.RemoveAt(0);
        }

        // Evaluate stable consensus from bounded pool
        var passingObs = pool.Where(o => o.PassedDualGate && o.BrandName != "BrandUnknown").ToList();
        if (passingObs.Count == 0)
        {
            SetResultForRegion(normalizedRegion, BrandResult.Unknown(
                normalizedRegion,
                obs.Similarity,
                obs.RunnerUpBrand,
                obs.RunnerUpSimilarity,
                obs.Margin));
            return;
        }

        // Group by brand name to check consistency
        var brandGroups = passingObs.GroupBy(o => o.BrandName).OrderByDescending(g => g.Count()).ToList();
        var topGroup = brandGroups.First();
        var topBrand = topGroup.Key;
        var topCount = topGroup.Count();
        var bestObsInGroup = topGroup.OrderByDescending(o => o.Similarity).First();

        if (topCount >= requiredStableCount)
        {
            // Stable brand established across >= 2 temporal frames
            SetResultForRegion(normalizedRegion, new BrandResult
            {
                BrandName = topBrand,
                State = BrandState.BrandStable,
                Region = normalizedRegion,
                Similarity = bestObsInGroup.Similarity,
                RunnerUpBrand = bestObsInGroup.RunnerUpBrand,
                RunnerUpSimilarity = bestObsInGroup.RunnerUpSimilarity,
                Margin = bestObsInGroup.Margin,
                MatchedExemplarFile = bestObsInGroup.MatchedExemplarFile,
                ObservationCount = topCount,
                LastUpdatedUtc = DateTime.UtcNow
            });
        }
        else
        {
            // Candidate brand evidence (1 observation so far)
            SetResultForRegion(normalizedRegion, new BrandResult
            {
                BrandName = topBrand,
                State = BrandState.BrandCandidate,
                Region = normalizedRegion,
                Similarity = bestObsInGroup.Similarity,
                RunnerUpBrand = bestObsInGroup.RunnerUpBrand,
                RunnerUpSimilarity = bestObsInGroup.RunnerUpSimilarity,
                Margin = bestObsInGroup.Margin,
                MatchedExemplarFile = bestObsInGroup.MatchedExemplarFile,
                ObservationCount = topCount,
                LastUpdatedUtc = DateTime.UtcNow
            });
        }
    }

    public void CopyFrom(TrackedBrandState source)
    {
        UpperBrand = source.UpperBrand;
        LowerBrand = source.LowerBrand;
        WatchBrand = source.WatchBrand;
        ShoeBrand = source.ShoeBrand;
        BagBrand = source.BagBrand;

        UpperObservations.Clear();
        UpperObservations.AddRange(source.UpperObservations);

        LowerObservations.Clear();
        LowerObservations.AddRange(source.LowerObservations);

        WatchObservations.Clear();
        WatchObservations.AddRange(source.WatchObservations);

        ShoeObservations.Clear();
        ShoeObservations.AddRange(source.ShoeObservations);

        BagObservations.Clear();
        BagObservations.AddRange(source.BagObservations);
    }

    private static string NormalizeRegion(string region) => region.ToLowerInvariant() switch
    {
        "upper" or "uppertorso" or "torso" => "Upper",
        "lower" or "lowerbody" or "legs" => "Lower",
        "watch" or "wrist" => "Watch",
        "shoe" or "shoes" or "feet" => "Shoes",
        "bag" or "backpack" => "Bag",
        _ => "Upper"
    };

    private List<BrandObservation> GetPoolForRegion(string normalizedRegion) => normalizedRegion switch
    {
        "Upper" => UpperObservations,
        "Lower" => LowerObservations,
        "Watch" => WatchObservations,
        "Shoes" => ShoeObservations,
        "Bag" => BagObservations,
        _ => UpperObservations
    };

    private BrandResult GetResultForRegion(string normalizedRegion) => normalizedRegion switch
    {
        "Upper" => UpperBrand,
        "Lower" => LowerBrand,
        "Watch" => WatchBrand,
        "Shoes" => ShoeBrand,
        "Bag" => BagBrand,
        _ => UpperBrand
    };

    public void SetResultForRegion(string normalizedRegion, BrandResult result)
    {
        switch (normalizedRegion)
        {
            case "Upper":
                UpperBrand = result;
                break;
            case "Lower":
                LowerBrand = result;
                break;
            case "Watch":
                WatchBrand = result;
                break;
            case "Shoes":
                ShoeBrand = result;
                break;
            case "Bag":
                BagBrand = result;
                break;
        }
    }
}
