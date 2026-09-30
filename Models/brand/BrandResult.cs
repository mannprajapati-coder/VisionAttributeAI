namespace VisionAttributeAI.Models.Brand;

public class BrandResult
{
    public string BrandName { get; set; } = "Unknown";
    public BrandState State { get; set; } = BrandState.Analyzing;
    public float Similarity { get; set; }
    public string RunnerUpBrand { get; set; } = string.Empty;
    public float RunnerUpSimilarity { get; set; }
    public float Margin { get; set; }
    public int ObservationCount { get; set; }
    public bool IsStable => State == BrandState.BrandStable;
    public string Region { get; set; } = string.Empty;
    public string MatchedExemplarFile { get; set; } = string.Empty;
    public DateTime LastUpdatedUtc { get; set; } = DateTime.UtcNow;

    public static BrandResult NotVisible(string region) => new()
    {
        BrandName = "Not Visible",
        State = BrandState.NotVisible,
        Region = region
    };

    public static BrandResult InsufficientVisualEvidence(string region) => new()
    {
        BrandName = "Insufficient Visual Evidence",
        State = BrandState.InsufficientVisualEvidence,
        Region = region
    };

    public static BrandResult NoLogoCandidate(string region) => new()
    {
        BrandName = "No Logo Candidate",
        State = BrandState.NoLogoCandidate,
        Region = region
    };

    public static BrandResult ModelUnavailable(string region) => new()
    {
        BrandName = "Model Unavailable",
        State = BrandState.ModelUnavailable,
        Region = region
    };

    public static BrandResult UnsupportedRegion(string region) => new()
    {
        BrandName = "Unsupported Region",
        State = BrandState.UnsupportedRegion,
        Region = region
    };

    public static BrandResult Unknown(string region, float sim = 0f, string runnerUp = "", float runnerUpSim = 0f, float margin = 0f) => new()
    {
        BrandName = "BrandUnknown",
        State = BrandState.BrandUnknown,
        Region = region,
        Similarity = sim,
        RunnerUpBrand = runnerUp,
        RunnerUpSimilarity = runnerUpSim,
        Margin = margin
    };

    public static BrandResult Candidate(string brandName, string region, float sim, string runnerUp, float runnerUpSim, float margin, string exemplar = "") => new()
    {
        BrandName = brandName,
        State = BrandState.BrandCandidate,
        Region = region,
        Similarity = sim,
        RunnerUpBrand = runnerUp,
        RunnerUpSimilarity = runnerUpSim,
        Margin = margin,
        MatchedExemplarFile = exemplar,
        ObservationCount = 1
    };
}
