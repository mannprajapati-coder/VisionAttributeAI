namespace VisionAttributeAI.Models.Brand;

public enum BrandState
{
    NotVisible = 0,
    InsufficientVisualEvidence = 1,
    NoLogoCandidate = 2,
    BrandUnknown = 3,
    BrandCandidate = 4,
    BrandStable = 5,
    ModelUnavailable = 6,
    UnsupportedRegion = 7,
    Analyzing = 8
}
