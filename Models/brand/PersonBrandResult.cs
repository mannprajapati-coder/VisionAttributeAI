namespace VisionAttributeAI.Models.Brand;

public class PersonBrandResult
{
    public BrandResult UpperBrand { get; set; } = BrandResult.NotVisible("Upper");
    public BrandResult LowerBrand { get; set; } = BrandResult.NotVisible("Lower");
    public BrandResult WatchBrand { get; set; } = BrandResult.NotVisible("Watch");
    public BrandResult ShoeBrand { get; set; } = BrandResult.NotVisible("Shoes");
    public BrandResult BagBrand { get; set; } = BrandResult.NotVisible("Bag");

    public bool HasAnyAcceptedBrand =>
        IsAccepted(UpperBrand) || IsAccepted(LowerBrand) || IsAccepted(WatchBrand) || IsAccepted(ShoeBrand) || IsAccepted(BagBrand);

    public bool HasAnyStableBrand =>
        UpperBrand.IsStable || LowerBrand.IsStable || WatchBrand.IsStable || ShoeBrand.IsStable || BagBrand.IsStable;

    public BrandResult? TopDetectedBrand
    {
        get
        {
            var list = new[] { UpperBrand, LowerBrand, WatchBrand, ShoeBrand, BagBrand }
                .Where(b => IsAccepted(b))
                .OrderByDescending(b => b.Similarity)
                .ToList();

            return list.FirstOrDefault();
        }
    }

    private static bool IsAccepted(BrandResult b) =>
        b.State is BrandState.BrandCandidate or BrandState.BrandStable &&
        b.BrandName != "BrandUnknown" &&
        b.BrandName != "Unknown" &&
        b.BrandName != "Not Visible";
}
