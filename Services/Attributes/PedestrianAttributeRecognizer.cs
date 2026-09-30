using OpenCvSharp;
using VisionAttributeAI.Models.AI;

namespace VisionAttributeAI.Services.Attributes;

/// <summary>
/// Adapter implementing IPedestrianAttributeRecognizer by delegating to IPedestrianAttributeService.
/// </summary>
public class PedestrianAttributeRecognizer : IPedestrianAttributeRecognizer
{
    private readonly IPedestrianAttributeService _attributeService;

    public bool IsModelLoaded => _attributeService.IsModelLoaded;

    public PedestrianAttributeRecognizer(IPedestrianAttributeService attributeService)
    {
        _attributeService = attributeService;
    }

    public async Task<PersonAttributeResult> RecognizeAttributesAsync(Mat personCrop, CancellationToken cancellationToken = default)
    {
        var result = new PersonAttributeResult();

        if (!_attributeService.IsModelLoaded)
        {
            result.SetAttribute("Status", "PAR model offline", 0.0f);
            return result;
        }

        var dto = await _attributeService.RecognizeAttributesAsync(personCrop, cancellationToken);

        if (dto.AppearanceSex != "Unknown") result.SetAttribute("AppearanceSex", dto.AppearanceSex, dto.AppearanceSexConfidence);
        if (!string.IsNullOrEmpty(dto.AgeGroup) && dto.AgeGroup != "Unknown") result.SetAttribute("AgeGroup", dto.AgeGroup, dto.AgeGroupConfidence);
        if (!string.IsNullOrEmpty(dto.Orientation) && dto.Orientation != "Unknown") result.SetAttribute("Orientation", dto.Orientation, dto.OrientationConfidence);
        if (dto.SleeveType != "Unknown") result.SetAttribute("SleeveType", dto.SleeveType, dto.SleeveTypeConfidence);

        if (dto.Trousers) result.SetAttribute("Trousers", "Yes", dto.TrousersConfidence);
        if (dto.Shorts) result.SetAttribute("Shorts", "Yes", dto.ShortsConfidence);
        if (dto.SkirtOrDress) result.SetAttribute("SkirtOrDress", "Yes", dto.SkirtOrDressConfidence);
        if (dto.LongCoat) result.SetAttribute("LongCoat", "Yes", dto.LongCoatConfidence);

        if (dto.HasHat) result.SetAttribute("Hat", "Yes", dto.HatConfidence);
        if (dto.HasGlasses) result.SetAttribute("Glasses", "Yes", dto.GlassesConfidence);
        if (dto.HasHandBag) result.SetAttribute("HandBag", "Yes", dto.HandBagConfidence);
        if (dto.HasShoulderBag) result.SetAttribute("ShoulderBag", "Yes", dto.ShoulderBagConfidence);
        if (dto.HasBackpack) result.SetAttribute("Backpack", "Yes", dto.BackpackConfidence);
        if (dto.HasBoots) result.SetAttribute("Boots", "Yes", dto.BootsConfidence);

        foreach (var pattern in dto.ClothingPatterns)
        {
            result.SetAttribute(pattern, "Yes", dto.RawAttributes.GetValueOrDefault(pattern, 1.0f));
        }

        return result;
    }

    public void Dispose()
    {
        _attributeService.Dispose();
        GC.SuppressFinalize(this);
    }
}
