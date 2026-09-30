using Microsoft.Extensions.Options;
using VisionAttributeAI.Models.AI;
using VisionAttributeAI.Models.Tracking;

namespace VisionAttributeAI.Services.Aggregation;

/// <summary>
/// Implements attribute-specific weighted rolling temporal aggregation, consensus verification,
/// hysteresis anti-flicker protection, and explicit 5-state lifecycle management.
/// Formula: AttributeWeight = AttributeVisualQuality * ClassificationReliability.
/// </summary>
public class TemporalAppearanceAggregator : ITemporalAppearanceAggregator
{
    private readonly PersonAnalysisOptions _options;
    private readonly ILogger<TemporalAppearanceAggregator> _logger;

    public TemporalAppearanceAggregator(
        IOptions<PersonAnalysisOptions> options,
        ILogger<TemporalAppearanceAggregator> logger)
    {
        _options = options.Value;
        _logger = logger;
    }

    public void Aggregate(TrackedPersonState track, PersonObservation newObservation)
    {
        ArgumentNullException.ThrowIfNull(track);
        ArgumentNullException.ThrowIfNull(newObservation);

        // Never modify a finalized track or one flagged with an active ID-switch risk
        if (track.IsFinalized || track.PossibleIdSwitch)
        {
            return;
        }

        // 1. Add valid numerical observation to track's rolling queue and update Best-Frame pools
        track.AddObservation(newObservation, _options.ObservationWindowSize, _options.TopKBestFrames);
        track.LatestVisibility = newObservation.Visibility;
        track.LatestQualityScores = newObservation.QualityScores;

        var observations = track.Observations;
        int minReq = _options.MinimumValidFrames;
        float marginThresh = _options.UnknownMarginThreshold;
        float minConsensus = _options.MinimumConsensusRatio;
        int maxHysteresis = _options.HysteresisContradictionFrames;

        // 2. Aggregate Appearance Sex
        AggregateAppearanceSex(track, observations, minReq, marginThresh, minConsensus, maxHysteresis);

        // 3. Aggregate Upper Clothing (Type & Color)
        AggregateUpperClothing(track, observations, minReq, marginThresh, minConsensus, maxHysteresis);

        // 4. Aggregate Lower Clothing (Type & Color)
        AggregateLowerClothing(track, observations, minReq, marginThresh, minConsensus, maxHysteresis);

        // 5. Aggregate Shoes
        AggregateShoes(track, observations, minReq, marginThresh, minConsensus, maxHysteresis);

        // 6. Watch Status
        AggregateWatch(track, newObservation);
    }

    public void FinalizeTrack(TrackedPersonState track)
    {
        ArgumentNullException.ThrowIfNull(track);

        // 1. Appearance Sex Finalization
        if (track.BestAppearanceFrames.Count > 0)
        {
            if (track.AppearanceSex == "Insufficient Evidence" || track.AppearanceSex == "Not Visible" || track.AppearanceSex.StartsWith("Analyzing") || string.IsNullOrEmpty(track.AppearanceSex))
            {
                var topApp = track.BestAppearanceFrames.OrderByDescending(f => f.QualityScore).ThenByDescending(f => f.Confidence).First();
                track.AppearanceSex = topApp.Prediction;
                track.AppearanceConfidence = topApp.Confidence;
            }
            if (track.AppearanceState == AttributeState.Analyzing || track.AppearanceState == AttributeState.InsufficientVisualEvidence || track.AppearanceState == AttributeState.NotVisible)
            {
                track.AppearanceState = AttributeState.Stable;
            }
        }
        else if (track.AppearanceState == AttributeState.Analyzing && track.Observations.Any(o => o.HasAppearanceObservation))
        {
            track.AppearanceState = track.AppearanceConsensus >= 0.50f ? AttributeState.Stable : AttributeState.Uncertain;
        }

        // 2. Upper Clothing Finalization
        if (track.BestUpperFrames.Count > 0)
        {
            if (track.UpperType == "Insufficient Evidence" || track.UpperType == "Not Visible" || track.UpperType.StartsWith("Analyzing") || string.IsNullOrEmpty(track.UpperType))
            {
                var topUpper = track.BestUpperFrames.OrderByDescending(f => f.QualityScore).ThenByDescending(f => f.Confidence).First();
                var parts = topUpper.Prediction.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length >= 2)
                {
                    if (track.UpperColor == "Insufficient Evidence" || track.UpperColor == "Not Visible" || track.UpperColor == "Unknown" || string.IsNullOrEmpty(track.UpperColor))
                        track.UpperColor = parts[0];
                    track.UpperType = string.Join(' ', parts.Skip(1));
                }
                else
                {
                    track.UpperType = topUpper.Prediction;
                }
                track.UpperTypeConfidence = topUpper.Confidence;
            }
            if (track.UpperTypeState == AttributeState.Analyzing || track.UpperTypeState == AttributeState.InsufficientVisualEvidence || track.UpperTypeState == AttributeState.NotVisible)
            {
                track.UpperTypeState = AttributeState.Stable;
            }
            if (track.UpperColorState == AttributeState.Analyzing || track.UpperColorState == AttributeState.InsufficientVisualEvidence || track.UpperColorState == AttributeState.NotVisible)
            {
                track.UpperColorState = AttributeState.Stable;
            }
        }
        else if (track.UpperTypeState == AttributeState.Analyzing && track.Observations.Any(o => o.HasUpperObservation))
        {
            track.UpperTypeState = track.UpperTypeConsensus >= 0.50f ? AttributeState.Stable : AttributeState.Uncertain;
            track.UpperColorState = track.UpperColorConsensus >= 0.50f ? AttributeState.Stable : AttributeState.Uncertain;
        }

        // 3. Lower Clothing Finalization
        if (track.BestLowerFrames.Count > 0)
        {
            if (track.LowerType == "Insufficient Evidence" || track.LowerType == "Not Visible" || track.LowerType.StartsWith("Analyzing") || string.IsNullOrEmpty(track.LowerType))
            {
                var topLower = track.BestLowerFrames.OrderByDescending(f => f.QualityScore).ThenByDescending(f => f.Confidence).First();
                var parts = topLower.Prediction.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length >= 2)
                {
                    if (track.LowerColor == "Insufficient Evidence" || track.LowerColor == "Not Visible" || track.LowerColor == "Unknown" || string.IsNullOrEmpty(track.LowerColor))
                        track.LowerColor = parts[0];
                    track.LowerType = string.Join(' ', parts.Skip(1));
                }
                else
                {
                    track.LowerType = topLower.Prediction;
                }
                track.LowerTypeConfidence = topLower.Confidence;
            }
            if (track.LowerTypeState == AttributeState.Analyzing || track.LowerTypeState == AttributeState.InsufficientVisualEvidence || track.LowerTypeState == AttributeState.NotVisible)
            {
                track.LowerTypeState = AttributeState.Stable;
            }
            if (track.LowerColorState == AttributeState.Analyzing || track.LowerColorState == AttributeState.InsufficientVisualEvidence || track.LowerColorState == AttributeState.NotVisible)
            {
                track.LowerColorState = AttributeState.Stable;
            }
        }
        else if (track.LowerTypeState == AttributeState.Analyzing && track.Observations.Any(o => o.HasLowerObservation))
        {
            track.LowerTypeState = track.LowerTypeConsensus >= 0.50f ? AttributeState.Stable : AttributeState.Uncertain;
            track.LowerColorState = track.LowerColorConsensus >= 0.50f ? AttributeState.Stable : AttributeState.Uncertain;
        }

        // 4. Shoes Finalization
        if (track.BestShoesFrames.Count > 0)
        {
            if (track.ShoesType == "Insufficient Evidence" || track.ShoesType == "Not Visible" || track.ShoesType.StartsWith("Analyzing") || string.IsNullOrEmpty(track.ShoesType))
            {
                var topShoes = track.BestShoesFrames.OrderByDescending(f => f.QualityScore).ThenByDescending(f => f.Confidence).First();
                track.ShoesType = topShoes.Prediction;
                track.ShoesConfidence = topShoes.Confidence;
            }
            if (track.ShoesState == AttributeState.Analyzing || track.ShoesState == AttributeState.InsufficientVisualEvidence || track.ShoesState == AttributeState.NotVisible)
            {
                track.ShoesState = AttributeState.Stable;
            }
        }
        else if (track.ShoesState == AttributeState.Analyzing && track.Observations.Any(o => o.HasShoesObservation))
        {
            track.ShoesState = track.ShoesConsensus >= 0.50f ? AttributeState.Stable : AttributeState.Uncertain;
        }

        // 5. Watch Finalization
        if (track.WatchPositiveObservations >= Math.Max(2, _options.WatchRequiredStableObservations - 1))
        {
            track.WatchState = AttributeState.Stable;
            track.WatchStatus = "Watch Detected";
        }
        else if (track.WatchNegativeObservations >= Math.Max(3, _options.WatchNegativeStableObservations - 1))
        {
            track.WatchState = AttributeState.Stable;
            track.WatchStatus = "No Watch Detected";
            if (track.WatchConfidence < 0.50f) track.WatchConfidence = 0.80f;
        }
        else if (track.WatchState == AttributeState.Analyzing)
        {
            track.WatchState = track.WatchPositiveObservations > 0 ? AttributeState.Stable : AttributeState.Uncertain;
            if (track.WatchPositiveObservations > 0) track.WatchStatus = "Watch Detected";
            else if (track.WatchNegativeObservations > 0) track.WatchStatus = "No Watch Detected";
        }
    }

    private void AggregateAppearanceSex(
        TrackedPersonState track,
        List<PersonObservation> observations,
        int minReq,
        float marginThresh,
        float minConsensus,
        int maxHysteresis)
    {
        var validObs = observations.Where(o => o.HasAppearanceObservation && o.SexConfidence > 0).ToList();
        int count = validObs.Count;

        if (!track.LatestVisibility.HeadVisible && count == 0)
        {
            if (track.BestAppearanceFrames.Count == 0)
            {
                track.AppearanceState = AttributeState.NotVisible;
                track.AppearanceSex = "Not Visible";
                track.AppearanceConfidence = 0f;
                track.AppearanceConsensus = 0f;
            }
            return;
        }

        if (count == 0)
        {
            if (track.BestAppearanceFrames.Count == 0)
            {
                track.AppearanceState = track.LatestQualityScores.HasAppearanceEvidence
                    ? AttributeState.Analyzing
                    : AttributeState.InsufficientVisualEvidence;
                track.AppearanceSex = track.LatestQualityScores.HasAppearanceEvidence ? "Analyzing..." : "Insufficient Evidence";
                track.AppearanceConfidence = 0.0f;
                track.AppearanceConsensus = 0.0f;
            }
            return;
        }

        float totalWeight = 0f;
        float weightedMale = 0f;
        float weightedFemale = 0f;
        int maleVotes = 0;
        int femaleVotes = 0;

        foreach (var obs in validObs)
        {
            float visualQ = Math.Max(0.1f, obs.QualityScores.AppearanceQuality);
            float reliability = Math.Clamp(obs.SexMargin / 0.30f, 0.25f, 1.0f);
            float w = visualQ * reliability;

            totalWeight += w;
            weightedMale += obs.MaleScore * w;
            weightedFemale += obs.FemaleScore * w;

            if (obs.MaleScore > obs.FemaleScore) maleVotes++;
            else femaleVotes++;
        }

        float avgMale = totalWeight > 0 ? (weightedMale / totalWeight) : 0.5f;
        float avgFemale = totalWeight > 0 ? (weightedFemale / totalWeight) : 0.5f;

        track.AggregatedMaleScore = avgMale;
        track.AggregatedFemaleScore = avgFemale;

        string candidateSex = (avgMale > avgFemale) ? "Male" : "Female";
        float candidateScore = Math.Max(avgMale, avgFemale);
        int matchingVotes = (candidateSex == "Male") ? maleVotes : femaleVotes;
        float consensusRatio = (float)matchingVotes / count;
        track.AppearanceConsensus = consensusRatio;

        // Always store clean candidate prediction
        track.AppearanceSex = candidateSex;
        track.AppearanceConfidence = candidateScore;

        if (count < minReq)
        {
            track.AppearanceState = AttributeState.Analyzing;
            return;
        }

        float margin = Math.Abs(avgMale - avgFemale);

        if (margin < marginThresh || consensusRatio < minConsensus)
        {
            track.AppearanceState = AttributeState.Uncertain;
            track.AppearanceContradictionStreak = 0;
            return;
        }

        // Hysteresis Rule
        if (track.AppearanceState == AttributeState.Stable && track.AppearanceSex != candidateSex)
        {
            track.AppearanceContradictionStreak++;
            if (track.AppearanceContradictionStreak < maxHysteresis)
            {
                return;
            }
        }

        track.AppearanceContradictionStreak = 0;
        track.AppearanceState = AttributeState.Stable;
    }

    private void AggregateUpperClothing(
        TrackedPersonState track,
        List<PersonObservation> observations,
        int minReq,
        float marginThresh,
        float minConsensus,
        int maxHysteresis)
    {
        var upperObs = observations.Where(o => o.HasUpperObservation).ToList();
        int count = upperObs.Count;

        if (!track.LatestVisibility.UpperBodyVisible && count == 0)
        {
            if (track.BestUpperFrames.Count == 0)
            {
                track.UpperTypeState = AttributeState.NotVisible;
                track.UpperType = "Not Visible";
                track.UpperTypeConfidence = 0.0f;
                track.UpperColorState = AttributeState.NotVisible;
                track.UpperColor = "Not Visible";
                track.UpperColorConfidence = 0.0f;
            }
            return;
        }

        if (count == 0)
        {
            if (track.BestUpperFrames.Count == 0)
            {
                track.UpperTypeState = AttributeState.InsufficientVisualEvidence;
                track.UpperType = "Insufficient Evidence";
                track.UpperColorState = AttributeState.InsufficientVisualEvidence;
                track.UpperColor = "Insufficient Evidence";
            }
            return;
        }

        // A. Aggregate Upper Type
        var (topType, typeConf, typeMargin, typeConsensus, typeScores) = AggregateCategoryWithReliability(
            upperObs,
            o => o.UpperTypeScores,
            o => o.QualityScores.UpperQuality,
            o => o.UpperTypeMargin);

        track.TopUpperTypeScores = typeScores;
        track.UpperTypeConsensus = typeConsensus;
        track.UpperType = topType;
        track.UpperTypeConfidence = typeConf;

        if (count < minReq)
        {
            track.UpperTypeState = AttributeState.Analyzing;
        }
        else if (typeMargin < marginThresh && typeConf < 0.45f || typeConsensus < minConsensus)
        {
            track.UpperTypeState = AttributeState.Uncertain;
            track.UpperTypeContradictionStreak = 0;
        }
        else
        {
            if (track.UpperTypeState == AttributeState.Stable && track.UpperType != topType)
            {
                track.UpperTypeContradictionStreak++;
                if (track.UpperTypeContradictionStreak >= maxHysteresis)
                {
                    track.UpperTypeState = AttributeState.Stable;
                    track.UpperTypeContradictionStreak = 0;
                }
            }
            else
            {
                track.UpperTypeContradictionStreak = 0;
                track.UpperTypeState = AttributeState.Stable;
            }
        }

        // B. Aggregate Upper Color
        var (topColor, colorConf, colorMargin, colorConsensus, colorScores) = AggregateCategoryWithReliability(
            upperObs,
            o => o.UpperColorScores,
            o => o.QualityScores.UpperQuality,
            o => o.UpperColorDetails != null ? o.UpperColorDetails.PrimaryPercentage - o.UpperColorDetails.SecondaryPercentage : 0.2f);

        track.UpperColorConsensus = colorConsensus;
        track.LatestUpperColorDetails = upperObs.LastOrDefault()?.UpperColorDetails;
        track.UpperColor = topColor;
        track.UpperColorConfidence = colorConf;

        if (count < minReq)
        {
            track.UpperColorState = AttributeState.Analyzing;
        }
        else if (colorMargin < marginThresh && colorConf < 0.45f || colorConsensus < minConsensus)
        {
            track.UpperColorState = AttributeState.Uncertain;
            track.UpperColorContradictionStreak = 0;
        }
        else
        {
            if (track.UpperColorState == AttributeState.Stable && track.UpperColor != topColor)
            {
                track.UpperColorContradictionStreak++;
                if (track.UpperColorContradictionStreak >= maxHysteresis)
                {
                    track.UpperColorState = AttributeState.Stable;
                    track.UpperColorContradictionStreak = 0;
                }
            }
            else
            {
                track.UpperColorContradictionStreak = 0;
                track.UpperColorState = AttributeState.Stable;
            }
        }
    }

    private void AggregateLowerClothing(
        TrackedPersonState track,
        List<PersonObservation> observations,
        int minReq,
        float marginThresh,
        float minConsensus,
        int maxHysteresis)
    {
        var lowerObs = observations.Where(o => o.HasLowerObservation).ToList();
        int count = lowerObs.Count;

        if (!track.LatestVisibility.LowerBodyVisible && count == 0)
        {
            if (track.BestLowerFrames.Count == 0)
            {
                track.LowerTypeState = AttributeState.NotVisible;
                track.LowerType = "Not Visible";
                track.LowerTypeConfidence = 0.0f;
                track.LowerColorState = AttributeState.NotVisible;
                track.LowerColor = "Not Visible";
                track.LowerColorConfidence = 0.0f;
            }
            return;
        }

        if (count == 0)
        {
            if (track.BestLowerFrames.Count == 0)
            {
                track.LowerTypeState = AttributeState.InsufficientVisualEvidence;
                track.LowerType = "Insufficient Evidence";
                track.LowerColorState = AttributeState.InsufficientVisualEvidence;
                track.LowerColor = "Insufficient Evidence";
            }
            return;
        }

        // A. Aggregate Lower Type
        var (topType, typeConf, typeMargin, typeConsensus, typeScores) = AggregateCategoryWithReliability(
            lowerObs,
            o => o.LowerTypeScores,
            o => o.QualityScores.LowerQuality,
            o => o.LowerTypeMargin);

        track.TopLowerTypeScores = typeScores;
        track.LowerTypeConsensus = typeConsensus;
        track.LowerType = topType;
        track.LowerTypeConfidence = typeConf;

        if (count < minReq)
        {
            track.LowerTypeState = AttributeState.Analyzing;
        }
        else if (typeMargin < marginThresh && typeConf < 0.45f || typeConsensus < minConsensus)
        {
            track.LowerTypeState = AttributeState.Uncertain;
            track.LowerTypeContradictionStreak = 0;
        }
        else
        {
            if (track.LowerTypeState == AttributeState.Stable && track.LowerType != topType)
            {
                track.LowerTypeContradictionStreak++;
                if (track.LowerTypeContradictionStreak >= maxHysteresis)
                {
                    track.LowerTypeState = AttributeState.Stable;
                    track.LowerTypeContradictionStreak = 0;
                }
            }
            else
            {
                track.LowerTypeContradictionStreak = 0;
                track.LowerTypeState = AttributeState.Stable;
            }
        }

        // B. Aggregate Lower Color
        var (topColor, colorConf, colorMargin, colorConsensus, colorScores) = AggregateCategoryWithReliability(
            lowerObs,
            o => o.LowerColorScores,
            o => o.QualityScores.LowerQuality,
            o => o.LowerColorDetails != null ? o.LowerColorDetails.PrimaryPercentage - o.LowerColorDetails.SecondaryPercentage : 0.2f);

        track.LowerColorConsensus = colorConsensus;
        track.LatestLowerColorDetails = lowerObs.LastOrDefault()?.LowerColorDetails;
        track.LowerColor = topColor;
        track.LowerColorConfidence = colorConf;

        if (count < minReq)
        {
            track.LowerColorState = AttributeState.Analyzing;
        }
        else if (colorMargin < marginThresh && colorConf < 0.45f || colorConsensus < minConsensus)
        {
            track.LowerColorState = AttributeState.Uncertain;
            track.LowerColorContradictionStreak = 0;
        }
        else
        {
            if (track.LowerColorState == AttributeState.Stable && track.LowerColor != topColor)
            {
                track.LowerColorContradictionStreak++;
                if (track.LowerColorContradictionStreak >= maxHysteresis)
                {
                    track.LowerColorState = AttributeState.Stable;
                    track.LowerColorContradictionStreak = 0;
                }
            }
            else
            {
                track.LowerColorContradictionStreak = 0;
                track.LowerColorState = AttributeState.Stable;
            }
        }
    }

    private void AggregateShoes(
        TrackedPersonState track,
        List<PersonObservation> observations,
        int minReq,
        float marginThresh,
        float minConsensus,
        int maxHysteresis)
    {
        var shoesObs = observations.Where(o => o.HasShoesObservation).ToList();
        int count = shoesObs.Count;

        if (!track.LatestVisibility.FeetVisible && count == 0)
        {
            if (track.BestShoesFrames.Count == 0)
            {
                track.ShoesState = AttributeState.NotVisible;
                track.ShoesType = "Not Visible";
                track.ShoesConfidence = 0.0f;
            }
            return;
        }

        if (count == 0)
        {
            if (track.BestShoesFrames.Count == 0)
            {
                track.ShoesState = AttributeState.InsufficientVisualEvidence;
                track.ShoesType = "Insufficient Evidence";
            }
            return;
        }

        var (topType, typeConf, typeMargin, typeConsensus, typeScores) = AggregateCategoryWithReliability(
            shoesObs,
            o => o.ShoesScores,
            o => o.QualityScores.ShoesQuality,
            o => o.ShoesMargin);

        track.TopShoesScores = typeScores;
        track.ShoesConsensus = typeConsensus;
        track.ShoesType = topType;
        track.ShoesConfidence = typeConf;

        if (count < minReq)
        {
            track.ShoesState = AttributeState.Analyzing;
        }
        else if (typeMargin < marginThresh && typeConf < 0.45f || typeConsensus < minConsensus)
        {
            track.ShoesState = AttributeState.Uncertain;
            track.ShoesContradictionStreak = 0;
        }
        else
        {
            if (track.ShoesState == AttributeState.Stable && track.ShoesType != topType)
            {
                track.ShoesContradictionStreak++;
                if (track.ShoesContradictionStreak >= maxHysteresis)
                {
                    track.ShoesState = AttributeState.Stable;
                    track.ShoesContradictionStreak = 0;
                }
            }
            else
            {
                track.ShoesContradictionStreak = 0;
                track.ShoesState = AttributeState.Stable;
            }
        }
    }

    private void AggregateWatch(TrackedPersonState track, PersonObservation newObservation)
    {
        if (newObservation.HasWatchObservation)
        {
            if (newObservation.WatchStatus == "WatchDetected")
            {
                track.WatchPositiveObservations++;
                track.WatchConfidence = Math.Max(track.WatchConfidence, newObservation.WatchConfidence);
                track.WatchDetails = newObservation.WatchDetails;
            }
            else if (newObservation.WatchStatus == "NoWatchDetected")
            {
                track.WatchNegativeObservations++;
                track.WatchDetails = newObservation.WatchDetails;
            }
        }

        // Determine current temporal state
        if (track.WatchPositiveObservations >= _options.WatchRequiredStableObservations)
        {
            track.WatchState = AttributeState.Stable;
            track.WatchStatus = "Watch Detected";
        }
        else if (track.WatchNegativeObservations >= _options.WatchNegativeStableObservations)
        {
            track.WatchState = AttributeState.Stable;
            track.WatchStatus = "No Watch Detected";
            if (track.WatchConfidence < 0.50f) track.WatchConfidence = 0.85f;
        }
        else if (track.WatchPositiveObservations > 0 || track.WatchNegativeObservations > 0)
        {
            track.WatchState = AttributeState.Analyzing;
            track.WatchStatus = track.WatchPositiveObservations > 0 ? "Watch Candidate" : "Analyzing...";
        }
        else if (!track.LatestVisibility.LeftWristVisible && !track.LatestVisibility.RightWristVisible)
        {
            track.WatchState = AttributeState.NotVisible;
            track.WatchStatus = "Not Visible";
            track.WatchConfidence = 0.0f;
        }
        else
        {
            track.WatchState = AttributeState.InsufficientVisualEvidence;
            track.WatchStatus = "Insufficient Evidence";
            track.WatchConfidence = 0.0f;
        }
    }

    private static (string TopCategory, float TopConfidence, float Margin, float ConsensusRatio, Dictionary<string, float> WeightedScores) AggregateCategoryWithReliability(
        List<PersonObservation> observations,
        Func<PersonObservation, Dictionary<string, float>> scoreSelector,
        Func<PersonObservation, float> qualitySelector,
        Func<PersonObservation, float> marginSelector)
    {
        var categoryWeights = new Dictionary<string, float>(StringComparer.OrdinalIgnoreCase);
        var categoryVoteCounts = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        float totalWeightSum = 0f;

        foreach (var obs in observations)
        {
            float visualQ = Math.Max(0.1f, qualitySelector(obs));
            float margin = marginSelector(obs);
            float reliability = Math.Clamp(margin / 0.25f, 0.25f, 1.0f);
            float w = visualQ * reliability;
            totalWeightSum += w;

            var scores = scoreSelector(obs);
            if (scores == null || scores.Count == 0) continue;

            string obsTopCat = string.Empty;
            float obsTopScore = -1f;

            foreach (var (cat, score) in scores)
            {
                categoryWeights[cat] = categoryWeights.GetValueOrDefault(cat, 0f) + (score * w);
                if (score > obsTopScore)
                {
                    obsTopScore = score;
                    obsTopCat = cat;
                }
            }

            if (!string.IsNullOrEmpty(obsTopCat))
            {
                categoryVoteCounts[obsTopCat] = categoryVoteCounts.GetValueOrDefault(obsTopCat, 0) + 1;
            }
        }

        if (totalWeightSum <= 0f || categoryWeights.Count == 0)
        {
            return ("Unknown", 0f, 0f, 0f, new Dictionary<string, float>());
        }

        // Normalize weighted scores
        var normalizedScores = new Dictionary<string, float>(StringComparer.OrdinalIgnoreCase);
        foreach (var (cat, weight) in categoryWeights)
        {
            normalizedScores[cat] = weight / totalWeightSum;
        }

        var sorted = normalizedScores.OrderByDescending(kvp => kvp.Value).ToList();
        var top = sorted[0];
        float secondScore = sorted.Count > 1 ? sorted[1].Value : 0f;
        float marginDiff = top.Value - secondScore;

        int topVotes = categoryVoteCounts.GetValueOrDefault(top.Key, 0);
        float consensusRatio = observations.Count > 0 ? (float)topVotes / observations.Count : 0f;

        return (top.Key, top.Value, marginDiff, consensusRatio, normalizedScores);
    }
}
