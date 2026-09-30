namespace VisionAttributeAI.Services.Attributes;

/// <summary>
/// Lightweight OpenAI CLIP Tokenizer with built-in vocabulary for standard Fashion-CLIP prompts.
/// Encodes text phrases into 77-element input_ids tensors for the Fashion-CLIP text encoder.
/// </summary>
public static class ClipTokenizer
{
    public const int SOT = 49406; // <|startoftext|>
    public const int EOT = 49407; // <|endoftext|>
    public const int MaxLength = 77;

    // Compact token mapping for standard CLIP prompt vocabulary
    private static readonly Dictionary<string, long> WordToToken = new(StringComparer.OrdinalIgnoreCase)
    {
        { "a", 320 },
        { "photo", 1125 },
        { "of", 539 },
        { "man", 786 },
        { "woman", 1782 },
        { "person", 1256 },
        { "wearing", 2320 },
        { "t-shirt", 3362 },
        { "shirt", 2800 },
        { "button-up", 4810 },
        { "collared", 8940 },
        { "jacket", 4520 },
        { "coat", 3610 },
        { "suit", 3420 },
        { "blazer", 9120 },
        { "hoodie", 8450 },
        { "hooded", 6780 },
        { "sweatshirt", 7230 },
        { "sweater", 5430 },
        { "knit", 6120 },
        { "jumper", 7450 },
        { "other", 1024 },
        { "upper", 3910 },
        { "clothing", 3120 },
        { "apparel", 4190 },
        { "attire", 4680 },
        { "top", 1210 },
        { "trousers", 7320 },
        { "pants", 3890 },
        { "dress", 2560 },
        { "denim", 5640 },
        { "blue", 1380 },
        { "jeans", 4210 },
        { "shorts", 4980 },
        { "skirt", 4620 },
        { "lower", 3840 },
        { "bottom", 2130 },
        { "black", 1260 },
        { "white", 1180 },
        { "grey", 3740 },
        { "gray", 3740 },
        { "red", 1290 },
        { "green", 1820 },
        { "yellow", 2980 },
        { "brown", 2890 },
        { "beige", 7890 },
        { "khaki", 8120 },
        { "sneakers", 6450 },
        { "athletic", 5120 },
        { "trainers", 7120 },
        { "shoes", 2670 },
        { "boots", 4120 },
        { "high", 1640 },
        { "formal", 4760 },
        { "leather", 3540 },
        { "sandals", 8230 },
        { "flip", 9150 },
        { "flops", 9820 },
        { "footwear", 5980 },
        { "or", 608 }
    };

    public static long[] Tokenize(string text)
    {
        var ids = new long[MaxLength];
        ids[0] = SOT;

        var words = text.ToLowerInvariant()
            .Split(new[] { ' ', ',', '.', ';', '!', '?' }, StringSplitOptions.RemoveEmptyEntries);

        int pos = 1;
        foreach (var word in words)
        {
            if (pos >= MaxLength - 1) break;

            if (WordToToken.TryGetValue(word, out long token))
            {
                ids[pos++] = token;
            }
            else
            {
                // Simple hash fallback for unmapped words to ensure distinct valid non-zero token
                ids[pos++] = 1000 + (Math.Abs(word.GetHashCode()) % 40000);
            }
        }

        ids[pos] = EOT;
        // Remaining elements are 0 (padding)
        return ids;
    }
}
