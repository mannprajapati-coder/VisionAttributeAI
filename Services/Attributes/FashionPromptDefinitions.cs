using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using System.Text;

namespace VisionAttributeAI.Services.Attributes;

/// <summary>
/// Static, canonical prompt definitions and templates for zero-shot Fashion-CLIP attribute classification.
/// Guarantees exact deterministic hashing and equivalence across cache generation and runtime validation.
/// </summary>
public static class FashionPromptDefinitions
{
    public static readonly string[] Templates =
    [
        "a photo of a {0}",
        "a person wearing a {0}",
        "a {0}"
    ];

    public static readonly Dictionary<string, string[]> SexCategories = new(StringComparer.OrdinalIgnoreCase)
    {
        { "Male", ["a man", "a male person", "a young man", "an adult man", "a male face", "a guy", "a gentleman"] },
        { "Female", ["a woman", "a female person", "a young woman", "an adult woman", "a female face", "a girl", "a lady"] }
    };

    public static readonly Dictionary<string, string[]> UpperTypeCategories = new(StringComparer.OrdinalIgnoreCase)
    {
        { "T-Shirt", ["casual t-shirt", "round crewneck t-shirt", "short sleeve cotton tee", "collarless t-shirt", "printed graphic t-shirt", "plain crew neck tee", "half sleeve t-shirt", "oversized t-shirt", "loose fitting cotton t-shirt", "round neck top"] },
        { "Shirt", ["button-up collared shirt", "formal dress shirt with collar", "button-down long-sleeve shirt", "formal office shirt", "linen collared shirt", "casual button-down shirt with collar", "oxford cotton button-down shirt", "polo shirt with collar"] },
        { "Jacket", ["zipper jacket", "outer zip jacket", "bomber jacket", "windbreaker jacket", "denim jacket", "leather jacket", "padded winter coat"] },
        { "Blazer", ["formal suit blazer", "business suit jacket", "tailored formal blazer", "tuxedo suit jacket", "smart casual blazer"] },
        { "Hoodie", ["hoodie with drawstring hood", "hooded sweatshirt with pocket", "hoodie pullover", "fleece hoodie", "casual hooded top"] },
        { "Sweater", ["knit wool sweater", "crewneck knit sweater", "pullover knit jumper", "knitted cardigan", "textured warm sweater"] },
        { "Other", ["traditional costume", "specialized uniform"] }
    };

    public static readonly Dictionary<string, string[]> LowerTypeCategories = new(StringComparer.OrdinalIgnoreCase)
    {
        { "Jeans", ["denim jeans", "blue denim pants", "casual denim jeans", "faded denim pants", "classic blue jeans", "distressed denim"] },
        { "Trousers", ["formal trousers", "dress pants", "chinos", "cotton trousers", "formal slacks", "casual pants", "office trousers", "suit trousers", "khaki trousers", "cargo pants", "tailored trousers"] },
        { "Shorts", ["shorts", "casual short pants", "summer shorts", "athletic sports shorts", "bermuda shorts", "denim shorts"] },
        { "Leggings", ["black athletic leggings", "tight yoga pants", "stretchy workout tights", "fitted spandex leggings", "activewear leggings"] },
        { "Skirt", ["skirt", "pleated skirt", "mini skirt", "midi skirt", "a-line skirt"] },
        { "Dress", ["one-piece dress", "women's dress", "maxi dress", "evening dress", "summer dress"] },
        { "Other", ["costume", "traditional lower wear"] }
    };

    public static readonly Dictionary<string, string[]> ShoesTypeCategories = new(StringComparer.OrdinalIgnoreCase)
    {
        { "Sneakers", ["sneakers", "athletic shoes", "running trainers", "white sneakers", "casual sneakers", "tennis shoes", "chunky sneakers", "sport trainers"] },
        { "Boots", ["leather boots", "ankle boots", "winter boots", "combat boots", "chelsea boots"] },
        { "Formal Shoes", ["formal dress shoes", "oxford shoes", "leather loafers", "business shoes", "derby shoes"] },
        { "Sandals", ["sandals", "open-toe sandals", "flip flops", "slides"] },
        { "Other", ["traditional slippers", "specialized protective boots", "costume footwear"] }
    };

    public static readonly Dictionary<string, string[]> ColorCategories = new(StringComparer.OrdinalIgnoreCase)
    {
        { "Black", ["black clothing", "dark black fabric", "black jacket", "black top", "jet black"] },
        { "White", ["white clothing", "white top", "pure white fabric", "bright white"] },
        { "Blue", ["blue denim", "blue jeans", "light blue denim", "navy blue", "cyan blue", "faded blue jeans"] },
        { "Grey", ["grey fabric", "gray clothing", "heather grey", "charcoal grey"] },
        { "Brown", ["brown jacket", "dark brown clothing", "chocolate brown", "tan brown", "coffee color"] },
        { "Red", ["red clothing", "bright red", "crimson red", "scarlet"] },
        { "Maroon", ["maroon shirt", "burgundy clothing", "wine red fabric", "dark maroon top"] },
        { "Green", ["green clothing", "olive green", "forest green", "emerald green", "sage green"] },
        { "Beige", ["beige clothing", "cream fabric", "khaki clothing", "sand beige", "off-white"] },
        { "Yellow", ["yellow clothing", "bright yellow", "mustard yellow"] },
        { "Pink", ["pink clothing", "light pink", "rose pink", "hot pink"] },
        { "Purple", ["purple clothing", "violet fabric", "lavender top"] }
    };

    /// <summary>
    /// Computes a deterministic SHA256 checksum over all categories, labels, synonyms, and templates.
    /// If any prompt or template changes, this checksum changes automatically.
    /// </summary>
    public static byte[] ComputePromptSetChecksum()
    {
        using var sha256 = SHA256.Create();
        var sb = new StringBuilder();

        foreach (var t in Templates) sb.Append($"T:{t}|");

        void AppendDict(string name, Dictionary<string, string[]> dict)
        {
            sb.Append($"BANK:{name}|");
            foreach (var (k, v) in dict)
            {
                sb.Append($"CAT:{k}:");
                foreach (var syn in v) sb.Append($"{syn},");
                sb.Append('|');
            }
        }

        AppendDict("Sex", SexCategories);
        AppendDict("UpperType", UpperTypeCategories);
        AppendDict("LowerType", LowerTypeCategories);
        AppendDict("ShoesType", ShoesTypeCategories);
        AppendDict("UpperColor", ColorCategories);
        AppendDict("LowerColor", ColorCategories);

        return sha256.ComputeHash(Encoding.UTF8.GetBytes(sb.ToString()));
    }
}
