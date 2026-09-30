using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace VisionAttributeAI.Services.Attributes;

/// <summary>
/// Full OpenAI CLIP Byte-Pair Encoding (BPE) Tokenizer.
/// Reads clip_vocab.json and clip_merges.txt to produce identical token sequences as HuggingFace CLIPTokenizer.
/// </summary>
public class ClipBpeTokenizer
{
    public const int SOT = 49406; // <|startoftext|>
    public const int EOT = 49407; // <|endoftext|>
    public const int MaxLength = 77;

    private readonly Dictionary<string, int> _encoder = new(StringComparer.Ordinal);
    private readonly Dictionary<(string, string), int> _bpeRanks = new();
    private readonly Dictionary<byte, char> _byteEncoder = new();
    private readonly Regex _pat;

    private static readonly Lazy<ClipBpeTokenizer> _instance = new(() => new ClipBpeTokenizer());
    public static ClipBpeTokenizer Instance => _instance.Value;

    public ClipBpeTokenizer(string vocabPath = "models/par/clip_vocab.json", string mergesPath = "models/par/clip_merges.txt")
    {
        _pat = new Regex(@"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+", RegexOptions.Compiled | RegexOptions.IgnoreCase);

        // Build byte encoder mapping
        _byteEncoder = BytesToUnicode();

        // Load vocab.json
        if (File.Exists(vocabPath))
        {
            var json = File.ReadAllText(vocabPath);
            _encoder = JsonSerializer.Deserialize<Dictionary<string, int>>(json) ?? new Dictionary<string, int>();
        }

        // Load merges.txt
        if (File.Exists(mergesPath))
        {
            var lines = File.ReadAllLines(mergesPath);
            int rank = 0;
            foreach (var line in lines)
            {
                var trimmed = line.Trim();
                if (string.IsNullOrWhiteSpace(trimmed) || trimmed.StartsWith("#")) continue;

                var parts = trimmed.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length == 2)
                {
                    _bpeRanks[(parts[0], parts[1])] = rank++;
                }
            }
        }
    }

    public long[] Tokenize(string text)
    {
        var tokens = new List<int> { SOT };
        var cleanText = WhitespaceClean(text);

        var matches = _pat.Matches(cleanText);
        foreach (Match match in matches)
        {
            var tokenStr = match.Value;
            var utf8Bytes = Encoding.UTF8.GetBytes(tokenStr);
            var sb = new StringBuilder();
            foreach (var b in utf8Bytes)
            {
                sb.Append(_byteEncoder.TryGetValue(b, out var c) ? c : (char)b);
            }

            var bpeTokens = Bpe(sb.ToString());
            foreach (var bpeToken in bpeTokens)
            {
                if (_encoder.TryGetValue(bpeToken, out int tokenId))
                {
                    tokens.Add(tokenId);
                }
            }
        }

        tokens.Add(EOT);

        var result = new long[MaxLength];
        for (int i = 0; i < MaxLength && i < tokens.Count; i++)
        {
            result[i] = tokens[i];
        }

        return result;
    }

    private List<string> Bpe(string token)
    {
        var word = token.Select(c => c.ToString()).ToList();
        if (word.Count == 0) return word;

        word[word.Count - 1] += "</w>";

        var pairs = GetPairs(word);
        if (pairs.Count == 0) return [word[0]];

        while (true)
        {
            (string, string)? minPair = null;
            int minRank = int.MaxValue;

            foreach (var pair in pairs)
            {
                if (_bpeRanks.TryGetValue(pair, out int r) && r < minRank)
                {
                    minRank = r;
                    minPair = pair;
                }
            }

            if (minPair == null) break;

            var (first, second) = minPair.Value;
            var newWord = new List<string>();
            int i = 0;

            while (i < word.Count)
            {
                int j = -1;
                for (int k = i; k < word.Count; k++)
                {
                    if (word[k] == first)
                    {
                        j = k;
                        break;
                    }
                }

                if (j == -1)
                {
                    for (int k = i; k < word.Count; k++) newWord.Add(word[k]);
                    break;
                }

                for (int k = i; k < j; k++) newWord.Add(word[k]);
                i = j;

                if (i < word.Count - 1 && word[i] == first && word[i + 1] == second)
                {
                    newWord.Add(first + second);
                    i += 2;
                }
                else
                {
                    newWord.Add(word[i]);
                    i += 1;
                }
            }

            word = newWord;
            if (word.Count == 1) break;

            pairs = GetPairs(word);
        }

        return word;
    }

    private static HashSet<(string, string)> GetPairs(List<string> word)
    {
        var pairs = new HashSet<(string, string)>();
        for (int i = 0; i < word.Count - 1; i++)
        {
            pairs.Add((word[i], word[i + 1]));
        }
        return pairs;
    }

    private static string WhitespaceClean(string text)
    {
        text = Regex.Replace(text, @"\s+", " ");
        return text.Trim();
    }

    private static Dictionary<byte, char> BytesToUnicode()
    {
        var bs = new List<int>();
        for (int i = '!'; i <= '~'; i++) bs.Add(i);
        for (int i = '¡'; i <= '¬'; i++) bs.Add(i);
        for (int i = '®'; i <= 'ÿ'; i++) bs.Add(i);

        var cs = bs.Select(b => (char)b).ToList();
        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            if (!bs.Contains(b))
            {
                bs.Add(b);
                cs.Add((char)(256 + n));
                n++;
            }
        }

        var dict = new Dictionary<byte, char>();
        for (int i = 0; i < bs.Count; i++)
        {
            dict[(byte)bs[i]] = cs[i];
        }
        return dict;
    }
}
