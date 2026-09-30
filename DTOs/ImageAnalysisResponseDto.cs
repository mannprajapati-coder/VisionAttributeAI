namespace VisionAttributeAI.DTOs;

public class ImageAnalysisResponseDto
{
    public bool Success { get; set; } = true;
    public string? ErrorMessage { get; set; }
    public int ImageWidth { get; set; }
    public int ImageHeight { get; set; }
    public int PersonCount => Persons.Count;
    public double TotalElapsedMs { get; set; }
    public double DetectionMs { get; set; }
    public double AttributeRecognitionMs { get; set; }
    public List<DetectedPersonDto> Persons { get; set; } = [];
}
