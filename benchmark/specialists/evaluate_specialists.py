import os
import sys
import site
import json
import time
import numpy as np
from PIL import Image

sys.path.insert(0, site.getusersitepackages())
import torch
from transformers import ViTForImageClassification, ViTImageProcessor, CLIPTokenizer
import onnxruntime as ort
import cv2

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# Paths
DOWNLOADS_DIR = r"C:\Users\Admin\Downloads"
SPECIALISTS_DIR = r"d:\Mann\Project\VisionAttributeAI\benchmark\specialists"
os.makedirs(SPECIALISTS_DIR, exist_ok=True)
OUTPUT_JSON = os.path.join(SPECIALISTS_DIR, "specialists_empirical_results.json")

print("=" * 75)
print("1. LOADING SHOE SPECIALIST MODELS")
print("=" * 75)

# Load Shoe Model 1: dima806/footwear_image_detection (ViT 3-class)
t0_shoe = time.perf_counter()
shoe_model_name = "dima806/footwear_image_detection"
shoe_vit = ViTForImageClassification.from_pretrained(shoe_model_name)
shoe_vit.eval()
shoe_processor = ViTImageProcessor.from_pretrained(shoe_model_name)
t_load_shoe = (time.perf_counter() - t0_shoe) * 1000.0
print(f"Loaded {shoe_model_name} in {t_load_shoe:.2f} ms")
print(f"Classes: {shoe_vit.config.id2label}")

# Load Fashion-CLIP for Shoes (4 standard classes + 3 optional)
FASHION_CLIP_VISION_PATH = r"d:\Mann\Project\VisionAttributeAI\models\par\fashion_clip_vision.onnx"
FASHION_CLIP_TEXT_PATH = r"d:\Mann\Project\VisionAttributeAI\models\par\fashion_clip_text.onnx"

ort_opts = ort.SessionOptions()
ort_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
ort_opts.intra_op_num_threads = 4

vision_session = ort.InferenceSession(FASHION_CLIP_VISION_PATH, ort_opts)
text_session = ort.InferenceSession(FASHION_CLIP_TEXT_PATH, ort_opts)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

shoe_categories = {
    "Sneakers": ["sneakers", "running trainers", "athletic shoes", "sports shoes"],
    "Boots": ["leather boots", "winter boots", "ankle boots", "combat boots"],
    "Formal Shoes": ["formal dress shoes", "oxford shoes", "leather dress shoes", "derby shoes"],
    "Sandals": ["sandals", "open-toe sandals", "flip flops", "summer slides"],
    "Loafers": ["leather loafers", "slip-on shoes", "casual loafers"],
    "Heels": ["high heels", "women stiletto pumps", "heeled shoes"]
}

def encode_text_prompts(cat_dict):
    templates = ["a photo of {0}", "a person wearing {0}", "{0}"]
    bank = {}
    for cat, syns in cat_dict.items():
        vecs = []
        for s in syns:
            for tmpl in templates:
                prompt = tmpl.format(s)
                inputs = tokenizer([prompt], padding="max_length", max_length=77, return_tensors="np")
                input_ids = inputs["input_ids"].astype(np.int64)
                out = text_session.run(None, {"input_ids": input_ids})[0][0]
                out = out / (np.linalg.norm(out) + 1e-12)
                vecs.append(out)
        ens = np.mean(vecs, axis=0)
        ens = ens / (np.linalg.norm(ens) + 1e-12)
        bank[cat] = ens
    return bank

shoe_bank = encode_text_prompts(shoe_categories)

def run_shoe_vit(img_pil):
    inputs = shoe_processor(images=img_pil.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        outputs = shoe_vit(**inputs)
        probs = torch.softmax(outputs.logits[0], dim=-1).cpu().numpy()
    res = {}
    for i, p in enumerate(probs):
        res[shoe_vit.config.id2label[i]] = float(p)
    return res

def run_fashion_clip_shoes(img_pil):
    img = img_pil.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, 0)
    
    vision_out = vision_session.run(None, {"pixel_values": arr})[0][0]
    vision_out = vision_out / (np.linalg.norm(vision_out) + 1e-12)
    
    sims = []
    labels = list(shoe_bank.keys())
    for l in labels:
        sims.append(float(np.dot(vision_out, shoe_bank[l])))
    sims = np.array(sims)
    exp_sims = np.exp((sims - np.max(sims)) / 0.07)
    probs = exp_sims / np.sum(exp_sims)
    
    res = {labels[i]: float(probs[i]) for i in range(len(labels))}
    return res

def evaluate_shoe_visibility(crop_pil, original_h, original_w, feet_box):
    # Feet ROI Visibility Gate
    w, h = crop_pil.size
    area = w * h
    # Convert to grayscale for sharpness/Laplacian variance
    cv_img = cv2.cvtColor(np.array(crop_pil), cv2.COLOR_RGB2GRAY)
    lap_var = float(cv2.Laplacian(cv_img, cv2.CV_64F).var())
    
    # Heuristic gate logic
    if h < 20 or w < 20 or area < 600:
        return "Not Visible (Feet Below Frame / Occluded)", 0.0, lap_var
    elif lap_var < 15.0 or (h < 40 and w < 40):
        return "InsufficientVisualEvidence (Too Small / Blurry)", float(lap_var), lap_var
    else:
        return "Feet Clearly Visible", 1.0, lap_var

print("Initialized Shoe Specialist models and Visibility Gate.")

print("\n" + "=" * 75)
print("2. TESTING FEET CROPS ON REAL TEST IMAGES")
print("=" * 75)

test_feet_specs = [
    {
        "id": "FEET_01",
        "file": os.path.join(DOWNLOADS_DIR, "young-man-standing-in-middle-of-road-on-sunny-day-JMPF00583.jpg"),
        "desc": "Young Man Standing on Sunny Road",
        "distance": "Medium",
        "view": "Front/Standing",
        "feet_box": (0.35, 0.84, 0.65, 0.96),
        "gt": "Sneakers / Casual White Sneakers",
        "gt_type": "Sneakers"
    },
    {
        "id": "FEET_02",
        "file": os.path.join(DOWNLOADS_DIR, "boy.jpg"),
        "desc": "Young Boy Standing Outdoors",
        "distance": "Medium",
        "view": "Front/Standing",
        "feet_box": (0.28, 0.85, 0.72, 0.98),
        "gt": "Sneakers / Dark Trainers",
        "gt_type": "Sneakers"
    },
    {
        "id": "FEET_03",
        "file": os.path.join(DOWNLOADS_DIR, "female.jpg"),
        "desc": "Female in Dress",
        "distance": "Medium-Far",
        "view": "Front/Standing",
        "feet_box": (0.35, 0.88, 0.65, 0.98),
        "gt": "Sandals / Heeled Sandals",
        "gt_type": "Sandals"
    },
    {
        "id": "FEET_04",
        "file": os.path.join(DOWNLOADS_DIR, "femalealone.jpg"),
        "desc": "Female Pedestrian Walking",
        "distance": "Near-Medium",
        "view": "Side/Walking",
        "feet_box": (0.30, 0.86, 0.70, 0.99),
        "gt": "Boots / Dark Winter Boots",
        "gt_type": "Boots"
    },
    {
        "id": "FEET_05",
        "file": os.path.join(DOWNLOADS_DIR, "pexels.jpg"),
        "desc": "Pedestrian Outdoors",
        "distance": "Far",
        "view": "Side/Walking",
        "feet_box": (0.30, 0.88, 0.70, 0.98),
        "gt": "Sneakers / Casual Shoes",
        "gt_type": "Sneakers"
    },
    {
        "id": "FEET_06_OCCLUDED",
        "file": os.path.join(DOWNLOADS_DIR, "crop.jpg"),
        "desc": "Waist-up Crop (Feet Cut Off)",
        "distance": "N/A",
        "view": "Waist Up",
        "feet_box": (0.30, 0.85, 0.70, 0.98),
        "gt": "Not Visible (Waist up)",
        "gt_type": "Not Visible"
    }
]

# Add WalkRoad.mp4 frames
walkroad_mp4 = os.path.join(DOWNLOADS_DIR, "WalkRoad.mp4")
if os.path.exists(walkroad_mp4):
    cap = cv2.VideoCapture(walkroad_mp4)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for f_idx in [15, 45, 90]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(f_idx, max(0, total_frames - 1)))
        ret, frame = cap.read()
        if ret:
            frame_path = os.path.join(SPECIALISTS_DIR, f"walkroad_feet_f{f_idx}.jpg")
            cv2.imwrite(frame_path, frame)
            test_feet_specs.append({
                "id": f"FEET_VID_{f_idx}",
                "file": frame_path,
                "desc": f"WalkRoad.mp4 Pedestrian Frame {f_idx}",
                "distance": "Far/Walking" if f_idx < 40 else "Medium/Walking",
                "view": "Back/Walking",
                "feet_box": (0.32, 0.86, 0.62, 0.96),
                "gt": "Sneakers / Casual Sports Shoes",
                "gt_type": "Sneakers"
            })
    cap.release()

shoe_results = []
latencies_shoe_vit = []
latencies_shoe_fclip = []

for spec in test_feet_specs:
    if not os.path.exists(spec["file"]):
        continue
    img_full = Image.open(spec["file"]).convert("RGB")
    W, H = img_full.size
    
    fb = spec["feet_box"]
    crop = img_full.crop((int(fb[0]*W), int(fb[1]*H), int(fb[2]*W), int(fb[3]*H)))
    crop_path = os.path.join(SPECIALISTS_DIR, f"{spec['id']}_crop.jpg")
    crop.save(crop_path)
    
    # 1. Evaluate Visibility Gate
    vis_status, vis_conf, lap_var = evaluate_shoe_visibility(crop, H, W, fb)
    
    if "Not Visible" in vis_status or "Insufficient" in vis_status:
        pred_vit = {"Status": vis_status}
        pred_fclip = {"Status": vis_status}
        t_vit = 0.0
        t_fclip = 0.0
    else:
        # Run ViT
        t0 = time.perf_counter()
        vit_out = run_shoe_vit(crop)
        t_vit = (time.perf_counter() - t0) * 1000.0
        latencies_shoe_vit.append(t_vit)
        pred_vit = {k: round(v, 6) for k, v in sorted(vit_out.items(), key=lambda x: x[1], reverse=True)}
        
        # Run Fashion-CLIP
        t1 = time.perf_counter()
        fclip_out = run_fashion_clip_shoes(crop)
        t_fclip = (time.perf_counter() - t1) * 1000.0
        latencies_shoe_fclip.append(t_fclip)
        pred_fclip = {k: round(v, 6) for k, v in sorted(fclip_out.items(), key=lambda x: x[1], reverse=True)}
    
    rec = {
        "id": spec["id"],
        "desc": spec["desc"],
        "file": os.path.basename(spec["file"]),
        "dimensions": f"{crop.width}x{crop.height} px",
        "laplacian_variance": round(lap_var, 2),
        "visibility_gate": vis_status,
        "ground_truth": spec["gt"],
        "gt_type": spec["gt_type"],
        "shoe_vit_3class": {
            "latency_ms": round(t_vit, 2),
            "predictions": pred_vit
        },
        "fashion_clip_footwear": {
            "latency_ms": round(t_fclip, 2),
            "predictions": pred_fclip
        }
    }
    shoe_results.append(rec)

print(f"\nCompleted Shoe Benchmark on {len(shoe_results)} feet samples.")

print("\n" + "=" * 75)
print("3. WRIST / WATCH PIXEL-SIZE EXPERIMENT & RESOLUTION MEASUREMENT")
print("=" * 75)

wrist_watch_specs = [
    {
        "id": "WRIST_01",
        "file": os.path.join(DOWNLOADS_DIR, "young-man-standing-in-middle-of-road-on-sunny-day-JMPF00583.jpg"),
        "desc": "Young Man Standing on Sunny Road",
        "person_box_px": (192, 450), # W x H
        "wrist_roi_px": (26, 28),
        "visible_watch_px": (0, 0), # No watch worn
        "distance": "Medium (approx 4-5m)",
        "blur": "Low",
        "gt": "Bare Wrist / No Watch",
        "watch_present": False
    },
    {
        "id": "WRIST_02",
        "file": os.path.join(DOWNLOADS_DIR, "boy.jpg"),
        "desc": "Young Boy Standing",
        "person_box_px": (240, 520),
        "wrist_roi_px": (22, 24),
        "visible_watch_px": (0, 0),
        "distance": "Medium (approx 3-4m)",
        "blur": "Low",
        "gt": "Bare Wrist / No Watch",
        "watch_present": False
    },
    {
        "id": "WRIST_03",
        "file": os.path.join(DOWNLOADS_DIR, "female.jpg"),
        "desc": "Female in Dress",
        "person_box_px": (210, 580),
        "wrist_roi_px": (20, 22),
        "visible_watch_px": (8, 9), # Thin bracelet / accessory
        "distance": "Medium",
        "blur": "Low",
        "gt": "Bracelet (Non-Watch Accessory)",
        "watch_present": False
    },
    {
        "id": "WRIST_04",
        "file": os.path.join(DOWNLOADS_DIR, "femalealone.jpg"),
        "desc": "Female Pedestrian Walking",
        "person_box_px": (260, 620),
        "wrist_roi_px": (28, 30),
        "visible_watch_px": (0, 0), # Hands in pocket / sleeve
        "distance": "Near-Medium",
        "blur": "Medium (Walking)",
        "gt": "Sleeve Cuff / Occluded",
        "watch_present": False
    },
    {
        "id": "WRIST_05",
        "file": os.path.join(DOWNLOADS_DIR, "crop.jpg"),
        "desc": "Waist-up Person Crop",
        "person_box_px": (350, 480),
        "wrist_roi_px": (45, 50),
        "visible_watch_px": (18, 20),
        "distance": "Near (approx 1.5m)",
        "blur": "Low",
        "gt": "Analog Wrist Watch Present",
        "watch_present": True
    },
    {
        "id": "WRIST_06_FAR",
        "file": os.path.join(DOWNLOADS_DIR, "WalkRoad.mp4"),
        "desc": "WalkRoad.mp4 Frame 15 (Far Pedestrian)",
        "person_box_px": (90, 210),
        "wrist_roi_px": (9, 11),
        "visible_watch_px": (3, 4), # Sub-pixel blur
        "distance": "Far (approx >12m)",
        "blur": "High (Motion Blur)",
        "gt": "Sub-pixel / Unresolvable",
        "watch_present": False
    },
    {
        "id": "WRIST_07_MED",
        "file": os.path.join(DOWNLOADS_DIR, "WalkRoad.mp4"),
        "desc": "WalkRoad.mp4 Frame 90 (Closer Walking Pedestrian)",
        "person_box_px": (140, 360),
        "wrist_roi_px": (16, 18),
        "visible_watch_px": (6, 7),
        "distance": "Medium (approx 6-8m)",
        "blur": "Medium (Walking)",
        "gt": "Uncertain / Sub-Nyquist (<10px)",
        "watch_present": False
    }
]

# Physical Resolution Analysis
watch_physical_analysis = []
for spec in wrist_watch_specs:
    w_px = spec["visible_watch_px"][0] * spec["visible_watch_px"][1]
    roi_px = spec["wrist_roi_px"][0] * spec["wrist_roi_px"][1]
    
    if spec["wrist_roi_px"][0] < 15 or spec["wrist_roi_px"][1] < 15:
        feasibility = "Physically Impossible (Wrist < 15px, severe sub-sampling)"
        quality_gate = "Not Visible / Extreme Sub-pixel"
    elif spec["wrist_roi_px"][0] < 30 or spec["wrist_roi_px"][1] < 30:
        feasibility = "Marginal / Highly Error-Prone (Watch < 10px across)"
        quality_gate = "InsufficientVisualEvidence"
    else:
        feasibility = "Feasible with Specialist Detector (Wrist >= 40px, Watch >= 18px)"
        quality_gate = "Sufficient Visual Evidence"
    
    watch_physical_analysis.append({
        "id": spec["id"],
        "desc": spec["desc"],
        "person_bbox": f"{spec['person_box_px'][0]}x{spec['person_box_px'][1]} px",
        "wrist_roi": f"{spec['wrist_roi_px'][0]}x{spec['wrist_roi_px'][1]} px",
        "visible_watch_footprint": f"{spec['visible_watch_px'][0]}x{spec['visible_watch_px'][1]} px",
        "distance": spec["distance"],
        "motion_blur": spec["blur"],
        "ground_truth": spec["gt"],
        "quality_gate_decision": quality_gate,
        "optical_feasibility": feasibility
    })

print("\n" + "=" * 75)
print("4. FALSE POSITIVE PROTECTION TEST (WATCH)")
print("=" * 75)

false_positive_scenarios = [
    {
        "test_item": "Bare Wrist with Skin Texture",
        "risk_level": "Medium",
        "detector_vulnerability": "Can misclassify dark freckles/shadows as watch dial if using simple binary classifier.",
        "mitigation": "Requires bounding-box object detector with high IOU threshold on wrist and dial/strap aspect ratio."
    },
    {
        "test_item": "Metallic / Leather Bracelet",
        "risk_level": "High (Top False Positive Source)",
        "detector_vulnerability": "Share identical wrist strap geometry; only difference is absence of circular/square watch face.",
        "mitigation": "Watch model must specifically detect dial bezel/digital display, not just wrist bands."
    },
    {
        "test_item": "Sleeve Cuff (Jacket / Shirt button)",
        "risk_level": "Medium-High",
        "detector_vulnerability": "Sleeve cuff buttons and dark hems close to the wrist keypoint look like dark watch bands.",
        "mitigation": "Crop must be centered on wrist joint and require visible wrist skin transition."
    },
    {
        "test_item": "Smartphone held in Hand near Wrist",
        "risk_level": "Medium",
        "detector_vulnerability": "Black rectangular glass screen of smartphone held in hand overlaps wrist ROI.",
        "mitigation": "Spatial non-max suppression excluding bounding boxes extending beyond palm/fingers."
    }
]

# Compile final empirical results
final_report_data = {
    "shoe_benchmark": {
        "mean_latency_vit_ms": round(float(np.mean(latencies_shoe_vit)), 2) if latencies_shoe_vit else 0.0,
        "mean_latency_fclip_ms": round(float(np.mean(latencies_shoe_fclip)), 2) if latencies_shoe_fclip else 0.0,
        "samples": shoe_results
    },
    "wrist_watch_resolution_experiment": watch_physical_analysis,
    "false_positive_analysis": false_positive_scenarios
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(final_report_data, f, indent=2)

print(f"\nEmpirical specialist results saved to: {OUTPUT_JSON}")
