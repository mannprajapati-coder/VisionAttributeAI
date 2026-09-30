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
MODEL_B_DIR = r"d:\Mann\Project\VisionAttributeAI\benchmark\model-b"
FASHION_CLIP_VISION_PATH = r"d:\Mann\Project\VisionAttributeAI\models\par\fashion_clip_vision.onnx"
FASHION_CLIP_TEXT_PATH = r"d:\Mann\Project\VisionAttributeAI\models\par\fashion_clip_text.onnx"
DOWNLOADS_DIR = r"C:\Users\Admin\Downloads"
RESULTS_JSON = os.path.join(MODEL_B_DIR, "empirical_benchmark_results.json")

print("=" * 70)
print("1. LOADING REAL MODEL B (ViT-Base 15-Class Clothing Classifier)")
print("=" * 70)

model_b = ViTForImageClassification.from_pretrained(MODEL_B_DIR)
model_b.eval()
processor_b = ViTImageProcessor.from_pretrained(MODEL_B_DIR)

id2label_b = model_b.config.id2label
print(f"Model B ID2LABEL ({len(id2label_b)} classes):")
for idx in sorted(id2label_b.keys(), key=lambda x: int(x)):
    print(f"  {idx}: {id2label_b[idx]}")

print("\n" + "=" * 70)
print("2. LOADING FASHION-CLIP ONNX ENCODERS")
print("=" * 70)

ort_opts = ort.SessionOptions()
ort_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
ort_opts.intra_op_num_threads = 4

vision_session = ort.InferenceSession(FASHION_CLIP_VISION_PATH, ort_opts)
text_session = ort.InferenceSession(FASHION_CLIP_TEXT_PATH, ort_opts)

# Fashion-CLIP Tokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# Categories defined in FashionClipService.cs
upper_categories = {
    "T-Shirt": ["t-shirt", "tshirt", "crewneck t-shirt", "short sleeve t-shirt", "cotton tee", "half sleeve t-shirt", "casual t-shirt", "round neck t-shirt"],
    "Shirt": ["button-up collared shirt", "formal dress shirt", "button-down long-sleeve shirt"],
    "Jacket": ["zipper jacket", "winter coat", "leather jacket", "bomber jacket"],
    "Blazer": ["formal suit blazer", "business blazer jacket", "tuxedo jacket"],
    "Hoodie": ["hoodie", "hooded sweatshirt with hood"],
    "Sweater": ["knit wool sweater", "sweater pullover", "knitwear jumper"],
    "Other": ["traditional costume", "specialized uniform"]
}

lower_categories = {
    "Trousers": ["formal trousers", "dress pants", "slacks"],
    "Jeans": ["blue jeans", "denim jeans", "denim pants"],
    "Shorts": ["shorts", "short pants"],
    "Skirt": ["skirt", "pleated skirt"],
    "Dress": ["dress", "one-piece dress"],
    "Other": ["bottom clothing", "pants"]
}

def encode_text_prompts(category_dict):
    templates = ["a photo of a {0}", "a person wearing a {0}", "a {0}"]
    bank = {}
    for cat_name, synonyms in category_dict.items():
        vecs = []
        for syn in synonyms:
            for tmpl in templates:
                prompt = tmpl.format(syn)
                inputs = tokenizer([prompt], padding="max_length", max_length=77, return_tensors="np")
                input_ids = inputs["input_ids"].astype(np.int64)
                out = text_session.run(None, {"input_ids": input_ids})[0]
                vec = out[0]
                vec = vec / (np.linalg.norm(vec) + 1e-12)
                vecs.append(vec)
        ensemble = np.mean(vecs, axis=0)
        ensemble = ensemble / (np.linalg.norm(ensemble) + 1e-12)
        bank[cat_name] = ensemble
    return bank

print("Precomputing Fashion-CLIP text embeddings...")
upper_text_bank = encode_text_prompts(upper_categories)
lower_text_bank = encode_text_prompts(lower_categories)

def preprocess_for_fashion_clip(img_pil):
    # Fashion CLIP expects 224x224, normalized with CLIP mean/std
    img = img_pil.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1)) # CHW
    arr = np.expand_dims(arr, 0) # 1, 3, 224, 224
    return arr

def run_fashion_clip(img_pil, target_bank):
    arr = preprocess_for_fashion_clip(img_pil)
    vision_out = vision_session.run(None, {"pixel_values": arr})[0][0]
    vision_out = vision_out / (np.linalg.norm(vision_out) + 1e-12)
    
    # Cosine similarities
    scores = {}
    labels = list(target_bank.keys())
    sims = []
    for label in labels:
        sim = float(np.dot(vision_out, target_bank[label]))
        sims.append(sim)
    
    # Softmax with temperature 0.05
    sims = np.array(sims)
    exp_sims = np.exp((sims - np.max(sims)) / 0.07)
    probs = exp_sims / np.sum(exp_sims)
    
    res = []
    for i, label in enumerate(labels):
        res.append((label, float(probs[i]), float(sims[i])))
    res.sort(key=lambda x: x[1], reverse=True)
    return res

def run_model_b(img_pil):
    inputs = processor_b(images=img_pil.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        outputs = model_b(**inputs)
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    
    res = []
    for i in range(len(probs)):
        label = id2label_b[str(i)] if str(i) in id2label_b else id2label_b[i]
        res.append((label, float(probs[i])))
    res.sort(key=lambda x: x[1], reverse=True)
    return res

print("Fashion-CLIP and Model B inference functions initialized.")

# Define real test samples from Downloads
test_sample_specs = [
    {
        "file": os.path.join(DOWNLOADS_DIR, "young-man-standing-in-middle-of-road-on-sunny-day-JMPF00583.jpg"),
        "person_desc": "Young man in dark blue T-shirt and beige khaki pants",
        "crops": [
            {"region": "upper", "crop_box": (0.28, 0.18, 0.72, 0.52), "ground_truth": "T-shirt", "notes": "Navy blue crewneck short-sleeve T-shirt"},
            {"region": "lower", "crop_box": (0.32, 0.50, 0.68, 0.88), "ground_truth": "Long Pants", "notes": "Beige/khaki full-length trousers/long pants"}
        ]
    },
    {
        "file": os.path.join(DOWNLOADS_DIR, "boy.jpg"),
        "person_desc": "Boy in striped/casual polo/t-shirt and dark pants/shorts",
        "crops": [
            {"region": "upper", "crop_box": (0.15, 0.10, 0.85, 0.55), "ground_truth": "T-shirt", "notes": "Casual young boy t-shirt/polo"},
            {"region": "lower", "crop_box": (0.20, 0.52, 0.80, 0.95), "ground_truth": "Jeans", "notes": "Casual pants/jeans"}
        ]
    },
    {
        "file": os.path.join(DOWNLOADS_DIR, "female.jpg"),
        "person_desc": "Woman in fashionable dress / top",
        "crops": [
            {"region": "upper", "crop_box": (0.20, 0.10, 0.80, 0.55), "ground_truth": "Dresses", "notes": "Floral one-piece dress / elegant dress top"},
            {"region": "lower", "crop_box": (0.20, 0.50, 0.80, 0.95), "ground_truth": "Dresses", "notes": "Bottom of dress / skirt"}
        ]
    },
    {
        "file": os.path.join(DOWNLOADS_DIR, "femalealone.jpg"),
        "person_desc": "Female pedestrian in casual jacket/sweater and jeans",
        "crops": [
            {"region": "upper", "crop_box": (0.20, 0.12, 0.80, 0.55), "ground_truth": "Jacket", "notes": "Outerwear zipper jacket/sweater"},
            {"region": "lower", "crop_box": (0.25, 0.52, 0.75, 0.95), "ground_truth": "Jeans", "notes": "Denim jeans"}
        ]
    },
    {
        "file": os.path.join(DOWNLOADS_DIR, "pexels.jpg"),
        "person_desc": "Pedestrian walking outdoors in street clothing",
        "crops": [
            {"region": "upper", "crop_box": (0.15, 0.15, 0.85, 0.55), "ground_truth": "T-shirt", "notes": "Casual summer top/t-shirt"},
            {"region": "lower", "crop_box": (0.20, 0.52, 0.80, 0.92), "ground_truth": "Shorts", "notes": "Casual shorts"}
        ]
    },
    {
        "file": os.path.join(DOWNLOADS_DIR, "crop.jpg"),
        "person_desc": "Close crop of pedestrian upper body",
        "crops": [
            {"region": "upper", "crop_box": (0.05, 0.05, 0.95, 0.95), "ground_truth": "Shirt", "notes": "Collared button shirt"}
        ]
    },
    {
        "file": os.path.join(DOWNLOADS_DIR, "check.jpg"),
        "person_desc": "Person wearing plaid / checkered shirt / clothing",
        "crops": [
            {"region": "upper", "crop_box": (0.10, 0.10, 0.90, 0.90), "ground_truth": "Shirt", "notes": "Patterned button shirt / top"}
        ]
    },
    {
        "file": os.path.join(DOWNLOADS_DIR, "check2.jpg"),
        "person_desc": "Person wearing layered jacket/coat",
        "crops": [
            {"region": "upper", "crop_box": (0.10, 0.10, 0.90, 0.90), "ground_truth": "Coat", "notes": "Outer winter coat / jacket"}
        ]
    }
]

# Extract video frames if available
walkroad_mp4 = os.path.join(DOWNLOADS_DIR, "WalkRoad.mp4")
if os.path.exists(walkroad_mp4):
    cap = cv2.VideoCapture(walkroad_mp4)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for f_idx in [15, 45, 90]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(f_idx, max(0, total_frames - 1)))
        ret, frame = cap.read()
        if ret:
            frame_path = os.path.join(MODEL_B_DIR, f"walkroad_frame_{f_idx}.jpg")
            cv2.imwrite(frame_path, frame)
            test_sample_specs.append({
                "file": frame_path,
                "person_desc": f"WalkRoad.mp4 video frame {f_idx} pedestrian",
                "crops": [
                    {"region": "upper", "crop_box": (0.30, 0.20, 0.65, 0.55), "ground_truth": "T-shirt", "notes": f"Pedestrian upper body frame {f_idx}"},
                    {"region": "lower", "crop_box": (0.30, 0.52, 0.65, 0.90), "ground_truth": "Long Pants", "notes": f"Pedestrian lower body frame {f_idx}"}
                ]
            })
    cap.release()

print(f"\nTotal test image/video specs loaded: {len(test_sample_specs)}")

# Execute Benchmark Runs
empirical_results = []
all_latencies_model_b = []
all_latencies_fashion_clip = []

sample_id_counter = 1

for spec in test_sample_specs:
    if not os.path.exists(spec["file"]):
        print(f"Warning: File {spec['file']} does not exist, skipping.")
        continue
    
    img_full = Image.open(spec["file"]).convert("RGB")
    W, H = img_full.size
    
    for c_spec in spec["crops"]:
        xmin = int(c_spec["crop_box"][0] * W)
        ymin = int(c_spec["crop_box"][1] * H)
        xmax = int(c_spec["crop_box"][2] * W)
        ymax = int(c_spec["crop_box"][3] * H)
        
        crop_img = img_full.crop((xmin, ymin, xmax, ymax))
        crop_save_path = os.path.join(MODEL_B_DIR, f"crop_sample_{sample_id_counter}_{c_spec['region']}.jpg")
        crop_img.save(crop_save_path)
        
        # Benchmark Model B Latency & Prediction
        t0 = time.perf_counter()
        top_model_b = run_model_b(crop_img)
        t_model_b = (time.perf_counter() - t0) * 1000.0
        all_latencies_model_b.append(t_model_b)
        
        # Benchmark Fashion-CLIP Latency & Prediction
        target_bank = upper_text_bank if c_spec["region"] == "upper" else lower_text_bank
        t1 = time.perf_counter()
        top_fashion_clip = run_fashion_clip(crop_img, target_bank)
        t_fashion_clip = (time.perf_counter() - t1) * 1000.0
        all_latencies_fashion_clip.append(t_fashion_clip)
        
        record = {
            "sample_id": sample_id_counter,
            "source_file": os.path.basename(spec["file"]),
            "person_desc": spec["person_desc"],
            "region": c_spec["region"],
            "ground_truth": c_spec["ground_truth"],
            "notes": c_spec["notes"],
            "crop_dimensions": f"{crop_img.width}x{crop_img.height}",
            "crop_saved_path": crop_save_path,
            "model_b": {
                "latency_ms": round(t_model_b, 3),
                "top1_label": top_model_b[0][0],
                "top1_prob": round(top_model_b[0][1], 8),
                "top2_label": top_model_b[1][0],
                "top2_prob": round(top_model_b[1][1], 8),
                "margin": round(top_model_b[0][1] - top_model_b[1][1], 8),
                "top5_full": [{"label": item[0], "prob": round(item[1], 8)} for item in top_model_b[:5]]
            },
            "fashion_clip": {
                "latency_ms": round(t_fashion_clip, 3),
                "top1_label": top_fashion_clip[0][0],
                "top1_prob": round(top_fashion_clip[0][1], 8),
                "top1_cos_sim": round(top_fashion_clip[0][2], 8),
                "top2_label": top_fashion_clip[1][0],
                "top2_prob": round(top_fashion_clip[1][1], 8),
                "margin": round(top_fashion_clip[0][1] - top_fashion_clip[1][1], 8),
                "top5_full": [{"label": item[0], "prob": round(item[1], 8), "cos_sim": round(item[2], 8)} for item in top_fashion_clip[:5]]
            }
        }
        empirical_results.append(record)
        sample_id_counter += 1

print("\n" + "=" * 70)
print("3. EMPIRICAL BENCHMARK RESULTS TABLE")
print("=" * 70)

for r in empirical_results:
    print(f"Sample #{r['sample_id']} | File: {r['source_file']} | Region: {r['region']} | GT: {r['ground_truth']}")
    print(f"  Model B (ViT): Top-1 = {r['model_b']['top1_label']} ({r['model_b']['top1_prob']:.6f}), Top-2 = {r['model_b']['top2_label']} ({r['model_b']['top2_prob']:.6f}), Margin = {r['model_b']['margin']:.6f}")
    print(f"  Top-5 Model B: " + ", ".join([f"{x['label']}: {x['prob']:.6f}" for x in r['model_b']['top5_full']]))
    print(f"  Fashion-CLIP : Top-1 = {r['fashion_clip']['top1_label']} ({r['fashion_clip']['top1_prob']:.6f}), Top-2 = {r['fashion_clip']['top2_label']} ({r['fashion_clip']['top2_prob']:.6f}), Margin = {r['fashion_clip']['margin']:.6f}")
    print(f"  Top-5 F-CLIP : " + ", ".join([f"{x['label']}: {x['prob']:.6f}" for x in r['fashion_clip']['top5_full']]))
    print("-" * 70)

avg_lat_b = np.mean(all_latencies_model_b)
avg_lat_fc = np.mean(all_latencies_fashion_clip)

summary = {
    "total_samples_evaluated": len(empirical_results),
    "model_b_mean_latency_ms": round(float(avg_lat_b), 2),
    "fashion_clip_mean_latency_ms": round(float(avg_lat_fc), 2),
    "results": empirical_results
}

with open(RESULTS_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"\nEmpirical results saved to: {RESULTS_JSON}")
print(f"Mean Latency (CPU) -> Model B: {avg_lat_b:.2f} ms | Fashion-CLIP: {avg_lat_fc:.2f} ms")
