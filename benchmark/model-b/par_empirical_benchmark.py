import os
import sys
import site
import json
import time
import numpy as np
from PIL import Image

sys.path.insert(0, site.getusersitepackages())
import onnxruntime as ort
import cv2

# Set random seeds
np.random.seed(42)

# Paths
PULC_PAR_PATH = r"d:\Mann\Project\VisionAttributeAI\models\par\pulc_person_attribute.onnx"
FASHION_CLIP_VISION_PATH = r"d:\Mann\Project\VisionAttributeAI\models\par\fashion_clip_vision.onnx"
FASHION_CLIP_TEXT_PATH = r"d:\Mann\Project\VisionAttributeAI\models\par\fashion_clip_text.onnx"
DOWNLOADS_DIR = r"C:\Users\Admin\Downloads"
OUTPUT_JSON = r"d:\Mann\Project\VisionAttributeAI\benchmark\model-b\par_empirical_results.json"

print("=" * 75)
print("1. LOADING REAL PAR MODEL (PA-100K PULC 26-Attribute Classifier)")
print("=" * 75)

t0_load = time.perf_counter()
ort_opts = ort.SessionOptions()
ort_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
ort_opts.intra_op_num_threads = 4

par_session = ort.InferenceSession(PULC_PAR_PATH, ort_opts)
load_time_par = (time.perf_counter() - t0_load) * 1000.0

print(f"PAR Model loaded in {load_time_par:.2f} ms")
print(f"Inputs: {[i.name for i in par_session.get_inputs()]} shape: {[i.shape for i in par_session.get_inputs()]}")
print(f"Outputs: {[o.name for o in par_session.get_outputs()]} shape: {[o.shape for o in par_session.get_outputs()]}")

PA100K_ATTRIBUTES = [
    "Hat",                # 0
    "Glasses",            # 1
    "ShortSleeve",        # 2
    "LongSleeve",         # 3
    "UpperStripe",        # 4
    "UpperLogo",          # 5
    "UpperPlaid",         # 6
    "UpperSplice",        # 7
    "LowerStripe",        # 8
    "LowerPattern",       # 9
    "LongCoat",           # 10
    "Trousers",           # 11
    "Shorts",             # 12
    "Skirt&Dress",        # 13
    "Boots",              # 14
    "HandBag",            # 15
    "ShoulderBag",        # 16
    "Backpack",           # 17
    "HoldObjectsInFront", # 18
    "AgeLess18",          # 19
    "Age18-60",           # 20
    "AgeOver60",          # 21
    "Female",             # 22
    "Front",              # 23
    "Side",               # 24
    "Back"                # 25
]

print(f"Supported PA-100K Attributes ({len(PA100K_ATTRIBUTES)} total):")
for idx, name in enumerate(PA100K_ATTRIBUTES):
    print(f"  [{idx:02d}] {name}")

print("\n" + "=" * 75)
print("2. LOADING FASHION-CLIP ONNX")
print("=" * 75)

t0_fclip = time.perf_counter()
vision_session = ort.InferenceSession(FASHION_CLIP_VISION_PATH, ort_opts)
text_session = ort.InferenceSession(FASHION_CLIP_TEXT_PATH, ort_opts)
load_time_fclip = (time.perf_counter() - t0_fclip) * 1000.0
print(f"Fashion-CLIP loaded in {load_time_fclip:.2f} ms")

from transformers import CLIPTokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# Define target prompts for Fashion-CLIP
sex_categories = {
    "Male": ["a man", "a male person", "a young man", "an adult man", "a guy"],
    "Female": ["a woman", "a female person", "a young woman", "an adult woman", "a girl", "a lady"]
}

upper_categories = {
    "ShortSleeve / T-Shirt": ["short sleeve t-shirt", "cotton tee", "half sleeve t-shirt", "crewneck t-shirt"],
    "LongSleeve / Shirt": ["button-up collared shirt", "formal dress shirt", "long sleeve shirt"],
    "Jacket / LongCoat": ["winter coat", "zipper jacket", "outerwear jacket", "long overcoat"]
}

lower_categories = {
    "Trousers / Pants": ["formal trousers", "pants", "dress slacks", "jeans", "denim pants"],
    "Shorts": ["shorts", "short pants"],
    "Skirt&Dress": ["skirt", "dress", "one-piece dress"]
}

def encode_text_prompts(cat_dict):
    templates = ["a photo of a {0}", "a person wearing a {0}", "a {0}"]
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

sex_bank = encode_text_prompts(sex_categories)
upper_bank = encode_text_prompts(upper_categories)
lower_bank = encode_text_prompts(lower_categories)

def preprocess_for_par(img_pil):
    # PAR expects [1, 3, 256, 192], RGB, normalized with ImageNet mean/std
    img = img_pil.convert("RGB").resize((192, 256), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1)) # CHW
    arr = np.expand_dims(arr, 0) # 1, 3, 256, 192
    return arr

def preprocess_for_clip(img_pil):
    img = img_pil.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, 0)
    return arr

def run_par_inference(img_pil):
    arr = preprocess_for_par(img_pil)
    out = par_session.run(None, {"x": arr})[0][0]
    # out is sigmoid probabilities for each of the 26 attributes
    res = {}
    for idx, name in enumerate(PA100K_ATTRIBUTES):
        res[name] = float(out[idx])
    return res

def run_clip_classification(img_pil, bank):
    arr = preprocess_for_clip(img_pil)
    vision_out = vision_session.run(None, {"pixel_values": arr})[0][0]
    vision_out = vision_out / (np.linalg.norm(vision_out) + 1e-12)
    
    sims = []
    labels = list(bank.keys())
    for l in labels:
        sims.append(float(np.dot(vision_out, bank[l])))
    sims = np.array(sims)
    exp_sims = np.exp((sims - np.max(sims)) / 0.07)
    probs = exp_sims / np.sum(exp_sims)
    
    res = {labels[i]: float(probs[i]) for i in range(len(labels))}
    return res

# Define test subjects
test_subjects = [
    {
        "id": "SUBJ_01",
        "file": os.path.join(DOWNLOADS_DIR, "young-man-standing-in-middle-of-road-on-sunny-day-JMPF00583.jpg"),
        "name": "Young Man on Road",
        "full_box": (0.28, 0.15, 0.72, 0.92),
        "upper_box": (0.28, 0.18, 0.72, 0.52),
        "lower_box": (0.32, 0.50, 0.68, 0.88),
        "gt": {
            "Sex": "Male (Female=0)",
            "Orientation": "Front",
            "Upper_Sleeve": "ShortSleeve",
            "Lower": "Trousers",
            "Hat": "No",
            "Glasses": "No",
            "Backpack": "No"
        }
    },
    {
        "id": "SUBJ_02",
        "file": os.path.join(DOWNLOADS_DIR, "boy.jpg"),
        "name": "Young Boy Standing",
        "full_box": (0.15, 0.08, 0.85, 0.95),
        "upper_box": (0.15, 0.10, 0.85, 0.55),
        "lower_box": (0.20, 0.52, 0.80, 0.95),
        "gt": {
            "Sex": "Male (Female=0)",
            "Orientation": "Front",
            "Upper_Sleeve": "ShortSleeve",
            "Lower": "Trousers",
            "Hat": "No",
            "Glasses": "No",
            "Backpack": "No"
        }
    },
    {
        "id": "SUBJ_03",
        "file": os.path.join(DOWNLOADS_DIR, "female.jpg"),
        "name": "Female in Dress",
        "full_box": (0.18, 0.05, 0.82, 0.96),
        "upper_box": (0.20, 0.10, 0.80, 0.55),
        "lower_box": (0.20, 0.50, 0.80, 0.95),
        "gt": {
            "Sex": "Female (Female=1)",
            "Orientation": "Front",
            "Upper_Sleeve": "ShortSleeve",
            "Lower": "Skirt&Dress",
            "Hat": "No",
            "Glasses": "No",
            "Backpack": "No"
        }
    },
    {
        "id": "SUBJ_04",
        "file": os.path.join(DOWNLOADS_DIR, "femalealone.jpg"),
        "name": "Female Pedestrian Walking",
        "full_box": (0.18, 0.08, 0.82, 0.96),
        "upper_box": (0.20, 0.12, 0.80, 0.55),
        "lower_box": (0.25, 0.52, 0.75, 0.95),
        "gt": {
            "Sex": "Female (Female=1)",
            "Orientation": "Front/Side",
            "Upper_Sleeve": "LongSleeve",
            "Lower": "Trousers",
            "Hat": "No",
            "Glasses": "No",
            "Backpack": "No"
        }
    },
    {
        "id": "SUBJ_05",
        "file": os.path.join(DOWNLOADS_DIR, "pexels.jpg"),
        "name": "Outdoor Street Pedestrian",
        "full_box": (0.15, 0.10, 0.85, 0.95),
        "upper_box": (0.15, 0.15, 0.85, 0.55),
        "lower_box": (0.20, 0.52, 0.80, 0.92),
        "gt": {
            "Sex": "Male (Female=0)",
            "Orientation": "Side",
            "Upper_Sleeve": "ShortSleeve",
            "Lower": "Shorts",
            "Hat": "No",
            "Glasses": "No",
            "Backpack": "No"
        }
    },
    {
        "id": "SUBJ_06",
        "file": os.path.join(DOWNLOADS_DIR, "crop.jpg"),
        "name": "Waist-up Pedestrian Crop",
        "full_box": (0.05, 0.05, 0.95, 0.95),
        "upper_box": (0.05, 0.05, 0.95, 0.80),
        "lower_box": (0.10, 0.60, 0.90, 0.95),
        "gt": {
            "Sex": "Male (Female=0)",
            "Orientation": "Front",
            "Upper_Sleeve": "LongSleeve",
            "Lower": "Unknown (Waist up)",
            "Hat": "No",
            "Glasses": "No",
            "Backpack": "No"
        }
    },
    {
        "id": "SUBJ_07",
        "file": os.path.join(DOWNLOADS_DIR, "check.jpg"),
        "name": "Person in Plaid Top",
        "full_box": (0.05, 0.05, 0.95, 0.95),
        "upper_box": (0.05, 0.05, 0.95, 0.85),
        "lower_box": (0.10, 0.70, 0.90, 0.95),
        "gt": {
            "Sex": "Male (Female=0)",
            "Orientation": "Front",
            "Upper_Sleeve": "LongSleeve",
            "Lower": "Trousers",
            "Hat": "No",
            "Glasses": "No",
            "Backpack": "No"
        }
    },
    {
        "id": "SUBJ_08",
        "file": os.path.join(DOWNLOADS_DIR, "check2.jpg"),
        "name": "Person in Coat",
        "full_box": (0.05, 0.05, 0.95, 0.95),
        "upper_box": (0.05, 0.05, 0.95, 0.85),
        "lower_box": (0.10, 0.70, 0.90, 0.95),
        "gt": {
            "Sex": "Male (Female=0)",
            "Orientation": "Front",
            "Upper_Sleeve": "LongSleeve",
            "Lower": "Trousers",
            "Hat": "No",
            "Glasses": "No",
            "Backpack": "No"
        }
    }
]

# Add video frames from WalkRoad.mp4
walkroad_mp4 = os.path.join(DOWNLOADS_DIR, "WalkRoad.mp4")
if os.path.exists(walkroad_mp4):
    cap = cv2.VideoCapture(walkroad_mp4)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for f_idx in [15, 45, 90]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(f_idx, max(0, total_frames - 1)))
        ret, frame = cap.read()
        if ret:
            frame_path = os.path.join(r"d:\Mann\Project\VisionAttributeAI\benchmark\model-b", f"walkroad_subj_f{f_idx}.jpg")
            cv2.imwrite(frame_path, frame)
            test_subjects.append({
                "id": f"SUBJ_VID_{f_idx}",
                "file": frame_path,
                "name": f"WalkRoad.mp4 Pedestrian Frame {f_idx}",
                "full_box": (0.28, 0.15, 0.68, 0.92),
                "upper_box": (0.28, 0.20, 0.65, 0.55),
                "lower_box": (0.30, 0.52, 0.65, 0.90),
                "gt": {
                    "Sex": "Male (Female=0)",
                    "Orientation": "Back",
                    "Upper_Sleeve": "ShortSleeve",
                    "Lower": "Trousers",
                    "Hat": "No",
                    "Glasses": "No",
                    "Backpack": "No"
                }
            })
    cap.release()

print(f"\nTotal test subjects: {len(test_subjects)}")

# Execution & Timing
par_latencies = []
clip_latencies = []
empirical_records = []

for subj in test_subjects:
    if not os.path.exists(subj["file"]):
        continue
    img_full = Image.open(subj["file"]).convert("RGB")
    W, H = img_full.size
    
    # 1. Full Person Crop
    fb = subj["full_box"]
    full_crop = img_full.crop((int(fb[0]*W), int(fb[1]*H), int(fb[2]*W), int(fb[3]*H)))
    
    # 2. Upper ROI Crop
    ub = subj["upper_box"]
    upper_crop = img_full.crop((int(ub[0]*W), int(ub[1]*H), int(ub[2]*W), int(ub[3]*H)))
    
    # 3. Lower ROI Crop
    lb = subj["lower_box"]
    lower_crop = img_full.crop((int(lb[0]*W), int(lb[1]*H), int(lb[2]*W), int(lb[3]*H)))
    
    # --- Measure PAR on Full Person Crop ---
    t0 = time.perf_counter()
    par_full = run_par_inference(full_crop)
    lat_par = (time.perf_counter() - t0) * 1000.0
    par_latencies.append(lat_par)
    
    # --- Measure PAR on Upper Crop & Lower Crop (to test Region vs Full Person) ---
    par_upper = run_par_inference(upper_crop)
    par_lower = run_par_inference(lower_crop)
    
    # --- Measure Fashion-CLIP on Full / Crops ---
    t1 = time.perf_counter()
    clip_sex = run_clip_classification(full_crop, sex_bank)
    clip_upper = run_clip_classification(upper_crop, upper_bank)
    clip_lower = run_clip_classification(lower_crop, lower_bank)
    lat_clip = (time.perf_counter() - t1) * 1000.0
    clip_latencies.append(lat_clip)
    
    rec = {
        "subject_id": subj["id"],
        "subject_name": subj["name"],
        "source_file": os.path.basename(subj["file"]),
        "ground_truth": subj["gt"],
        "par_full_person": {
            "latency_ms": round(lat_par, 3),
            "attributes": {k: round(v, 6) for k, v in par_full.items()}
        },
        "par_upper_crop_experiment": {
            "attributes": {k: round(v, 6) for k, v in par_upper.items()}
        },
        "par_lower_crop_experiment": {
            "attributes": {k: round(v, 6) for k, v in par_lower.items()}
        },
        "fashion_clip": {
            "latency_ms": round(lat_clip, 3),
            "sex": {k: round(v, 6) for k, v in clip_sex.items()},
            "upper": {k: round(v, 6) for k, v in clip_upper.items()},
            "lower": {k: round(v, 6) for k, v in clip_lower.items()}
        }
    }
    empirical_records.append(rec)

# Calculate latency stats
par_lat_arr = np.array(par_latencies)
clip_lat_arr = np.array(clip_latencies)

summary_stats = {
    "par_mean_ms": round(float(np.mean(par_lat_arr)), 2),
    "par_p50_ms": round(float(np.percentile(par_lat_arr, 50)), 2),
    "par_p95_ms": round(float(np.percentile(par_lat_arr, 95)), 2),
    "par_min_ms": round(float(np.min(par_lat_arr)), 2),
    "par_max_ms": round(float(np.max(par_lat_arr)), 2),
    "clip_mean_ms": round(float(np.mean(clip_lat_arr)), 2),
    "clip_p50_ms": round(float(np.percentile(clip_lat_arr, 50)), 2),
    "clip_p95_ms": round(float(np.percentile(clip_lat_arr, 95)), 2),
    "total_subjects_tested": len(empirical_records),
    "results": empirical_records
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(summary_stats, f, indent=2)

print("\n" + "=" * 75)
print("3. EMPIRICAL PAR & FASHION-CLIP BENCHMARK SUMMARY")
print("=" * 75)
for r in empirical_records:
    print(f"Subject: {r['subject_id']} ({r['subject_name']}) | File: {r['source_file']}")
    par = r['par_full_person']['attributes']
    clip = r['fashion_clip']
    print(f"  [PAR Full]  Female: {par['Female']:.6f} | ShortSleeve: {par['ShortSleeve']:.6f} | LongSleeve: {par['LongSleeve']:.6f} | Trousers: {par['Trousers']:.6f} | Shorts: {par['Shorts']:.6f} | Skirt&Dress: {par['Skirt&Dress']:.6f}")
    print(f"  [PAR Full]  Front: {par['Front']:.6f} | Side: {par['Side']:.6f} | Back: {par['Back']:.6f} | Hat: {par['Hat']:.6f} | Glasses: {par['Glasses']:.6f} | Boots: {par['Boots']:.6f}")
    print(f"  [F-CLIP]    Sex: Male={clip['sex']['Male']:.6f}, Female={clip['sex']['Female']:.6f}")
    print(f"  [F-CLIP]    Upper: {list(clip['upper'].items())[0][0]}={list(clip['upper'].items())[0][1]:.6f}, {list(clip['upper'].items())[1][0]}={list(clip['upper'].items())[1][1]:.6f}")
    print(f"  [F-CLIP]    Lower: {list(clip['lower'].items())[0][0]}={list(clip['lower'].items())[0][1]:.6f}, {list(clip['lower'].items())[1][0]}={list(clip['lower'].items())[1][1]:.6f}")
    print("-" * 75)

print(f"\nLatency Statistics (CPU 4 Threads):")
print(f"  PAR Model   : Mean = {summary_stats['par_mean_ms']} ms | P50 = {summary_stats['par_p50_ms']} ms | P95 = {summary_stats['par_p95_ms']} ms")
print(f"  Fashion-CLIP: Mean = {summary_stats['clip_mean_ms']} ms | P50 = {summary_stats['clip_p50_ms']} ms | P95 = {summary_stats['clip_p95_ms']} ms")
print(f"\nSaved detailed JSON to: {OUTPUT_JSON}")
