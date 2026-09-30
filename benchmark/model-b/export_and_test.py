import site
import sys
user_site = site.getusersitepackages()
if user_site not in sys.path:
    sys.path.insert(0, user_site)

import os
import json
import time
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image

def main():
    model_dir = r"d:\Mann\Project\VisionAttributeAI\benchmark\model-b"
    onnx_path = os.path.join(model_dir, "model.onnx")
    
    print("=" * 65)
    print("TASK 2 — VERIFYING MODEL FILES & CONFIG DIRECTLY FROM LOCAL DISK")
    print("=" * 65)
    
    config_path = os.path.join(model_dir, "config.json")
    preproc_path = os.path.join(model_dir, "preprocessor_config.json")
    safetensors_path = os.path.join(model_dir, "model.safetensors")
    
    with open(config_path, "r") as f:
        config_data = json.load(f)
        
    with open(preproc_path, "r") as f:
        preproc_data = json.load(f)
        
    print(f"Model Directory:         {model_dir}")
    print(f"config.json Path:        {config_path}")
    print(f"preprocessor_config:     {preproc_path}")
    print(f"model.safetensors Path:  {safetensors_path}")
    print(f"model.safetensors Size:  {os.path.getsize(safetensors_path):,} bytes")
    
    print("\n--- ACTUAL VALUES READ FROM config.json ---")
    print(f"num_labels:              {len(config_data['id2label'])}")
    print(f"model_type:              {config_data.get('model_type')}")
    print(f"architectures:           {config_data.get('architectures')}")
    print(f"image_size:              {config_data.get('image_size')}")
    print(f"hidden_size:             {config_data.get('hidden_size')}")
    print(f"num_hidden_layers:       {config_data.get('num_hidden_layers')}")
    print(f"num_attention_heads:     {config_data.get('num_attention_heads')}")
    print(f"patch_size:              {config_data.get('patch_size')}")
    
    print("\n--- ACTUAL id2label MAPPING READ FROM config.json ---")
    for k in sorted(config_data["id2label"].keys(), key=lambda x: int(x)):
        print(f"  {k:>2}: {config_data['id2label'][k]}")
        
    print("\n--- ACTUAL VALUES READ FROM preprocessor_config.json ---")
    for k, v in preproc_data.items():
        print(f"  {k}: {v}")
        
    print("\n" + "=" * 65)
    print("EXPORTING MODEL TO ONNX FORMAT FOR EMBEDDED C# INFERENCE")
    print("=" * 65)
    
    t0 = time.time()
    model = ViTForImageClassification.from_pretrained(
        model_dir,
        config=config_path,
        local_files_only=True
    )
    model.eval()
    print(f"PyTorch model loaded in {(time.time() - t0)*1000:.1f}ms")
    
    dummy_input = torch.randn(1, 3, 224, 224)
    print(f"Exporting to: {onnx_path}...")
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch_size"}, "logits": {0: "batch_size"}}
    )
    print(f"Export completed! ONNX file size: {os.path.getsize(onnx_path):,} bytes")

if __name__ == "__main__":
    main()
