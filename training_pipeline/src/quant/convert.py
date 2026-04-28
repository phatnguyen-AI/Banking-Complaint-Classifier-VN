"""
Production Export - Banking Complaint Classifier

Separates model weights into external binary and copies config files
into a self-contained production directory ready for Docker deployment.

Input:  models/onnx_int8/model_int8.onnx
Output: models/production/{model_main.onnx, weights.bin, tokenizer.json, config.json}
"""

import onnx
import os
import shutil
from pathlib import Path

BASE_DIR = Path.cwd()
INPUT_MODEL_PATH = os.path.join(BASE_DIR, "models/onnx_int8/model_int8.onnx")
SOURCE_DIR = os.path.join(BASE_DIR, "models/onnx_int8")
OUTPUT_DIR = os.path.join(BASE_DIR, "models/production")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Separate weights into external file for better Docker layer caching
# (code changes don't invalidate the large weight file layer)
print(f"[INFO] Loading: {INPUT_MODEL_PATH}")
model = onnx.load(INPUT_MODEL_PATH)

onnx.save_model(
    model,
    os.path.join(OUTPUT_DIR, "model_main.onnx"),
    save_as_external_data=True,
    all_tensors_to_one_file=False,
    location="weights.bin",
    size_threshold=1024,  # Tensors > 1KB go to external file
)

# Copy tokenizer and config; fallback to raw_model if not in onnx_int8
for fname in ["tokenizer.json", "config.json"]:
    src = os.path.join(SOURCE_DIR, fname)
    dst = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(src):
        shutil.copy2(src, dst)
    else:
        fallback = os.path.join(BASE_DIR, "models/raw_model", fname)
        if os.path.exists(fallback):
            shutil.copy2(fallback, dst)
        else:
            print(f"[WARNING] {fname} not found")

print(f"[DONE] Production artifacts: {OUTPUT_DIR}")
