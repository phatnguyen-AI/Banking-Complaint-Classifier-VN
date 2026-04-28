"""
ONNX Export + INT8 Quantization - Banking Complaint Classifier

Converts PyTorch checkpoint to ONNX and applies dynamic INT8 quantization.
Reduces model size ~75% with <1% accuracy loss, enabling deployment on
Render Free Tier (0.1 CPU, 512MB RAM).

Input:  models/raw_model/
Output: models/onnx_int8/model_int8.onnx + tokenizer files
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path

BASE_DIR = Path.cwd()
INPUT_PATH = os.path.join(BASE_DIR, "models/raw_model")
TEMP_PATH = os.path.join(BASE_DIR, "models/onnx_temp")      # Intermediate float32
OUTPUT_PATH = os.path.join(BASE_DIR, "models/onnx_int8")     # Final quantized

os.makedirs(TEMP_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Load model on CPU (export is CPU-only regardless of training device)
print(f"[INFO] Loading model from {INPUT_PATH}")
try:
    model = AutoModelForSequenceClassification.from_pretrained(INPUT_PATH).cpu()
    tokenizer = AutoTokenizer.from_pretrained(INPUT_PATH)
    model.eval()  # Disable dropout for deterministic export
except Exception as e:
    print(f"[ERROR] {e}")
    exit(1)

# Dummy input for ONNX graph tracing (text content is irrelevant)
inputs = tokenizer("Sample text for tracing", return_tensors="pt")
onnx_float_file = os.path.join(TEMP_PATH, "model.onnx")

print(f"[INFO] Exporting ONNX float32: {onnx_float_file}")
torch.onnx.export(
    model,
    (inputs["input_ids"], inputs["attention_mask"],
     inputs.get("token_type_ids", torch.zeros_like(inputs["input_ids"]))),
    onnx_float_file,
    input_names=["input_ids", "attention_mask", "token_type_ids"],
    output_names=["logits"],
    # Dynamic axes: accept variable batch size and sequence length at runtime
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "token_type_ids": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"},
    },
    opset_version=14,  # Minimum opset supporting all MiniLM operators
)

# Dynamic quantization: weights -> uint8, activations quantized at runtime
# optimize_model=False: let ONNX Runtime handle graph optimization at session creation
final_model_file = os.path.join(OUTPUT_PATH, "model_int8.onnx")
print(f"[INFO] Quantizing to INT8: {final_model_file}")

try:
    quantize_dynamic(
        model_input=onnx_float_file,
        model_output=final_model_file,
        weight_type=QuantType.QUInt8,
        optimize_model=False,
    )
except Exception as e:
    print(f"[ERROR] {e}")
    exit(1)

tokenizer.save_pretrained(OUTPUT_PATH)
print(f"[DONE] Quantized model saved to {OUTPUT_PATH}")
