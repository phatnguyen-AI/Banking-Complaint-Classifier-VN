"""
Production Inference API - Banking Complaint Classifier

Serves ONNX INT8 model via FastAPI for Vietnamese complaint classification.
Optimized for Render Free Tier (0.1 CPU, 512MB RAM).

Author: Phat Nguyen
"""

import os
import json
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tokenizers import Tokenizer
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

# Force single-threaded execution. Multi-threading on 0.1 CPU adds overhead.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["ORT_TENSORRT_FP16_ENABLE"] = "0"  # No GPU on this host

# Module-level state: loaded once at startup, reused across all requests.
model = None
tokenizer = None
labels = {}
MAX_LEN = 32  # Shorter than training (64) to reduce per-request memory


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, release on shutdown."""
    global model, tokenizer, labels

    # Load label mapping (id -> category name)
    try:
        with open("models/production/config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
            labels = {str(k): v for k, v in cfg.get("id2label", {}).items()}
    except (FileNotFoundError, json.JSONDecodeError):
        labels = {
            "0": "CARD_ISSUE", "1": "APP_LOGIN", "2": "TRANSACTION",
            "3": "LOAN_SAVING", "4": "FRAUD_REPORT", "5": "OTHERS"
        }

    # Load tokenizer (Rust-backed, avoids importing full transformers package)
    # and ONNX session with memory-optimized settings
    try:
        tokenizer = Tokenizer.from_file("models/production/tokenizer.json")
        tokenizer.enable_truncation(max_length=MAX_LEN)
        tokenizer.enable_padding(length=MAX_LEN)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1          # Match single-core allocation
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.enable_cpu_mem_arena = False       # Trade latency for predictable RAM

        model = ort.InferenceSession(
            "models/production/model_main.onnx", opts,
            providers=["CPUExecutionProvider"]
        )
        print(">>> Model loaded successfully")
    except Exception as e:
        print(f">>> Error loading model: {e}")

    yield
    model = None
    tokenizer = None


app = FastAPI(title="Banking Complaint Classifier", lifespan=lifespan)

# Wildcard CORS: API is public and stateless, no auth to protect.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    text: str


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax (subtract max to prevent overflow)."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


@app.get("/health")
def health():
    """Health probe for Render container checks and frontend status polling."""
    return {"status": "ok"} if model else {"status": "error"}


@app.post("/predict")
def predict(item: PredictionRequest):
    """Classify Vietnamese complaint text. Returns {label, score}."""
    if not model:
        raise HTTPException(status_code=503, detail="Model not ready")

    text = item.text.strip() or "empty"

    # Tokenize (pre-configured with truncation and padding)
    enc = tokenizer.encode(text)
    inputs = {
        "input_ids": np.array([enc.ids], dtype=np.int64),
        "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
        "token_type_ids": np.array([enc.type_ids], dtype=np.int64),
    }

    # Run inference and return top prediction
    try:
        logits = model.run(None, inputs)[0][0]
        probs = softmax(logits)
        pred_idx = np.argmax(probs)
        return {
            "label": labels.get(str(pred_idx), "Unknown"),
            "score": round(float(probs[pred_idx]), 4),
        }
    except Exception:
        raise HTTPException(status_code=500, detail="Inference failed")