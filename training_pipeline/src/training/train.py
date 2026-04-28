"""
Model Fine-tuning - Banking Complaint Classifier

Fine-tunes paraphrase-multilingual-MiniLM-L12-v2 for 6-class Vietnamese
complaint classification. Chosen over PhoBERT/XLM-R for its small footprint
(fits Render Free Tier at inference time).

Input:  data/processed/{train,val}.csv
Output: models/raw_model/ (PyTorch checkpoint + tokenizer + config)
"""

import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path

BASE_DIR = Path.cwd()
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OUTPUT_DIR = str(BASE_DIR / "models" / "raw_model")

# Hyperparameters (tuned on Colab T4):
#   batch=32: max that fits 16GB VRAM
#   epochs=5: F1 plateaus after epoch 4-5
#   lr=2e-5: standard for transformer fine-tuning
#   max_len=64: covers 99%+ of complaint texts
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 2e-5
MAX_LEN = 64

# Label mapping: saved into config.json, consumed by inference API
id2label = {
    0: "CARD_ISSUE", 1: "APP_LOGIN", 2: "TRANSACTION",
    3: "LOAN_SAVING", 4: "FRAUD_REPORT", 5: "OTHERS",
}
label2id = {v: k for k, v in id2label.items()}

# Load preprocessed data
train_df = pd.read_csv(str(BASE_DIR / "data" / "processed" / "train.csv"))
val_df = pd.read_csv(str(BASE_DIR / "data" / "processed" / "val.csv"))

# Convert to HF Dataset (rename label_id -> labels for Trainer compatibility)
train_dataset = Dataset.from_pandas(train_df[["text_clean", "label_id"]].rename(columns={"label_id": "labels"}))
val_dataset = Dataset.from_pandas(val_df[["text_clean", "label_id"]].rename(columns={"label_id": "labels"}))

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_function(examples):
    return tokenizer(examples["text_clean"], padding="max_length", truncation=True, max_length=MAX_LEN)


tokenized_train = train_dataset.map(tokenize_function, batched=True).remove_columns(["text_clean"])
tokenized_val = val_dataset.map(tokenize_function, batched=True).remove_columns(["text_clean"])

# Attach classification head (hidden_size -> 6 classes)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(id2label), id2label=id2label, label2id=label2id,
)


def compute_metrics(eval_pred):
    """Weighted F1 accounts for class imbalance in complaint categories."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,              # L2 regularization
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,    # Restore best checkpoint after training
    metric_for_best_model="f1",     # Select by F1, not loss
    save_total_limit=2,             # Limit disk usage
    logging_steps=50,
    report_to="none",               # No W&B/MLflow
    fp16=True,                      # Mixed precision for GPU speed
    dataloader_num_workers=2,
)

trainer = Trainer(
    model=model, args=training_args,
    train_dataset=tokenized_train, eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

print(f"[INFO] Fine-tuning {MODEL_NAME} | {len(tokenized_train)} train, {len(tokenized_val)} val")
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"[DONE] Model saved to {OUTPUT_DIR}")
