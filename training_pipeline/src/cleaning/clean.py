"""
Data Cleaning - Banking Complaint Classifier

Cleans Vietnamese text, encodes labels, and splits into train/val/test.

Input:  data/raw/banking_text.csv
Output: data/processed/{train,val,test}.csv (80/10/10 stratified split)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import re
from pathlib import Path

BASE_DIR = Path.cwd()
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "banking_text.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

# Must stay in sync with id2label in train.py and config.json
LABEL_MAP = {
    "CARD_ISSUE": 0, "APP_LOGIN": 1, "TRANSACTION": 2,
    "LOAN_SAVING": 3, "FRAUD_REPORT": 4, "OTHERS": 5,
}


def clean_text(text: str) -> str:
    """Normalize Vietnamese text: strip special chars, lowercase, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    # Regex preserves Vietnamese diacritics, alphanumerics, and basic punctuation
    text = re.sub(
        r'[^\w\s\d,?.!àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễ'
        r'ìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]',
        ' ', text
    )
    return " ".join(text.lower().split())


def main():
    print(f"[INFO] Reading: {RAW_DATA_PATH}")
    df = pd.read_csv(str(RAW_DATA_PATH))
    print(f"[INFO] Loaded {len(df)} rows")

    # Clean, filter short texts, and deduplicate to prevent data leakage
    df = df.dropna(subset=["text"])
    df["text_clean"] = df["text"].apply(clean_text)
    df = df[df["text_clean"].str.len() > 3]
    df = df.drop_duplicates(subset=["text_clean"])

    # Encode labels
    df["label_id"] = df["label"].map(LABEL_MAP)
    unmapped = df["label_id"].isna().sum()
    if unmapped > 0:
        print(f"[WARNING] Dropping {unmapped} rows with unknown labels")
        df = df.dropna(subset=["label_id"])
        df["label_id"] = df["label_id"].astype(int)

    print(f"[INFO] After cleaning: {len(df)} rows")

    # Stratified split preserves class distribution across splits
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label_id"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label_id"])

    print(f"[INFO] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "val.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)
    print(f"[DONE] Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
