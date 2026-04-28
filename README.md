# Banking Complaint Auto-Router — Vietnamese NLP Classification System

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-INT8-7B68EE?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Live Demo](https://img.shields.io/badge/Live_Demo-Click_Here-FF6B6B?style=for-the-badge)](https://llmfontend.vercel.app/)

---

Vietnamese banks handle thousands of customer complaints daily — from blocked cards to fraudulent transactions — across mobile apps, email, hotlines, and branches. Manually routing each complaint to the correct department is slow, error-prone, and expensive. Misrouted tickets inflate resolution time, erode customer satisfaction, and expose the bank to compliance risk.

**This project solves that.** It is an end-to-end NLP system that automatically classifies Vietnamese banking complaints into 6 actionable categories and routes them to the appropriate department — in under 100 milliseconds, at zero hosting cost.

**[Try the Live Demo](https://llmfontend.vercel.app/)**

---

## Business Impact & Value Proposition

This is not a toy classifier — it is a **production-grade pilot** designed to demonstrate real operational value:

- **Eliminate manual routing bottleneck** — Complaints are classified and routed instantly, removing the need for human triage on first contact.
- **Faster complaint resolution** — Correct department receives the ticket immediately, reducing average resolution time from hours to minutes.
- **Lower operational cost** — Automates a repetitive, high-volume task that currently requires dedicated staff.
- **Improved customer satisfaction** — Faster resolution directly correlates with higher NPS scores and lower churn.
- **Zero-cost deployment** — The entire system runs on free-tier infrastructure (Render + Vercel), making it viable as an MVP or pilot without budget approval.
- **Scalable foundation** — Architecture is designed to scale from pilot to production by upgrading infrastructure, not rewriting code.

---

## System Architecture

```
┌─────────────┐     HTTPS      ┌──────────────────┐     Inference     ┌─────────────────┐
│             │   ──────────►  │                  │   ────────────►   │                 │
│  Customer   │                │   FastAPI Server  │                   │  ONNX Runtime   │
│  (Browser)  │                │   (Render)        │                   │  INT8 Model     │
│             │   ◄──────────  │                  │   ◄────────────   │                 │
└─────────────┘    JSON Result └──────────────────┘    Classification  └─────────────────┘
                                        │
       ┌────────────────────────────────┘
       ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Department Routing Output                      │
├──────────┬──────────┬─────────────┬────────────┬────────────────┤
│ CARD     │ APP      │ TRANSACTION │ LOAN &     │ FRAUD          │
│ ISSUE    │ LOGIN    │             │ SAVING     │ REPORT         │
└──────────┴──────────┴─────────────┴────────────┴────────────────┘
```

**Frontend** → Vanilla HTML/CSS/JS + Bootstrap 5 hosted on **Vercel** (global CDN)  
**Backend** → FastAPI + ONNX Runtime hosted on **Render** (Docker, auto-deploy on push)  
**Model** → Fine-tuned Sentence Transformer quantized to INT8 for minimal resource usage

---

## Key Technical Decisions

Every technical choice in this project was driven by a **business constraint**, not personal preference:

| Decision | Choice | Business Rationale |
|:---|:---|:---|
| **Base Model** | `paraphrase-multilingual-MiniLM-L12-v2` | Supports Vietnamese natively with a fraction of the size of PhoBERT or XLM-R. Critical for deploying on free-tier infrastructure without sacrificing multilingual capability. |
| **Quantization** | ONNX + Dynamic INT8 | Reduced model size by **~75%** while maintaining ~87% accuracy. This is the tradeoff that makes zero-cost deployment possible. |
| **Inference Runtime** | ONNX Runtime (single-thread) | Replaces PyTorch for serving — 3-5x faster inference, dramatically lower memory footprint. Enables <100ms latency on 0.1 CPU. |
| **Backend Framework** | FastAPI | Async-native, auto-generated API docs, minimal overhead. Ideal for ML serving where every MB of RAM counts. |
| **Deployment** | Render Free Tier + Vercel | $0/month hosting with auto-deploy CI/CD. Proves the system is production-viable before any infrastructure investment. |
| **Frontend** | Vanilla JS + Bootstrap 5 | No build step, instant load, CDN-delivered. Keeps the frontend as a thin client — all intelligence lives server-side. |
| **Training Pipeline** | Modular Python (clean → train → quantize → export) | Fully reproducible. Any team member can retrain on new data without understanding the full codebase. |

---

## Model Performance

| Metric | Value |
|:---|:---|
| **Accuracy** | ~87% |
| **F1-Score** | ~87% |
| **Inference Latency** | <100ms per request |
| **Model Size (post-quantization)** | ~129MB |
| **Memory per Inference** | <100MB |
| **Hardware Requirement** | 0.1 CPU, 512MB RAM |

### Fine-tuned Model (Float32)

<img width="523" height="271" alt="Model performance — Float32" src="https://github.com/user-attachments/assets/56637ef3-7769-4d91-ad08-b0894ba44d67" />

### After ONNX + INT8 Quantization

<img width="531" height="251" alt="Model performance — ONNX INT8" src="https://github.com/user-attachments/assets/45c56e7a-dcea-458b-8652-cee7e81fc012" />

> **The Tradeoff:** Quantization reduced model size by ~75% with negligible accuracy loss. This single decision enabled deployment on free-tier infrastructure (0.1 CPU, 512MB RAM) — turning a model that required a paid GPU server into one that runs at zero cost.

---

## Classification Categories

| Category | Description | Example Complaint | Routed To |
|:---|:---|:---|:---|
| `CARD_ISSUE` | Card-related problems | *"My debit card was declined at the store even though I have sufficient balance"* | Card Operations |
| `APP_LOGIN` | Mobile/web app access issues | *"I can't log in to the banking app after updating my phone"* | Digital Banking Support |
| `TRANSACTION` | Payment & transfer issues | *"I transferred money 3 hours ago but the recipient hasn't received it"* | Transaction Processing |
| `LOAN_SAVING` | Loan & savings account issues | *"My savings interest rate doesn't match what was advertised"* | Lending & Deposits |
| `FRAUD_REPORT` | Suspected fraud or unauthorized activity | *"I see two transactions I didn't make on my credit card statement"* | Fraud Investigation |
| `OTHERS` | General inquiries & uncategorized | *"What are your branch operating hours during holidays?"* | General Customer Service |

---

## Project Structure

```
Banking-Complaint-Classifier-VN/
├── backend/                    # Production API server
│   ├── api.py                 # FastAPI app — model loading, inference, health check
│   ├── Dockerfile             # Container config optimized for Render Free Tier
│   └── requirements.txt      # Runtime dependencies
├── frontend/                   # Client-side UI
│   ├── index.html             # Main interface — input form, results display
│   ├── script.js              # API integration, health polling, error handling
│   └── styles.css             # Responsive styling with Bootstrap 5
├── data/                       # Dataset
│   ├── raw/                   # Original banking complaint corpus (~4,640 samples)
│   └── processed/             # Train/Val/Test splits (80/10/10)
├── models/                     # ML artifacts
│   ├── production/            # Deployment-ready: ONNX model + tokenizer + config
│   ├── raw_model/             # Fine-tuned PyTorch checkpoint
│   └── onnx_int8/             # Quantized ONNX model
├── training_pipeline/          # Reproducible training workflow
│   ├── notebooks/             # EDA & analysis notebooks
│   ├── requirements.txt       # Training dependencies
│   └── src/
│       ├── cleaning/          # Data preprocessing & Vietnamese text normalization
│       ├── training/          # Fine-tuning with Hugging Face Transformers
│       └── quant/             # ONNX conversion & INT8 quantization
└── README.md
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/phatnguyen-AI/Banking-Complaint-Classifier-VN.git
cd Banking-Complaint-Classifier-VN

# Backend dependencies
cd backend
pip install -r requirements.txt
```

### 2. Run the API Server

```bash
cd backend
uvicorn api:app --host 0.0.0.0 --port 8000
```

### 3. Test the API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Thẻ của tôi bị lỗi không thể thanh toán được"}'
```

Response:
```json
{
  "label": "CARD_ISSUE",
  "score": 0.95
}
```

### 4. Docker Deployment

```bash
docker build -t banking-complaint-classifier .
docker run -p 8000:8000 banking-complaint-classifier
```

---

## Training Pipeline

The training workflow is modular and fully reproducible — each step is an independent script that reads from and writes to well-defined paths:

```
Data Cleaning ──► Fine-tuning ──► Quantization ──► Production Export
```

```bash
cd training_pipeline/src

# Step 1: Clean & normalize Vietnamese text, split into train/val/test
python -m cleaning.clean

# Step 2: Fine-tune paraphrase-multilingual-MiniLM-L12-v2
python -m training.train

# Step 3: Convert to ONNX + apply INT8 dynamic quantization
python -m quant.onnx_int8

# Step 4: Export production-ready model with separated weights
python -m quant.convert
```

| Stage | Input | Output | Purpose |
|:---|:---|:---|:---|
| **Cleaning** | `data/raw/banking_text.csv` | `data/processed/{train,val,test}.csv` | Normalize Unicode, remove noise, stratified split |
| **Training** | Processed CSVs | `models/raw_model/` | Fine-tune sentence transformer (lr=2e-5, batch=32, 5 epochs) |
| **Quantization** | PyTorch model | `models/onnx_int8/` | ONNX conversion + INT8 dynamic quantization (~75% size reduction) |
| **Export** | Quantized model | `models/production/` | Production-ready artifacts: `model_main.onnx` + `tokenizer.json` + `config.json` |

---

## Production Deployment

The system is **live and publicly accessible** at zero cost:

| Component | URL | Infrastructure |
|:---|:---|:---|
| **Frontend** | [llmfontend.vercel.app](https://llmfontend.vercel.app/) | Vercel — global CDN, auto-deploy on push |
| **Backend API** | [llm-vhhs.onrender.com](https://llm-vhhs.onrender.com) | Render Free Plan — 0.1 CPU, 512MB RAM, Docker |
| **Health Check** | [/health](https://llm-vhhs.onrender.com/health) | Returns `{"status": "ok"}` when model is loaded |

**CI/CD:** Both frontend and backend auto-deploy when code is pushed to the `main` branch on GitHub. No manual deployment steps.

> **Note:** Render Free Tier spins down after inactivity. First request after idle may take ~10 seconds for cold start. Subsequent requests respond in <100ms.

---

<details>
<summary><h2>Optimization Deep Dive</h2></summary>

Running an ML model on 0.1 CPU and 512MB RAM requires aggressive optimization at every layer. Here's what was done and **why**:

### Backend Optimization

The 512MB RAM constraint is the primary bottleneck. Every optimization targets memory reduction:

- **`OMP_NUM_THREADS=1`** — Limits OpenMP parallelism. Multi-threading on 0.1 CPU creates overhead without speedup.
- **`intra_op_num_threads=1`** — ONNX Runtime single-thread execution. Matches the single-core constraint.
- **`enable_cpu_mem_arena=False`** — Disables ONNX Runtime's pre-allocated memory pool. Trades slight latency for predictable memory usage within the 512MB ceiling.
- **`ORT_SEQUENTIAL` execution mode** — Prevents concurrent operator execution that would spike memory.
- **Async lifecycle management** — Model loads once at startup via FastAPI's `lifespan` context manager, not per-request.
- **Minimal dependencies** — Uses `tokenizers` library directly instead of full `transformers`, saving ~200MB of unused imports.

### Frontend Optimization

- **CDN-delivered assets** — Bootstrap and Font Awesome loaded from CDN, not bundled. Zero build step.
- **30-second health polling** — Non-blocking async check of API status. Provides real-time feedback without flooding the free-tier backend.
- **Graceful degradation** — When the API is unavailable (cold start or downtime), the UI shows a clear status indicator and retry mechanism instead of crashing.
- **CSS transforms over JS animations** — GPU-accelerated transitions for smooth UX even on low-end devices.

### Network Optimization

- **Minimal payload** — API returns only `{label, score}`. No unnecessary metadata.
- **Gzip compression** — Reduces response size for slow connections.
- **Keep-alive connections** — Reduces TCP handshake overhead for repeat requests.
- **CORS wildcard** — Eliminates preflight request failures during cross-origin communication.

</details>

---

## Future Roadmap

Framed as **product evolution**, not just technical improvements:

| Phase | Initiative | Business Value |
|:---|:---|:---|
| **Short-term** | Confidence-based escalation | Low-confidence predictions (< 70%) are flagged for human review → prevents misrouting on ambiguous complaints |
| **Short-term** | Complaint analytics dashboard | Real-time visibility into complaint volume, category distribution, and trends → enables proactive product fixes |
| **Mid-term** | Batch classification API | Enterprise integration for bulk processing → supports email ingestion and CRM workflows |
| **Mid-term** | Multi-language support | Extend to English and Chinese for international banking operations → enables regional expansion |
| **Long-term** | Feedback loop & active learning | Human corrections feed back into model retraining → accuracy improves continuously with real-world data |
| **Long-term** | Priority scoring | Combine classification with urgency detection → fraud reports get escalated faster than general inquiries |

---

## Author

**Phat Nguyen**

- Email: [tanphat6406@gmail.com](mailto:tanphat6406@gmail.com)
- LinkedIn: [linkedin.com/in/phat-nguyen-a264722b7](https://www.linkedin.com/in/phat-nguyen-a264722b7/)

---

## License

This project is distributed under the [MIT License](LICENSE).

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/) — Pre-trained multilingual sentence transformer
- [ONNX Runtime](https://onnxruntime.ai/) — High-performance inference engine
- [FastAPI](https://fastapi.tiangolo.com/) — Modern async API framework
- [Render](https://render.com/) — Free-tier cloud hosting with Docker support
- [Vercel](https://vercel.com/) — Frontend hosting with global CDN
