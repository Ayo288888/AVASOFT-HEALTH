---
title: AVASOFT-HEALTH Clinical Engine & AfroCare-Dx Dataset
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# 🩺 AVASOFT-HEALTH: AI Multi-Agent Pan-African Medical Triage Engine

**AVASOFT-HEALTH** is an end-to-end multi-agent clinical decision support engine designed for low-resource primary healthcare triage across Pan-African and global healthcare settings. It combines a Stage 1 multi-label neural classifier (`DeBERTa-v3-large`) with a Stage 2 WHO/NCDC-grounded clinical LLM reasoning engine.

---

## 📊 AfroCare-Dx: Harmonized Pan-African & Multi-Source Clinical Dataset (1.18M Records)

We release **AfroCare-Dx**, a harmonized multi-source medical dataset containing **1,180,989 unique deduplicated records** across 1,308,398 raw clinical encounters for public research, benchmark evaluation, and model training.

### 📁 Dataset Files & Documentation
- **Dataset Documentation:** [`data/README.txt`](file:///c:/Users/wisdo/OneDrive/Documents/GitHub/AVASOFT-HEALTH/data/README.txt)
- **Dataset License & Provenance:** [`data/LICENSES.md`](file:///c:/Users/wisdo/OneDrive/Documents/GitHub/AVASOFT-HEALTH/data/LICENSES.md)
- **Master Dataset File (CSV):** `data/combined_dataset.csv` *(1,180,989 rows)*
- **Master Dataset File (JSONL):** `data/combined_dataset.jsonl` *(1,180,989 records)*

### 📈 AfroCare-Dx Composition & Sources

| Source Tag | Source Name | Uncapped Record Count | Primary Content Description |
| :--- | :--- | :--- | :--- |
| **`ddxplus`** | DDXPlus (NeurIPS 2022) | **1,025,602** | Decoded clinical EHR symptom histories |
| **`kaggle773`** | Kaggle 773 Diseases & Symptoms | **246,945** | Binary symptom vector chief complaints |
| **`symcat`** | SymCAT Probabilistic Pipeline | **20,576** | Probabilistic condition presentations across 801 diseases |
| **`afrimedqa`** | AfriMed-QA v2 (Intron Health Phase 2) | **15,275** | Pan-African clinical Q&A & consumer queries (18 nations) |

### 🛠️ Schema Contract
Every record in **AfroCare-Dx** (`data/combined_dataset.csv`) follows the standardized 4-column canonical schema:
1. `id` *(String)*: Unique record identifier (e.g. `DDX_0000001`, `KAG_000001`, `AFR_000001`, `SYM_000001`).
2. `text` *(String)*: Natural English clinical complaint, patient symptom history, or prompt.
3. `labels` *(String)*: Pipe-delimited target pathology or medical specialty (`label1|label2`).
4. `source` *(String)*: Data provenance tag (`ddxplus`, `kaggle773`, `afrimedqa`, `symcat`).

### ☁️ Downloading AfroCare-Dx
Due to GitHub file size limits (>100MB), the full 1.18M master CSV (`combined_dataset.csv`, ~250MB) is hosted on Google Drive:
- 🔗 **Google Drive Direct Download Link:** *(User Google Drive Link Placeholder)*
- 🐙 **Git LFS Tracked:** Configured via `.gitattributes` for Git Large File Storage.

To build **AfroCare-Dx** locally from raw sources:
```bash
python scripts/build_canonical_datasets.py
python scripts/deduplicate_and_combine.py
```

---

## 🚀 Key Features

* **Multi-Agent Orchestration:** Uses a Stage 1 multi-label classifier (`DeBERTa-v3-large`) for top-3 differential diagnosis logit estimation, then grounds Stage 2 LLM generation using WHO AFRO & NCDC clinical guidelines.
* **Voice-Activated Triage:** Integrates Groq's ultra-fast `whisper-large-v3` model, allowing patients to speak their symptoms.
* **Out-of-Distribution (OOD) Safety:** Computes Shannon entropy on Stage 1 probability distributions to flag rare or non-medical inputs safely.

---

## 🧠 Tech Stack
* **Language & Frameworks:** Python 3.11+, PyTorch, Hugging Face `transformers`, FastAPI
* **Speech Recognition:** Groq Cloud (`whisper-large-v3`)
* **LLM Engine:** Google Gemini / Groq LLaMA-3.3-70B

---

## 📦 Installation & Setup

1. **Clone the Repository:**
```bash
git clone https://github.com/Ayo288888/AVASOFT-HEALTH.git
cd AVASOFT-HEALTH
```

2. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run Backend Server:**
```bash
uvicorn main:app --reload
```

4. **Run Unit Verification:**
```bash
python scratch/run_tests.py
```

---

## ⚠️ Important Disclaimer
This project is for educational and research demonstration purposes only. It is not a certified medical device and should not replace professional clinical diagnosis. Always consult a certified healthcare professional.
