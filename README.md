---
title: AVASOFT-HEALTH Clinical Engine & AfroCare-Dx Dataset
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# AVASOFT-HEALTH: Multi-Agent Clinical Triage Engine & AfroCare-Dx Dataset

AVASOFT-HEALTH is an end-to-end multi-agent clinical decision support system designed for rapid symptom analysis, differential diagnosis, and primary healthcare triage across Pan-African and global healthcare settings. 

The platform integrates a Stage 1 multi-label neural classifier (DeBERTa-v3-large), a baseline BioBERT model (Iloriayomide/Symptom_Prediction), voice-activated clinical intake via Groq Whisper (whisper-large-v3), medical image diagnostic upload capabilities, and a Stage 2 LLM reasoning engine grounded in WHO AFRO and NCDC clinical practice guidelines.

---

## AfroCare-Dx: Harmonized Pan-African & Multi-Source Clinical Dataset (1.18M Records)

AfroCare-Dx is a harmonized, multi-source medical dataset created for public research, model training, and clinical AI benchmarking. It merges four distinct medical datasets into a single 4-column canonical schema comprising 1,180,989 unique deduplicated records extracted from a raw pool of 1,308,398 clinical encounters.

### Dataset Files and Documentation
- Dataset Documentation: data/README.txt
- Dataset Licensing & Provenance: data/LICENSES.md
- Master Dataset File (CSV): data/combined_dataset.csv (1,180,989 rows)
- Master Dataset File (JSONL): data/combined_dataset.jsonl (1,180,989 records)

### Data Composition and Source Breakdown

| Source Tag | Source Name | Uncapped Record Count | Primary Content Description |
| :--- | :--- | :--- | :--- |
| ddxplus | DDXPlus (NeurIPS 2022) | 1,025,602 | Decoded clinical EHR symptom histories and evidence codes |
| kaggle773 | Kaggle 773 Diseases & Symptoms | 246,945 | Binary symptom vector chief complaints (773 diseases, 377 symptoms) |
| symcat | SymCAT Probabilistic Pipeline | 20,576 | Probabilistic condition presentations across 801 conditions |
| afrimedqa | AfriMed-QA v2 (Intron Health Phase 2) | 15,275 | Pan-African clinical Q&A and consumer queries across 18 nations |

### Schema Contract
Every record in AfroCare-Dx follows a standardized 4-column schema:
1. id (String): Unique record identifier (e.g. DDX_0000001, KAG_000001, AFR_000001, SYM_000001).
2. text (String): Natural English clinical complaint, patient symptom history, or prompt.
3. labels (String): Pipe-delimited target pathology or medical specialty (label1|label2).
4. source (String): Data provenance tag (ddxplus, kaggle773, afrimedqa, symcat).

### Downloading AfroCare-Dx
Due to size constraints on remote repositories (>100MB), the master CSV file (combined_dataset.csv, ~250MB) is tracked via Git LFS and hosted on Google Drive.
- Google Drive Direct Download Link: (User Google Drive Link Placeholder)
- Local Pipeline Build Commands:
  python scripts/build_canonical_datasets.py
  python scripts/deduplicate_and_combine.py

---

## Key System Features

- Multi-Agent Orchestration: Uses PyTorch and Hugging Face transformers (DeBERTa-v3-large / BioBERT) for rapid baseline classification, then passes logit probability distributions to Google Gemini or Groq LLaMA for natural language clinical reasoning.
- Voice-Activated Triage: Integrates Groq's whisper-large-v3 model to transcribe spoken patient chief complaints in real-time.
- Medical Image Diagnostic Upload: Supports patient image uploads for visual diagnostic assessment integrated with clinical text intake.
- Top-K Differential Predictions: Computes the top 3 most likely disease matches with calibrated confidence percentages.
- Out-of-Distribution (OOD) Safety: Calculates Shannon entropy across logit probability outputs to detect non-medical inputs, nonsensical text, or rare diseases requiring human clinician escalation.

---

## Technical Stack

- Backend: Python 3.11+, FastAPI, Asynchronous REST API, Uvicorn
- Machine Learning: PyTorch, Hugging Face Transformers, DeBERTa-v3-large, BioBERT
- Speech Recognition: Groq Cloud (whisper-large-v3 API)
- Cloud LLM Reasoning: Google GenAI SDK (Gemini 2.5 / 3), Groq Cloud (LLaMA-3.3-70B)
- Environment Isolation: python-dotenv for API key security

---

## System Architecture and Workflow

1. Patient Intake: The patient inputs text, records spoken audio (transcribed via Groq Whisper), or uploads clinical images via the frontend interface.
2. Stage 1 Classification: The FastAPI backend routes text to the neural classifier, generating probability distributions across target pathologies.
3. Safety Check: The system computes Shannon entropy. If entropy exceeds the safety threshold, the system flags the query as Out-of-Distribution (OOD) and prompts clinician review.
4. Stage 2 Reasoning: Top-3 candidate differential diagnoses and confidence scores are formatted into a structured prompt grounded by WHO AFRO and NCDC clinical guidelines.
5. Patient Summary Generation: The LLM outputs clear, empathetic, non-diagnostic triage advice, red-flag warning signs, and next steps for seeking professional care.

---

## Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Ayo288888/AVASOFT-HEALTH.git
cd AVASOFT-HEALTH
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

Required packages include: fastapi, uvicorn, transformers, torch, pydantic, pandas, numpy, python-dotenv, google-genai, groq.

### 3. Configure Environment Variables
Create a .env file in the root directory:
```env
HF_TOKEN=your_huggingface_token
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
```

### 4. Run the Backend Server
```bash
uvicorn main:app --reload
```
The API server will start at http://127.0.0.1:8000.

### 5. Run Verification Suite
```bash
python scratch/run_tests.py
```

### 6. Launch the Web Interface
Open index.html in any web browser to interact with the symptom input, voice intake, and image upload interfaces.

---

## Future Roadmap and Development Plans

1. Stage 1 Distributed Model Training: Train DeBERTa-v3-large across 2x Tesla T4 GPUs on Kaggle using the full uncapped AfroCare-Dx dataset (1.18M rows).
2. Advanced Retrieval-Augmented Generation (RAG): Build a vector index of WHO AFRO guidelines, NCDC outbreak protocols, and clinical management pathways to ground Stage 2 LLM outputs.
3. Pan-African Multilingual Support: Expand the speech and text translation layer to support major African languages including Swahili, Hausa, Yoruba, Igbo, Amharic, and French.
4. Offline Edge Deployment: Package the Stage 1 model and a quantized 3B LLM (e.g. LLaMA-3-3B-Instruct) using ONNX Runtime for deployment on low-power tablets in rural clinics with limited internet connectivity.
5. EHR Integration (HL7 / FHIR): Export generated triage summaries into HL7 FHIR standard formats for seamless integration into hospital electronic health record systems.

---

## Important Disclaimer

This platform is an artificial intelligence research project built for educational and demonstration purposes. It is not a certified medical device and must not be used as a substitute for professional medical diagnosis, advice, or treatment. Always consult a certified healthcare professional for medical concerns.
