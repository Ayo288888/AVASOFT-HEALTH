# Kaggle Stage 1 Multi-Label Model Training Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a production-ready, fully self-contained Jupyter Notebook (`notebooks/kaggle_stage1_deberta_training.ipynb`) for training and evaluating the Stage 1 Multi-Label Clinical Differential Diagnosis classifier on Kaggle (2x Tesla T4 GPUs) using the uncapped **AfroCare-Dx** dataset (`data/combined_dataset.csv`).

**Tech Stack:** Python 3.11, PyTorch, Hugging Face Transformers (`microsoft/deberta-v3-base` & `microsoft/deberta-v3-large`), Scikit-Learn (`MultiLabelBinarizer`), Accelerated GPU Training (`fp16`).

---

## Global Constraints & ML Best Practices
- **Data Leakage Prevention:** `MultiLabelBinarizer` and preprocessing pipelines MUST be fit strictly on the Training set, then applied to Validation and Test sets.
- **Data Split:** Fixed 80% Train / 10% Validation / 10% Test split with fixed random seed (`seed=42`).
- **Loss Function:** `BCEWithLogitsLoss` for multi-label binary cross-entropy across all target disease categories.
- **Narrative Format:** Every code cell in the notebook MUST be followed by an analytical markdown cell discussing metrics, loss curves, and clinical implications.

---

## Task 1: Model Selection & Architecture Design

- [ ] **Step 1: Compare & Select Model Architectures**
  - **Primary Model:** `microsoft/deberta-v3-base` (86M params) / `microsoft/deberta-v3-large` (435M params). Uses disentangled attention and enhanced mask decoder, outperforming BERT/BioBERT on complex medical sentence parsing.
  - **Baseline Model:** `distilbert-base-uncased` (66M params) for fast baseline comparison.

- [ ] **Step 2: Define Loss & Metric Evaluation Harness**
  - Multi-label Loss: `torch.nn.BCEWithLogitsLoss()`
  - Key Metrics: Top-1 Accuracy, Top-3 Accuracy, Micro-F1, Macro-F1, Hamming Loss, and Shannon Entropy OOD score.

---

## Task 2: Create Kaggle GPU Training Notebook (`notebooks/kaggle_stage1_deberta_training.ipynb`)

- [ ] **Step 1: Section 1 - Environment Setup & Dataset Load**
  - Verify GPU availability (`torch.cuda.is_available()`, T4 x2 info).
  - Load `combined_dataset.csv` and log initial row count and column schemas.

- [ ] **Step 2: Section 2 - Exploratory Data Analysis & Target Binarization**
  - Fit `MultiLabelBinarizer` on target labels (`label1|label2`).
  - Plot class frequency distribution for top 30 disease categories.

- [ ] **Step 3: Section 3 - Train / Val / Test Splitting & PyTorch Dataset**
  - Create 80/10/10 random split with fixed seed.
  - Implement PyTorch `ClinicalDataset` class wrapping Hugging Face `AutoTokenizer`.

- [ ] **Step 4: Section 4 - Model Training & Evaluation Loop**
  - Implement PyTorch training loop with `fp16` mixed precision, `AdamW` optimizer, and `CosineAnnealingLR` scheduler.
  - Evaluate on Validation set after each epoch.

- [ ] **Step 5: Section 5 - Test Set Benchmarking & Error Analysis**
  - Benchmark fine-tuned DeBERTa model on unseen Test set.
  - Plot Top-1 vs Top-3 accuracy, confusion matrices, and Shannon Entropy distribution for OOD safety verification.
  - Save trained model weights and label map (`stage1_deberta_weights.pt` & `label_binarizer.pkl`).

---

## Task 3: Local Verification & Test Suite Integration

- [ ] **Step 1: Test Notebook Syntax & Imports locally**
  - Execute test runner script to verify that `notebooks/kaggle_stage1_deberta_training.ipynb` contains valid JSON and code cells.
- [ ] **Step 2: Run unit test suite**
  - Ensure all 12 existing unit tests pass cleanly (`python scratch/run_tests.py`).
