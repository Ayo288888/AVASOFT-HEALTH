# Kaggle Stage 1 Multi-Label Model Training Plan (Comprehensive & Revised)

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task.

## Executive Context & System Architecture

This plan governs the training of **Stage 1** of AfroCare AI's 2-stage clinical decision support system: a multi-label neural classifier that takes patient chief complaints (free-text, voice-transcribed audio, or clinical notes) and outputs a probability distribution over disease categories. The predictions are gated by a mathematical **Shannon-entropy Out-of-Distribution (OOD) safety check** before being passed downstream to a Stage 2 RAG-grounded LLM (WHO AFRO / NCDC clinical guidelines).

### Dataset Composition & Critical Imbalance Framing
Target production inputs are informal, patient-reported text, but the training corpus (**AfroCare-Dx** — 1,180,989 deduplicated records) is ~87%+ synthetic or templated. 

| Source | Records | % of Total | Dataset Character & Nature |
| :--- | :--- | :--- | :--- |
| **DDXPlus (NeurIPS 2022)** | 1,025,602 | ~87% | Synthesized EHRs, templated clinical evidence phrases |
| **Kaggle 773** | 246,945 | ~21%* | Disease-symptom co-occurrence vectors (773 diseases) |
| **AfriMed-QA v2 (Intron Health)** | 15,275 | ~1.3% | **Authentic Pan-African patient Q&A across 18 nations** |
| **SymCAT** | 20,576 | ~1.7% | Probabilistic synthetic condition presentation profiles |

*(Percentages reflect raw-to-cleaned source distribution. Exact post-dedup composition is verified in Section 1 EDA.)*

**Critical Framing:** AfriMed-QA — the only genuinely representative slice of real target African users — accounts for ~1.3% of the dataset. Therefore, **every evaluation step and metric report in this pipeline MUST be source-stratified**, evaluating performance on the authentic AfriMed-QA slice specifically alongside aggregate metrics.

### Hardware & Compute Constraints
- **Hardware:** Kaggle Notebooks with **2x Tesla T4 GPUs** (16GB VRAM each, Turing architecture).
- **Precision:** **fp16 mixed precision only** (`torch.cuda.amp.autocast`). Note: T4 GPUs do **NOT** support `bf16`.

---

## Global Constraints & ML Best Practices

- **Strict Featurization Ordering:** `MultiLabelBinarizer` and preprocessing pipelines MUST be fit strictly on the Training set, then applied to Validation and Test sets independently.
- **Data Splitting & Stratification:** 80/10/10 split with **iterative stratification for multi-label targets** (`skmultilearn` / `iterative_train_test_split`), pre-split deduplication of synthetic templates, fixed `seed=42`, and post-split source-composition verification.
- **Loss Function:** `torch.nn.BCEWithLogitsLoss(pos_weight=...)` with per-class `pos_weight` computed strictly from train-set label frequencies (`pos_weight[c] = num_negatives[c] / num_positives[c]`, clipped to avoid extreme values from rare classes).
- **Narrative Format Requirement:** Every code cell in the notebook MUST be followed by an analytical markdown cell discussing metrics, loss curves, clinical implications, and source-stratified results (focusing on the authentic AfriMed-QA slice).

---

## Task 1: Model Selection & Architecture Design

### Step 1: Model Roster & Sequential Experimentation Logic
Implement all three of the following models sequentially in this order. Do not skip ahead to `-large` before `-base` has trained cleanly:

1. **TF-IDF + Linear Classifier Baseline (`OneVsRestClassifier(LogisticRegression)`)**
   - Fast CPU baseline trained in minutes to establish a sanity-check performance floor before touching GPU resources.
2. **`distilbert-base-uncased` (66M params)**
   - Mid-tier Transformer baseline to measure speed vs accuracy trade-offs.
3. **`microsoft/deberta-v3-base` (86M params) — PRIMARY PRODUCTION TARGET**
   - Primary model. Utilizes disentangled attention to parse complex, multi-clause medical text.
   - **fp16 Stability Mitigations:** Apply gradient clipping (`max_norm=1.0`), attention-mask-aware mean pooling, lower learning rate with warmup, and gradient accumulation.
4. **`microsoft/deberta-v3-large` (435M params) — CONDITIONAL SECOND-PASS ONLY**
   - Train ONLY if `-base` completes a full run without numerical instability or NaN loss spikes. Apply lower learning rate with extended warmup and embedding layer fp32 precision if attempted.

### Step 2: Precise Metric Definitions & OOD Formulation
- **Top-1 / Top-3 Accuracy:** Defined as *"the true label set contains at least one of the model's top-k highest-scoring predictions"*.
- **Micro-F1:** Overall global F1-score across all binary target decisions.
- **Macro-F1:** Reported overall, and separately for classes above/below a minimum support threshold ($\ge 50$ positive examples in train).
- **Hamming Loss:** Multi-label binary error rate.
- **Multi-Label Shannon Entropy (OOD Score):**
  $$p_c = \text{sigmoid}(\text{logit}_c)$$
  $$H_c = -\big(p_c \log(p_c + 1e-9) + (1-p_c) \log(1-p_c + 1e-9)\big)$$
  $$\text{OOD\_score} = \text{mean}(H_c \text{ across all classes})$$
- **Decision Threshold Calibration:** Per-class decision thresholds tuned on the Validation set to maximize per-class F1-score (compared against flat 0.5 cutoff).

---

## Task 2: Environment Setup & Hardware Acceleration

- **Dependency Installation Cell:** Explicitly install `sentencepiece`, `protobuf`, `skmultilearn`, `accelerate`, and `transformers` in Section 1.
- **Version Pinning:** Pin library versions for PyTorch, Transformers, and Scikit-Learn.
- **Multi-GPU Strategy:** Configure PyTorch `DataParallel` or Hugging Face `Accelerate` for dual Tesla T4 GPUs, logging free VRAM on both GPUs prior to execution.

---

## Task 3: Exploratory Data Analysis & Target Binarization

- Fit `MultiLabelBinarizer` on **train split only**.
- Plot class frequency distribution for top 30 disease categories + report the number of rare classes below the minimum support threshold ($\ge 50$ examples).
- **Source-Stratified EDA Breakdown:** Report row counts, label distribution, and average text token length broken down by source column (`ddxplus`, `kaggle773`, `afrimedqa`, `symcat`).

---

## Task 4: Train / Val / Test Splitting & PyTorch Dataset

- **Pre-Split Synthetic Deduplication:** Deduplicate near-identical synthetic template rows in DDXPlus and SymCAT before splitting to eliminate data leakage.
- **Iterative Multi-Label Stratification:** Split using `skmultilearn.model_selection.iterative_train_test_split` into 80% Train / 10% Val / 10% Test with fixed `seed=42`.
- **Source Composition Verification:** Log per-source split percentages to verify AfriMed-QA is evenly distributed across train, val, and test.
- **Attention-Mask-Aware Custom Pooling Head:**
  $$\text{pooled\_embeddings} = \frac{\sum (\text{hidden\_states} \times \text{attention\_mask.unsqueeze}(-1))}{\sum \text{attention\_mask} + 1e-9}$$

---

## Task 5: Model Training & Evaluation Loop

- Mixed precision `fp16` (`torch.cuda.amp.autocast`), `AdamW` optimizer, and `CosineAnnealingLR` scheduler with warmup.
- Gradient clipping (`max_norm=1.0`) and gradient accumulation.
- **Timed Pilot Run:** Execute a timed subset pilot run (5,000–10,000 rows, 1 epoch) to extrapolate exact per-epoch wall-clock time before launching the full run.
- **Checkpointing Strategy:** Save the best checkpoint based on Validation Macro-F1 and AfriMed-QA subset Macro-F1 (retaining last known good checkpoint in case of fp16 loss spikes).

---

## Task 6: Test Set Benchmarking, Source Stratification & OOD Probe

- **Calibrated Test Benchmarking:** Benchmark on held-out test set using per-class calibrated thresholds tuned on Validation.
- **Source-Stratified Performance Reporting:** Report Top-1, Top-3, Micro-F1, and Macro-F1 metrics **broken down individually for each source** (`ddxplus`, `kaggle773`, `afrimedqa`, `symcat`), highlighting the authentic AfriMed-QA slice.
- **OOD Probe Set Verification:** Evaluate Shannon Entropy on an explicit Out-of-Distribution probe set (non-clinical strings, random text, held-out non-medical classes) to prove entropy is significantly higher on OOD inputs vs. in-distribution test samples.
- **Artifact Export:** Save `stage1_deberta_weights.pt`, `label_binarizer.pkl`, and `calibrated_thresholds.pkl`.

---

## Task 7: Notebook Syntax Check & Local Verification

- **Notebook Validation:** Execute generator script to verify valid JSON structure and complete cell definitions in `notebooks/kaggle_stage1_deberta_training.ipynb`.
- **Unit Test Suite:** Run `python scratch/run_tests.py` and confirm all 12 unit tests pass 100% (`OK`).

---

## Execution Handoff

Plan complete and saved to [`docs/superpowers/plans/2026-08-17-kaggle-stage1-training-plan.md`](file:///c:/Users/wisdo/OneDrive/Documents/GitHub/AVASOFT-HEALTH/docs/superpowers/plans/2026-08-17-kaggle-stage1-training-plan.md).
