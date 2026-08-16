# Master Multi-Source Dataset Generation Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the full harmonization pipeline to transform DDXPlus, Kaggle 773, AfriMed-QA v2, and SymCAT into the final master dataset (`data/combined_dataset.csv` and `data/combined_dataset.jsonl`).

**Architecture:** Run `scripts/build_canonical_datasets.py` to produce standardized JSONL files in `data/canonical/`, then run `scripts/deduplicate_and_combine.py` to perform MD5 text deduplication and export `data/combined_dataset.csv`.

**Tech Stack:** Python 3.11, Pandas, JSONL, hashlib.

## Global Constraints
- Inputs: `data/DDXPlus/`, `data/Final_Augmented_dataset_Diseases_and_Symptoms.csv`, `data/Afrimed/afri_med_qa_15k_v2.5_phase_2_15275.csv`, `data/raw/symcat_pipeline/`
- Outputs: `data/canonical/*.jsonl`, `data/combined_dataset.csv`, `data/combined_dataset.jsonl`
- Git: Dataset CSV/JSONL files excluded via `.gitignore` to prevent GitHub large file issues.

---

### Task 1: Generate Canonical JSONL Files Across All 4 Sources

**Files:**
- Execute: `scripts/build_canonical_datasets.py`
- Consumes: All 4 local dataset folders/files in `data/`
- Produces: `data/canonical/ddxplus.jsonl`, `data/canonical/kaggle773.jsonl`, `data/canonical/afrimedqa.jsonl`, `data/canonical/symcat.jsonl`

- [ ] **Step 1: Execute `build_canonical_datasets.py`**

```bash
python scripts/build_canonical_datasets.py
```
Expected output: Processing logs with total canonical record counts for all 4 datasets.

---

### Task 2: Perform MD5 Text Deduplication & Export Master CSV

**Files:**
- Execute: `scripts/deduplicate_and_combine.py`
- Consumes: `data/canonical/*.jsonl`
- Produces: `data/combined_dataset.csv` and `data/combined_dataset.jsonl`

- [ ] **Step 1: Execute `deduplicate_and_combine.py`**

```bash
python scripts/deduplicate_and_combine.py
```
Expected output: Deduplication stats and saved `data/combined_dataset.csv`.

---

### Task 3: Verify Dataset Row Counts & Master File Size

**Files:**
- Test: `scratch/run_tests.py`

- [ ] **Step 1: Run unit tests to verify pipeline integrity**

```bash
python scratch/run_tests.py
```
Expected output: `Ran 12 tests in 0.100s OK`.

---

## Execution Handoff

Plan complete and saved to [`docs/superpowers/plans/2026-08-16-master-dataset-build-plan.md`](file:///c:/Users/wisdo/OneDrive/Documents/GitHub/AVASOFT-HEALTH/docs/superpowers/plans/2026-08-16-master-dataset-build-plan.md). Two execution options:

1. **Subagent-Driven (recommended)** - I dispatch a fresh subagent per task to execute and verify each step cleanly.
2. **Inline Execution** - I execute the build commands directly in this session right now.

Which approach would you like me to take?
