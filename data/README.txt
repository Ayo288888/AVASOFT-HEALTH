================================================================================
AfroCare-Dx: Harmonized Pan-African & Multi-Source Clinical Triage Dataset
(1.18 Million Records - AVASOFT-HEALTH Release)
================================================================================

Overview:
---------
AfroCare-Dx is a large-scale, harmonized multi-source clinical dataset designed 
for training and evaluating Automatic Symptom Detection (ASD), Clinical Triage, 
and Differential Diagnosis (AD) artificial intelligence systems.

It combines 4 diverse medical datasets into a unified 4-column canonical schema:
1. DDXPlus (NeurIPS 2022) - Synthesized patient EHRs with decoded symptom trees.
2. Kaggle 773 - Disease-symptom co-occurrence matrices (773 diseases, 377 symptoms).
3. AfriMed-QA v2 (Intron Health Phase 2) - Pan-African clinical Q&A & consumer queries.
4. SymCAT - Probabilistic disease symptom profiles across 801 conditions.

Dataset Statistics:
-------------------
- Dataset Name:                 AfroCare-Dx
- Total Raw Input Pool:         1,308,398 records
- Exact MD5 Duplicates Removed: 127,409 records (9.74% deduplication rate)
- Final Unique Master Size:     1,180,989 records

Per-Source Breakdown:
---------------------
Source Tag    | Source Name           | Uncapped Count | Description
--------------+-----------------------+----------------+---------------------------------------------------
ddxplus       | DDXPlus Dataset       | 1,025,602      | Decoded clinical EHR symptom histories
kaggle773     | Kaggle 773            | 246,945        | Binary symptom matrix chief complaints
symcat        | SymCAT Pipeline       | 20,576         | Probabilistic condition presentations
afrimedqa     | AfriMed-QA v2         | 15,275         | Pan-African clinical Q&A & consumer queries

Schema Contract (4 Columns):
----------------------------
1. id      [String]  : Unique sample identifier (e.g. DDX_0000001, KAG_000001, AFR_000001, SYM_000001).
2. text    [String]  : Natural English clinical complaint, patient symptom history, or prompt.
3. labels  [String]  : Pipe-delimited target pathology or medical specialty (e.g. "panic disorder|Anxiety").
4. source  [String]  : Data source provenance tag ("ddxplus", "kaggle773", "afrimedqa", "symcat").

File Formats:
-------------
- combined_dataset.csv   : Standard CSV file (UTF-8 encoded)
- combined_dataset.jsonl : Line-delimited JSON format for streaming ingestion

Licensing & Attribution:
------------------------
- DDXPlus: CC-BY 4.0 (Mila - Quebec AI Institute / NeurIPS 2022)
- Kaggle 773: Public Domain / CC0
- AfriMed-QA v2: CC BY-SA 4.0 (Intron Health Phase 2 Release)
- SymCAT: Open Source (Scraped SymCAT database / Dialogue Health)

Contact & Repository:
---------------------
Repository: https://github.com/Ayo288888/AVASOFT-HEALTH
Project: AVASOFT-HEALTH Pan-African Medical AI Engine
Dataset Identifier: AfroCare-Dx (1.18M)
