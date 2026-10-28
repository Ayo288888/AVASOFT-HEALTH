"""
Label map v2: maps every source label in combined_dataset.csv onto the 60 conditions
in data/conditions_v2.csv.

Targets:
  <condition name>  one of the 60 conditions
  OTHER             a real medical condition outside the 60 (trains the "other" class)
  RELABEL           AfriMed-QA rows: labelled by specialty, not disease, so they need relabelling
  DROP              unusable rows (SymCAT: the label is written inside the text)

Usage:
  python scripts/build_label_map.py path/to/combined_dataset.csv [extra canonical .jsonl files...]
Extra JSONL files count sources not yet merged into the combined dataset.
Writes data/label_map_v2.csv and data/label_coverage_v2.csv.
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CONDITIONS_CSV = ROOT / "data" / "conditions_v2.csv"

DDXPLUS = {
    "URTI": "Upper respiratory tract infection / common cold",
    "Influenza": "Upper respiratory tract infection / common cold",
    "Acute laryngitis": "Upper respiratory tract infection / common cold",
    "Acute rhinosinusitis": "Upper respiratory tract infection / common cold",
    "Viral pharyngitis": "Pharyngotonsillitis",
    "Pneumonia": "Pneumonia",
    "Bronchiolitis": "Bronchiolitis",
    "Bronchospasm / acute asthma exacerbation": "Asthma (acute attack)",
    "Acute COPD exacerbation / infection": "Chronic obstructive pulmonary disease",
    "Acute otitis media": "Otitis media",
    "Tuberculosis": "Pulmonary tuberculosis",
    "HIV (initial infection)": "HIV infection / advanced HIV disease",
    "Anemia": "Anaemia (iron deficiency and nutritional)",
    "GERD": "Peptic ulcer disease / dyspepsia",
    "Possible NSTEMI / STEMI": "Acute coronary syndrome",
    "Unstable angina": "Acute coronary syndrome",
    "Acute pulmonary edema": "Heart failure",
    "Panic attack": "Anxiety disorder",
}

KAGGLE773 = {
    "malaria": "Malaria (uncomplicated)",
    "typhoid fever": "Typhoid fever",
    "sepsis": "Sepsis",
    "meningitis": "Bacterial meningitis",
    "infectious gastroenteritis": "Acute gastroenteritis / watery diarrhoea",
    "noninfectious gastroenteritis": "Acute gastroenteritis / watery diarrhoea",
    "tuberculosis": "Pulmonary tuberculosis",
    "human immunodeficiency virus infection (hiv)": "HIV infection / advanced HIV disease",
    "viral hepatitis": "Viral hepatitis B",  # shortcut: hepatitis type unspecified, re-check if hep A/C matter
    "pneumonia": "Pneumonia",
    "common cold": "Upper respiratory tract infection / common cold",
    "flu": "Upper respiratory tract infection / common cold",
    "acute sinusitis": "Upper respiratory tract infection / common cold",
    "laryngitis": "Upper respiratory tract infection / common cold",
    "strep throat": "Pharyngotonsillitis",
    "tonsillitis": "Pharyngotonsillitis",
    "pharyngitis": "Pharyngotonsillitis",
    "acute bronchiolitis": "Bronchiolitis",
    "asthma": "Asthma (acute attack)",
    "acute bronchospasm": "Asthma (acute attack)",
    "chronic obstructive pulmonary disease (copd)": "Chronic obstructive pulmonary disease",
    "acute otitis media": "Otitis media",
    "otitis media": "Otitis media",
    "chronic otitis media": "Otitis media",
    "hypertensive heart disease": "Hypertension",
    "malignant hypertension": "Hypertensive emergency",
    "heart failure": "Heart failure",
    "stroke": "Stroke",
    "intracerebral hemorrhage": "Stroke",
    "transient ischemic attack": "Stroke",
    "heart attack": "Acute coronary syndrome",
    "diabetes": "Diabetes mellitus",
    "diabetic ketoacidosis": "Diabetic emergency (DKA, HHS or hypoglycaemia)",
    "hypoglycemia": "Diabetic emergency (DKA, HHS or hypoglycaemia)",
    "hyperosmotic hyperketotic state": "Diabetic emergency (DKA, HHS or hypoglycaemia)",
    "chronic kidney disease": "Kidney disease (acute or chronic)",
    "acute kidney injury": "Kidney disease (acute or chronic)",
    "kidney failure": "Kidney disease (acute or chronic)",
    "sickle cell crisis": "Sickle cell crisis",
    "sickle cell anemia": "Sickle cell crisis",
    "iron deficiency anemia": "Anaemia (iron deficiency and nutritional)",
    "anemia": "Anaemia (iron deficiency and nutritional)",
    "gastroduodenal ulcer": "Peptic ulcer disease / dyspepsia",
    "gastritis": "Peptic ulcer disease / dyspepsia",
    "indigestion": "Peptic ulcer disease / dyspepsia",
    "gastroesophageal reflux disease (gerd)": "Peptic ulcer disease / dyspepsia",
    "cirrhosis": "Chronic liver disease / cirrhosis",
    "liver disease": "Chronic liver disease / cirrhosis",
    "alcoholic liver disease": "Chronic liver disease / cirrhosis",
    "appendicitis": "Acute appendicitis",
    "intestinal obstruction": "Intestinal obstruction / intussusception",
    "intussusception": "Intestinal obstruction / intussusception",
    "volvulus": "Intestinal obstruction / intussusception",
    "urinary tract infection": "Urinary tract infection",
    "cystitis": "Urinary tract infection",
    "pyelonephritis": "Urinary tract infection",
    "gonorrhea": "Sexually transmitted infection / genital discharge",
    "chlamydia": "Sexually transmitted infection / genital discharge",
    "trichomonas infection": "Sexually transmitted infection / genital discharge",
    "urethritis": "Sexually transmitted infection / genital discharge",
    "syphilis": "Sexually transmitted infection / genital discharge",
    "genital herpes": "Sexually transmitted infection / genital discharge",
    "pelvic inflammatory disease": "Pelvic inflammatory disease",
    "preeclampsia": "Pre-eclampsia / eclampsia",
    "hypertension of pregnancy": "Pre-eclampsia / eclampsia",
    "ectopic pregnancy": "Ectopic pregnancy",
    "spontaneous abortion": "Miscarriage / threatened miscarriage",
    "threatened pregnancy": "Miscarriage / threatened miscarriage",
    "missed abortion": "Miscarriage / threatened miscarriage",
    "uterine atony": "Postpartum haemorrhage",
    "breast cancer": "Breast cancer (warning signs)",
    "cervical cancer": "Cervical cancer (warning signs)",
    "prostate cancer": "Prostate cancer (warning signs)",
    "pyogenic skin infection": "Skin and soft tissue infection (cellulitis, abscess, impetigo)",
    "impetigo": "Skin and soft tissue infection (cellulitis, abscess, impetigo)",
    "infection of open wound": "Skin and soft tissue infection (cellulitis, abscess, impetigo)",
    "scabies": "Scabies",
    "fungal infection of the skin": "Fungal skin infection (tinea)",
    "athlete's foot": "Fungal skin infection (tinea)",
    "fungal infection of the hair": "Fungal skin infection (tinea)",
    "conjunctivitis": "Conjunctivitis",
    "conjunctivitis due to allergy": "Conjunctivitis",
    "conjunctivitis due to bacteria": "Conjunctivitis",
    "conjunctivitis due to virus": "Conjunctivitis",
    "dental caries": "Dental caries / gum disease",
    "gum disease": "Dental caries / gum disease",
    "tooth abscess": "Dental caries / gum disease",
    "chronic back pain": "Low back pain",
    "lumbago": "Low back pain",
    "osteoarthritis": "Osteoarthritis",
    "arthritis of the hip": "Osteoarthritis",
    "epilepsy": "Seizures (epilepsy or febrile convulsion)",
    "depression": "Depression",
    "dysthymic disorder": "Depression",
    "postpartum depression": "Depression",
    "anxiety": "Anxiety disorder",
    "panic disorder": "Anxiety disorder",
    "panic attack": "Anxiety disorder",
    "neonatal jaundice": "Neonatal jaundice",
}

SYMPTOM2DISEASE = {
    "Malaria": "Malaria (uncomplicated)",
    "Typhoid": "Typhoid fever",
    "Common Cold": "Upper respiratory tract infection / common cold",
    "Pneumonia": "Pneumonia",
    "Bronchial Asthma": "Asthma (acute attack)",
    "Hypertension": "Hypertension",
    "diabetes": "Diabetes mellitus",
    "urinary tract infection": "Urinary tract infection",
    "gastroesophageal reflux disease": "Peptic ulcer disease / dyspepsia",
    "peptic ulcer disease": "Peptic ulcer disease / dyspepsia",
    "Fungal infection": "Fungal skin infection (tinea)",
    "Impetigo": "Skin and soft tissue infection (cellulitis, abscess, impetigo)",
}

ZENODO_INFECTIOUS = {
    "Shigellosis (Bacillary Dysentery)": "Dysentery (bloody diarrhoea)",
    "Norovirus Infection": "Acute gastroenteritis / watery diarrhoea",
    "Measles": "Measles",
    "Strep Throat": "Pharyngotonsillitis",
    "Common Cold": "Upper respiratory tract infection / common cold",
    "Conjunctivitis (Pink Eye)": "Conjunctivitis",
    "Impetigo": "Skin and soft tissue infection (cellulitis, abscess, impetigo)",
}

# AfriMed-QA seed labels are lowercased with list numbering stripped by the adapter.
_AFS = {
    "Malaria (uncomplicated)": ["malaria", "malaria (mosquito-borne illness)", "malaria in pregnancy"],
    "Severe malaria (cerebral or severe anaemia)": ["cerebral malaria", "malaria-induced neurological issues"],
    "Typhoid fever": ["typhoid fever"],
    "Bacterial meningitis": ["meningitis", "tuberculosis meningitis"],
    "Lassa fever": ["lassa fever"],
    "Yellow fever": ["yellow fever"],
    "Cholera": ["cholera"],
    "Acute gastroenteritis / watery diarrhoea": ["diarrheal diseases", "gastroenteritis", "gastrointestinal infections"],
    "Dysentery (bloody diarrhoea)": ["dysentery"],
    "Pulmonary tuberculosis": ["tuberculosis", "tuberculosis (tb)"],
    "HIV infection / advanced HIV disease": ["hiv/aids"],
    "Measles": ["measles"],
    "Viral hepatitis B": ["hepatitis b and c"],  # shortcut: B and C share one label here, split if C matters
    "Schistosomiasis": ["schistosomiasis"],
    "Pneumonia": ["pneumonia"],
    "Upper respiratory tract infection / common cold": [
        "upper respiratory infection", "upper respiratory infections", "upper respiratory tract infections",
        "laryngitis", "sinusitis", "acute sinusitis pain"],
    "Pharyngotonsillitis": ["tonsillitis", "pharyngitis"],
    "Asthma (acute attack)": ["asthma", "asthma exacerbation (related to air quality changes)"],
    "Chronic obstructive pulmonary disease": ["chronic obstructive pulmonary disease (copd)", "chronic bronchitis"],
    "Otitis media": ["ear infections", "ear infection", "otitis media", "acute ear pain from infections"],
    "Hypertension": ["hypertension"],
    "Heart failure": ["heart failure"],
    "Stroke": ["stroke", "subarachnoid hemorrhage"],
    "Acute coronary syndrome": ["heart attack"],
    "Diabetes mellitus": ["diabetes mellitus"],
    "Kidney disease (acute or chronic)": [
        "chronic kidney disease", "chronic kidney disease (ckd)", "acute kidney injury", "kidney disease"],
    "Sickle cell crisis": [
        "sickle cell disease", "sickle cell bone crisis", "sickle cell crisis pain",
        "sickle cell anemia", "sickle cell anaemia"],
    "Anaemia (iron deficiency and nutritional)": ["iron deficiency anemia", "anemia", "prenatal anemia"],
    "Peptic ulcer disease / dyspepsia": [
        "dyspepsia", "abdominal pain due to peptic ulcers", "peptic ulcer disease", "peptic ulcers",
        "gastritis", "gastroesophageal reflux disease (gerd)"],
    "Acute appendicitis": ["appendicitis"],
    "Intestinal obstruction / intussusception": ["intestinal obstructions"],
    "Urinary tract infection": [
        "urinary tract infections", "urinary tract infection (uti)", "urinary tract infections (utis)",
        "urinary tract infection", "urinary tract infection pain", "kidney infection (pyelonephritis)"],
    "Sexually transmitted infection / genital discharge": [
        "genital ulcers (stds related)", "urethritis", "genital tract infections"],
    "Pelvic inflammatory disease": ["pelvic inflammatory disease", "pelvic inflammatory disease (pid)"],
    "Pre-eclampsia / eclampsia": ["pre-eclampsia/eclampsia", "preeclampsia"],
    "Ectopic pregnancy": ["ectopic pregnancy"],
    "Miscarriage / threatened miscarriage": ["miscarriage"],
    "Postpartum haemorrhage": ["obstetric hemorrhage"],  # shortcut: includes antepartum bleeding, check with a clinician
    "Breast cancer (warning signs)": ["breast cancer", "breast lumps and changes", "breast lump"],
    "Cervical cancer (warning signs)": ["cervical cancer"],
    "Prostate cancer (warning signs)": ["prostate cancer"],
    "Skin and soft tissue infection (cellulitis, abscess, impetigo)": ["skin infections", "abscesses", "folliculitis"],
    "Scabies": ["scabies"],
    "Fungal skin infection (tinea)": ["tinea capitis"],
    "Conjunctivitis": ["conjunctivitis", "conjunctivitis (allergic)", "conjunctivitis (pink eye)"],
    "Dental caries / gum disease": [
        "dental caries", "dental pain", "gingivitis", "dental problems", "dental and oral health problems"],
    "Low back pain": ["low back pain"],
    "Osteoarthritis": ["osteoarthritis", "osteoarthritis pain"],
    "Seizures (epilepsy or febrile convulsion)": ["epilepsy", "epilepsy/seizures"],
    "Depression": ["depression", "postpartum depression"],
    "Anxiety disorder": ["anxiety disorders", "generalized anxiety disorder", "panic disorder"],
    # shortcut: AfriMed "malnutrition" is mostly weight loss and stunting, not strictly SAM; clinician to confirm
    "Severe acute malnutrition": ["malnutrition", "malnutrition-related problems", "malnutrition-related issues"],
}
_AFS_VAGUE = [
    "infections", "non-communicable diseases", "mental health disorders", "respiratory tract infections",
    "acute respiratory infections", "respiratory infections", "hypertensive disorders",
    "hypertension-related problems", "genitourinary infections", "dermatitis and skin infections",
    "eye infections", "eye conditions", "eye conditions (e.g., trachoma)", "neonatal disorders",
    "congenital disorders", "pregnancy-related issues", "maternal health issues", "zoonotic diseases",
    "hepatitis", "parasitic infections", "heart disease", "diabetic complications", "trauma",
]
AFRIMEDQA_SEED = {name: target for target, names in _AFS.items() for name in names}
AFRIMEDQA_SEED.update({name: "DROP" for name in _AFS_VAGUE})

EXPLICIT = {
    "ddxplus": DDXPLUS,
    "kaggle773": KAGGLE773,
    "symptom2disease": SYMPTOM2DISEASE,
    "zenodo_infectious": ZENODO_INFECTIOUS,
    "afrimedqa_seed": AFRIMEDQA_SEED,
}
DEFAULT = {
    "ddxplus": "OTHER",
    "kaggle773": "OTHER",
    "symptom2disease": "OTHER",
    "zenodo_infectious": "OTHER",
    "afrimedqa_seed": "OTHER",
    "afrimedqa": "RELABEL",
    "symcat": "DROP",
}


def load_conditions() -> dict:
    with open(CONDITIONS_CSV, encoding="utf-8") as f:
        return {r["condition"]: r["id"] for r in csv.DictReader(f)}


def target_for(source: str, label: str) -> str:
    return EXPLICIT.get(source, {}).get(label, DEFAULT[source])


def count_labels(combined_csv: str) -> Counter:
    counts = Counter()
    for chunk in pd.read_csv(combined_csv, usecols=["labels", "source"], chunksize=200_000, dtype=str):
        for labels, source in zip(chunk["labels"].fillna(""), chunk["source"]):
            for label in labels.split("|"):
                if label.strip():
                    counts[(source, label.strip())] += 1
    return counts


def count_jsonl_labels(jsonl_path: str) -> Counter:
    counts = Counter()
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            for label in rec["labels"]:
                counts[(rec["source"], label.strip())] += 1
    return counts


def main(combined_csv: str, extra_jsonl: list[str]):
    conditions = load_conditions()
    counts = count_labels(combined_csv)
    for path in extra_jsonl:
        counts.update(count_jsonl_labels(path))

    map_rows, coverage = [], Counter()
    for (source, label), n in sorted(counts.items()):
        target = target_for(source, label)
        map_rows.append([source, label, n, conditions.get(target, target), target])
        coverage[target] += n

    with open(ROOT / "data" / "label_map_v2.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "source_label", "rows", "target_id", "target"])
        w.writerows(map_rows)

    with open(ROOT / "data" / "label_coverage_v2.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "condition", "rows"])
        for name, cid in conditions.items():
            w.writerow([cid, name, coverage[name]])
        for special in ("OTHER", "RELABEL", "DROP"):
            w.writerow([special, special, coverage[special]])

    for name, cid in conditions.items():
        print(f"{cid}  {coverage[name]:>8,}  {name}")
    for special in ("OTHER", "RELABEL", "DROP"):
        print(f"{special:<7} {coverage[special]:>8,}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "combined_dataset.csv"), sys.argv[2:])
