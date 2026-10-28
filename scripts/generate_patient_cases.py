"""
Generates synthetic patient-voice complaints for thinly covered conditions, using
data/condition_profiles_v2.json and AfriMed-QA clinician seeds, with Gemini.

Every case that names its condition (profile leak_terms) is discarded, so the model
must learn from symptoms, not from the label. Output is a review sheet: nothing here
should be trained on until a clinician has checked it.

Usage:
  python scripts/generate_patient_cases.py --seeds data/canonical/afrimedqa_seed.jsonl \
      --per-condition 10 --out data/generated/cases_batch1.csv [--only C01,C06]
Needs GEMINI_API_KEY in the environment or in the .env given by --env.
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_label_map import load_conditions, target_for

PROFILES_JSON = ROOT / "data" / "condition_profiles_v2.json"
MODEL = "gemini-2.5-flash"  # Gemini free tier: 20 requests per day
MIN_WORDS = 8  # shorter cases were mostly too vague to belong to one condition


def load_profiles() -> dict:
    data = json.load(open(PROFILES_JSON, encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def load_seeds(seed_jsonl: str) -> dict:
    """Maps condition name -> list of AfriMed-QA symptom phrases."""
    seeds = defaultdict(list)
    if not seed_jsonl or not Path(seed_jsonl).exists():
        return seeds
    for line in open(seed_jsonl, encoding="utf-8"):
        rec = json.loads(line)
        target = target_for(rec["source"], rec["labels"][0])
        seeds[target].append(rec["text"].removeprefix("I have ").rstrip("."))
    return seeds


def build_prompt(profile: dict, seeds: list, n: int) -> str:
    seed_txt = "; ".join(seeds[:8]) if seeds else "none"
    return f"""You are helping build training data for a Nigerian health triage chatbot.
Write {n} different messages that real patients or caregivers in Nigeria might type into a health chatbot.

Who is writing: {profile['narrator']}
Symptoms this illness can cause: {'; '.join(profile['key_symptoms'])}
Danger signs that may appear in some messages: {'; '.join(profile['red_flags']) or 'none'}
Typical situations: {'; '.join(profile['context'])}
How Nigerian clinicians summarised such complaints: {seed_txt}

Rules:
- Each message mentions 2 to 5 of the symptoms, in everyday words, the way a person would describe them.
- Every message must include at least one telling sign or situation from the lists above, so a doctor
  reading it would think of this illness and not of a general fever or stomach upset.
- Vary age, sex and how long it has lasted.
- At most 3 in 10 messages say what was already tried, and never in the same words twice.
- About 6 in 10 in Nigerian English and 4 in 10 in Nigerian Pidgin. Use pidgin words only in their real meaning.
- Every message is at least 10 words. Vary length up to four sentences. Some have typos or no punctuation.
- Never name or guess the illness. Never use these words: {', '.join(profile['leak_terms'])}.
- No medical jargon a patient would not use.

Return only a JSON array of {n} strings."""


def parse_response(text: str) -> list:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [s for s in data if isinstance(s, str)]


def clean_cases(cases: list, leak_terms: list, required_clues: list = ()) -> list:
    out, seen = [], set()
    for case in cases:
        # Models emit non-breaking hyphens and curly quotes that real patients never type.
        text = " ".join(case.replace("‑", "-").replace("’", "'").split())
        key = text.lower()
        if len(text.split()) < MIN_WORDS or key in seen:
            continue
        if any(term.lower() in key for term in leak_terms):
            continue
        if not all(any(clue in key for clue in group) for group in required_clues):
            continue
        seen.add(key)
        out.append(text)
    return out


def generate_for_condition(cid: str, name: str, profile: dict, seeds: list, n: int, llm) -> list:
    asked = n * 2  # ask for extra: the leak, clue, length and dedupe filters discard some
    raw = parse_response(llm(build_prompt(profile, seeds, asked)))
    cases = clean_cases(raw, profile["leak_terms"], profile.get("required_clues", []))[:n]
    return [{"condition_id": cid, "condition": name, "text": t, "source": profile["source"],
             "reviewer_ok": "", "reviewer_note": ""} for t in cases]


def merge_rows(existing: list, new: list) -> list:
    """Keeps existing rows for conditions this run did not regenerate, so a partial rerun never loses data."""
    redone = {r["condition_id"] for r in new}
    return [r for r in existing if r["condition_id"] not in redone] + new


def gemini_llm(api_key: str, model: str):
    from google import genai
    client = genai.Client(api_key=api_key)

    def call(prompt: str) -> str:
        res = client.models.generate_content(
            model=model, contents=prompt,
            config={"temperature": 1.0, "response_mime_type": "application/json"},
        )
        return res.text or ""
    return call


def groq_llm(api_key: str, model: str):
    from groq import Groq
    client = Groq(api_key=api_key)

    def call(prompt: str) -> str:
        # Reasoning models spend the default 3,072-token budget thinking and return nothing.
        res = client.chat.completions.create(
            model=model, temperature=1.0, messages=[{"role": "user", "content": prompt}],
            reasoning_effort="low", max_completion_tokens=8000,
        )
        if res.choices[0].finish_reason == "length":
            raise RuntimeError("Groq reply was cut off at the token limit")
        return res.choices[0].message.content or ""
    return call


PROVIDERS = {
    "gemini": ("GEMINI_API_KEY", MODEL, gemini_llm),
    "groq": ("GROQ_API_KEY", "openai/gpt-oss-120b", groq_llm),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=str(ROOT / "data" / "canonical" / "afrimedqa_seed.jsonl"))
    ap.add_argument("--per-condition", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "data" / "generated" / "cases_batch1.csv"))
    ap.add_argument("--only", default="", help="comma-separated condition ids")
    ap.add_argument("--env", default="", help="path to a .env file holding the provider's API key")
    ap.add_argument("--provider", choices=sorted(PROVIDERS), default="gemini")
    ap.add_argument("--model", default="", help="override the provider's default model")
    ap.add_argument("--pause", type=float, default=0, help="seconds between requests (Groq free tier: ~8k tokens/min)")
    args = ap.parse_args()

    if args.env:
        from dotenv import load_dotenv
        load_dotenv(args.env)
    key_name, default_model, make_llm = PROVIDERS[args.provider]
    api_key = os.getenv(key_name)
    if not api_key:
        sys.exit(f"{key_name} is not set.")
    model = args.model or default_model

    names = {cid: name for name, cid in load_conditions().items()}
    profiles = load_profiles()
    only = set(filter(None, args.only.split(",")))
    seeds = load_seeds(args.seeds)
    llm = make_llm(api_key, model)

    rows = []
    for cid, profile in profiles.items():
        if only and cid not in only:
            continue
        time.sleep(args.pause)
        got = None
        for attempt in (1, 2):
            try:
                got = generate_for_condition(cid, names[cid], profile, seeds.get(names[cid], []), args.per_condition, llm)
                break
            except Exception as e:  # one failed condition should not lose the rest of the batch
                msg = str(e)
                print(f"{cid} failed (attempt {attempt}): {msg[:160]}")
                if "per day" in msg or "RESOURCE_EXHAUSTED" in msg:
                    break  # daily quota: retrying today is pointless
                if attempt == 1 and ("per minute" in msg or "rate" in msg.lower()):
                    time.sleep(65)  # per-minute limit: wait for the window to reset
        if got is None:
            if "per day" in msg or "RESOURCE_EXHAUSTED" in msg:
                print("Daily quota exhausted, stopping. Rerun tomorrow or with another --provider.")
                break
            continue
        for r in got:
            r["generator"] = f"{args.provider}:{model}"
        print(f"{cid} {names[cid]}: {len(got)} cases")
        rows.extend(got)

    fields = ["condition_id", "condition", "text", "source", "generator", "reviewer_ok", "reviewer_note"]
    out = Path(args.out)
    existing = list(csv.DictReader(open(out, encoding="utf-8"))) if out.exists() else []
    merged = merge_rows(existing, rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in fields} for r in merged)
    print(f"Generated {len(rows)} cases; {len(merged)} cases now in {out}")


if __name__ == "__main__":
    main()
