"""
Triage Pipeline Module: Processes Stage 1 Classifier Logits,
calculates Top-K Differential Diagnosis probabilities, computes Shannon Entropy
for Out-of-Distribution (OOD) uncertainty checking, and constructs Stage 2 prompts.
"""

import numpy as np

def compute_entropy(probs: np.ndarray) -> float:
    """Calculates Shannon Entropy for probability array."""
    probs = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(probs * np.log(probs)))

def format_stage1_output(logits: np.ndarray, id2label: dict, top_k: int = 3, entropy_threshold: float = 2.5) -> dict:
    """
    Takes raw logits from Stage 1 Multi-label model, computes Sigmoid probabilities,
    ranks Top-K diseases, checks prediction entropy, and returns formatted JSON.
    """
    probs = 1 / (1 + np.exp(-logits))
    top_indices = np.argsort(probs)[::-1][:top_k]
    
    top_predictions = [
        {
            "disease": id2label[idx],
            "confidence": float(round(probs[idx], 4))
        }
        for idx in top_indices
    ]
    
    top_probs = np.array([p["confidence"] for p in top_predictions], dtype=np.float64)
    top_probs_norm = top_probs / (np.sum(top_probs) + 1e-12)
    entropy = compute_entropy(top_probs_norm)
    
    max_conf = top_predictions[0]["confidence"] if top_predictions else 0.0
    is_ood = bool(max_conf < 0.20 or entropy > entropy_threshold)
    
    return {
        "top_predictions": top_predictions,
        "max_confidence": float(max_conf),
        "entropy_score": float(round(entropy, 4)),
        "ood_uncertainty_flag": is_ood
    }

def construct_stage2_llm_prompt(patient_query: str, stage1_result: dict, rag_guidelines_text: str = "") -> str:
    """
    Constructs a grounded prompt for the Stage 2 Generative LLM.
    """
    top_ddx_str = "\n".join([
        f"- {item['disease']}: {item['confidence']*100:.1f}%"
        for item in stage1_result["top_predictions"]
    ])
    
    prompt = f"""
[SYSTEM INSTRUCTION]
You are an expert clinical triage assistant trained on African epidemiology and WHO guidelines.
Given the patient complaint and the Stage 1 Differential Diagnosis predictions, generate a structured triage response.

[PATIENT CHIEF COMPLAINT]
"{patient_query}"

[STAGE 1 DIFFERENTIAL DIAGNOSIS PROBABILITIES]
{top_ddx_str}

[RETRIEVED WHO/NCDC CLINICAL GUIDELINES]
{rag_guidelines_text if rag_guidelines_text else "Standard WHO AFRO clinical protocols apply."}

[REQUIRED OUTPUT FORMAT]
1. Primary & Differential Diagnoses Summary
2. Urgency Triage Level (EMERGENCY / URGENT / ROUTINE)
3. Recommended Diagnostic Tests (e.g., RDT, Blood Smear, Full Blood Count)
4. Safe First-Aid & Hydration Instructions
5. Red-Flag Emergency Symptoms requiring immediate hospital referral
6. 1-2 Clarifying Follow-Up Questions

[SAFETY MANDATE]
Do NOT provide exact prescription drug dosages without lab test disclaimers.
"""
    return prompt.strip()
