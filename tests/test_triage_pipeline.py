import unittest
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.triage_pipeline import compute_entropy, format_stage1_output, construct_stage2_llm_prompt

class TestTriagePipeline(unittest.TestCase):
    def test_compute_entropy(self):
        probs = np.array([0.5, 0.5])
        entropy = compute_entropy(probs)
        self.assertAlmostEqual(entropy, 0.6931, places=3)

    def test_format_stage1_output_normal(self):
        logits = np.array([4.0, 1.5, -0.5, -3.0])
        id2label = {0: "Malaria_Plasmodium_falciparum", 1: "Typhoid_Fever", 2: "Cholera", 3: "Asthma"}
        
        result = format_stage1_output(logits, id2label, top_k=3)
        self.assertEqual(len(result["top_predictions"]), 3)
        self.assertEqual(result["top_predictions"][0]["disease"], "Malaria_Plasmodium_falciparum")
        self.assertFalse(result["ood_uncertainty_flag"])

    def test_format_stage1_output_ood(self):
        logits = np.array([-2.0, -3.0, -4.0, -5.0])
        id2label = {0: "Malaria", 1: "Typhoid", 2: "Cholera", 3: "Asthma"}
        
        result = format_stage1_output(logits, id2label, top_k=3)
        self.assertTrue(result["ood_uncertainty_flag"])

    def test_prompt_construction(self):
        stage1_res = {
            "top_predictions": [
                {"disease": "Malaria_Plasmodium_falciparum", "confidence": 0.85},
                {"disease": "Typhoid_Fever", "confidence": 0.25}
            ]
        }
        prompt = construct_stage2_llm_prompt("I have high fever and chills", stage1_res)
        self.assertIn("Malaria_Plasmodium_falciparum: 85.0%", prompt)

if __name__ == "__main__":
    unittest.main()
