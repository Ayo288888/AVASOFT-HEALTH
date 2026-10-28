import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_patient_cases import (build_prompt, clean_cases, generate_for_condition, load_profiles,
                                            merge_rows, parse_response)

PROFILE = {
    "narrator": "adult describing own symptoms",
    "key_symptoms": ["fever at night", "chills"],
    "red_flags": [],
    "context": ["sleeps without a net"],
    "leak_terms": ["malaria", "plasmodium"],
    "source": "https://example.org",
}


class TestParseResponse(unittest.TestCase):
    def test_plain_and_fenced_json(self):
        self.assertEqual(parse_response('["a", "b"]'), ["a", "b"])
        self.assertEqual(parse_response('```json\n["a"]\n```'), ["a"])

    def test_bad_output_returns_empty(self):
        self.assertEqual(parse_response("not json"), [])
        self.assertEqual(parse_response('{"text": "a"}'), [])
        self.assertEqual(parse_response('["ok", 3, null]'), ["ok"])


class TestCleanCases(unittest.TestCase):
    def test_drops_leaks_short_and_duplicates(self):
        cases = [
            "My body dey hot every night and I dey shiver well well",
            "my body dey hot every night and  I dey shiver well well",   # duplicate after normalising
            "I think na MALARIA, I get fever and chills since Monday",    # names the condition
            "fever",                                                     # too short
        ]
        self.assertEqual(clean_cases(cases, PROFILE["leak_terms"]),
                         ["My body dey hot every night and I dey shiver well well"])

    def test_requires_a_clue_from_every_group(self):
        clues = [["blood"], ["urine", "piss", "stool"]]
        cases = [
            "I see blood at the end when I pass urine after swimming",   # both groups
            "My lower belly has been paining me badly for one week now",  # no clue at all
            "There is blood on my pant but I don't know where from o",    # only first group
        ]
        self.assertEqual(clean_cases(cases, [], clues),
                         ["I see blood at the end when I pass urine after swimming"])

    def test_normalises_model_punctuation(self):
        self.assertEqual(clean_cases(["My 5‑year‑old daughter’s stool is like rice water today"], []),
                         ["My 5-year-old daughter's stool is like rice water today"])

    def test_drops_cases_under_eight_words(self):
        self.assertEqual(clean_cases(["Snake bite. Painful swelling on my leg."], []), [])


class TestPromptAndGeneration(unittest.TestCase):
    def test_prompt_contains_profile_and_forbidden_words(self):
        p = build_prompt(PROFILE, ["fever, chills"], 10)
        self.assertIn("fever at night", p)
        self.assertIn("fever, chills", p)
        self.assertIn("malaria", p)
        self.assertIn("10", p)

    def test_generate_filters_with_fake_llm(self):
        fake_llm = lambda prompt: '["I get fever every night and cold dey catch me well", "Na malaria I get, my body dey hot"]'
        rows = generate_for_condition("C01", "Malaria (uncomplicated)", PROFILE, [], 2, fake_llm)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["condition_id"], "C01")
        self.assertEqual(rows[0]["text"], "I get fever every night and cold dey catch me well")

    def test_rerun_keeps_conditions_it_did_not_regenerate(self):
        existing = [{"condition_id": "C01", "text": "old malaria"}, {"condition_id": "C02", "text": "old severe"}]
        new = [{"condition_id": "C02", "text": "new severe"}]
        self.assertEqual(merge_rows(existing, new),
                         [{"condition_id": "C01", "text": "old malaria"}, {"condition_id": "C02", "text": "new severe"}])

    def test_real_profiles_file_is_complete(self):
        profiles = load_profiles()
        self.assertEqual(len(profiles), 20)
        for cid, p in profiles.items():
            for key in ("narrator", "key_symptoms", "leak_terms", "source", "required_clues"):
                self.assertIn(key, p, f"{cid} missing {key}")
            for group in p["required_clues"]:
                self.assertTrue(group, f"{cid} has an empty clue group")
                for clue in group:  # a clue that is also a leak term would reject every case
                    self.assertFalse(any(t.lower() in clue for t in p["leak_terms"]), f"{cid}: {clue}")


if __name__ == "__main__":
    unittest.main()
