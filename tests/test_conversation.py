import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main


def history(*pairs):
    return json.dumps([{"role": r, "content": c} for r, c in pairs])


class TestParseConversation(unittest.TestCase):
    def test_plain_text_is_single_user_message(self):
        self.assertEqual(
            main.parse_conversation("I have a headache"),
            [{"role": "user", "content": "I have a headache"}],
        )

    def test_drops_greeting_maps_ai_role_and_strips_html(self):
        raw = history(
            ("assistant", "Welcome, describe your symptoms."),
            ("user", "Fever and chills"),
            ("ai", "<h4>Assessment</h4><p>Could be malaria.</p>"),
            ("user", "Should I get tested?"),
        )
        self.assertEqual(main.parse_conversation(raw), [
            {"role": "user", "content": "Fever and chills"},
            {"role": "assistant", "content": "Assessment Could be malaria."},
            {"role": "user", "content": "Should I get tested?"},
        ])

    def test_history_is_capped(self):
        turns = [("user", f"msg {i}") if i % 2 == 0 else ("ai", f"reply {i}") for i in range(60)]
        parsed = main.parse_conversation(history(*turns))
        self.assertLessEqual(len(parsed), main.MAX_HISTORY_MESSAGES)
        self.assertEqual(parsed[-1]["content"], "reply 59")


@mock.patch.object(main, "client", object())
class TestPerformTriage(unittest.TestCase):
    @mock.patch.object(main, "generate_llm_response", return_value="<h4>Assessment</h4>")
    @mock.patch.object(main, "query_huggingface_model",
                       return_value=[{"condition": "Malaria", "confidence": "80.0%"}])
    def test_first_turn_uses_classifier_and_template(self, hf, llm):
        preds, note = main.perform_triage(history(("assistant", "Welcome"), ("user", "Fever and chills")))
        hf.assert_called_once()
        self.assertEqual(preds[0]["condition"], "Malaria")
        prompt = llm.call_args.args[0][-1]["content"]
        self.assertIn("<h4>", prompt)
        self.assertIn("follow-up question", prompt)

    @mock.patch.object(main, "generate_llm_response", return_value="Yes, get a malaria test.")
    @mock.patch.object(main, "query_huggingface_model")
    def test_follow_up_is_conversational_with_history(self, hf, llm):
        raw = history(("user", "Fever and chills"), ("ai", "<p>Could be malaria.</p>"),
                      ("user", "Should I get tested?"))
        preds, note = main.perform_triage(raw)
        hf.assert_not_called()
        self.assertEqual(preds, [])
        self.assertEqual(note, "Yes, get a malaria test.")
        messages = llm.call_args.args[0]
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "user"])
        self.assertIn("question", llm.call_args.kwargs["system"])


if __name__ == "__main__":
    unittest.main()
