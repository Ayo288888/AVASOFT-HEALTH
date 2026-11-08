import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_label_map import DEFAULT, EXPLICIT, load_conditions, target_for


class TestLabelMap(unittest.TestCase):
    def test_conditions_file_has_60_unique_conditions(self):
        self.assertEqual(len(load_conditions()), 60)

    def test_every_mapped_target_is_a_real_condition(self):
        names = set(load_conditions()) | {"OTHER", "DROP"}
        for source, mapping in EXPLICIT.items():
            self.assertIn(source, DEFAULT, f"{source} has no default target")
            for label, target in mapping.items():
                self.assertIn(target, names, f"{source}:{label} -> {target}")

    def test_defaults_per_source(self):
        self.assertEqual(target_for("ddxplus", "Chagas"), "OTHER")
        self.assertEqual(target_for("kaggle773", "lipoma"), "OTHER")
        self.assertEqual(target_for("afrimedqa", "General_Medicine"), "RELABEL")
        self.assertEqual(target_for("symcat", "anything"), "DROP")
        self.assertEqual(target_for("kaggle773", "malaria"), "Malaria (uncomplicated)")


if __name__ == "__main__":
    unittest.main()
