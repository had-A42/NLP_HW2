import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.utils.inference_sweep import run_suite, validate_plan


class InferenceSweepTests(unittest.TestCase):
    def test_propagates_beams_checks_format_timing_and_repeats(self):
        plan = {
            "repeats": 2,
            "default_tag": "beam2",
            "variants": [
                {
                    "tag": "beam2",
                    "num_beams": 2,
                    "max_new_tokens": 128,
                    "batch_size": 64,
                }
            ],
        }
        # The matrix file explicitly sets alpha; test a copied complete variant.
        plan["variants"][0]["length_penalty"] = 0.6
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0")
            self.assertIn("num_beams=2", command)
            options = dict(arg.split("=", 1) for arg in command if "=" in arg)
            Path(options["output_path"]).write_text(
                '{"code": "def solve(): return 1"}\n' * 1919
            )
            Path(options["report_path"]).write_text(
                json.dumps(
                    {
                        "passes_required_benchmark": True,
                        "syntax_parse_rate": 1,
                        "without_eos": 0,
                    }
                )
            )

        with tempfile.TemporaryDirectory() as root, patch(
            "src.utils.inference_sweep.subprocess.run", side_effect=fake_run
        ), patch(
            "src.utils.inference_sweep.time.perf_counter", side_effect=[0, 10, 20, 201]
        ):
            summary = run_suite("model.pt", "prepared", Path(root) / "sweep", plan)
        self.assertEqual(len(calls), 2)
        self.assertFalse(summary["default_passes"])
        self.assertFalse(summary["quality_measured"])
        self.assertTrue(summary["variants"][0]["identical_predictions"])

    def test_rejects_unsafe_or_duplicate_tags(self):
        for tag in ["../escape", "", "bad/tag"]:
            with self.assertRaises(ValueError):
                validate_plan({"default_tag": tag, "variants": [{"tag": tag}]})
