"""A tiny training fixture verifies exact resume including dropout and data order."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_dir

from datasets import Dataset
from src.datasets.code_preparation import is_validation
from src.tokenization import CodeTokenizer
from src.trainer.code_trainer import run_training
from src.utils.code_runtime import load_checkpoint, save_checkpoint

PROJECT = Path(__file__).resolve().parent.parent


def prepared_fixture(root):
    root.mkdir()
    tokenizer = CodeTokenizer.train(
        iter(["fixture source", "def solve():\n    return 1\n"]),
        root / "tokenizer.json",
        300,
        1,
    )
    rows = [
        {
            "row_id": i,
            "source_ids": [4, 5, 2],
            "target_ids": [6] * (i % 3 + 1) + [2],
            "source_length": 3,
            "target_length": i % 3 + 2,
        }
        for i in range(12)
    ]
    Dataset.from_list(rows[:8]).save_to_disk(str(root / "train"))
    Dataset.from_list(rows[8:]).save_to_disk(str(root / "val"))
    Dataset.from_list([{"row_id": 0, "question": "fixture source"}]).save_to_disk(
        str(root / "test")
    )
    manifest = {
        "format_version": 1,
        "vocab_size": tokenizer.vocab_size,
        "pad_id": tokenizer.pad_id,
        "bos_id": tokenizer.bos_id,
        "eos_id": tokenizer.eos_id,
        "tokenizer_sha256": tokenizer.sha256,
        "max_source_length": 32,
        "max_target_length": 32,
        "splits": {"train": {"rows": 8}, "val": {"rows": 4}, "test": {"rows": 1}},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))


def training_config(prepared, output, resume=None):
    overrides = [
        f"prepared_dir={prepared}",
        f"run_dir={output}",
        "run_name=resume-fixture",
        "writer.mode=disabled",
        "trainer.device=cpu",
        "trainer.max_steps=2",
        "trainer.accumulation_steps=2",
        "trainer.log_every=1",
        "trainer.eval_every=1",
        "trainer.save_every=1",
        "dataloader.token_budget=12",
        "dataloader.max_batch_size=4",
        "dataloader.bucket_size=4",
        "dataloader.num_workers=0",
        "model.d_model=16",
        "model.n_heads=2",
        "model.encoder_layers=1",
        "model.decoder_layers=1",
        "model.ffn_dim=32",
        "model.dropout=0.1",
        "scheduler.warmup_steps=1",
    ]
    if resume:
        overrides.append(f"trainer.resume_from={resume}")
    with initialize_config_dir(str(PROJECT / "src/configs"), version_base=None):
        return compose(config_name="code_baseline", overrides=overrides)


class TrainingResumeTests(unittest.TestCase):
    def test_epoch_limit_and_resume_across_epochs(self):
        original = training_config

        def epoch_config(*args, **kwargs):
            config = original(*args, **kwargs)
            config.trainer.max_steps = None
            config.trainer.max_epochs = 2
            config.trainer.max_hours = None
            config.model.norm_style = "pre"
            return config

        with patch(f"{__name__}.training_config", side_effect=epoch_config):
            self.test_resume_replays_uninterrupted_weights()

    def test_resume_with_shared_layer_repeats(self):
        original = training_config

        def repeated_config(*args, **kwargs):
            config = original(*args, **kwargs)
            config.model.layer_repeats = 3
            return config

        with patch(f"{__name__}.training_config", side_effect=repeated_config):
            self.test_resume_replays_uninterrupted_weights()

    def test_resume_replays_uninterrupted_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared_fixture(root / "data")
            captured = root / "step1.pt"

            def save_and_capture(path, payload):
                save_checkpoint(path, payload)
                if Path(path).name == "last.pt" and payload["step"] == 1:
                    save_checkpoint(captured, payload)

            with patch(
                "src.trainer.code_trainer.save_checkpoint", side_effect=save_and_capture
            ):
                run_training(training_config(root / "data", root / "full"))
            run_training(training_config(root / "data", root / "resumed", captured))
            uninterrupted = load_checkpoint(root / "full/last.pt")
            resumed = load_checkpoint(root / "resumed/last.pt")
            plan = uninterrupted["training_plan"]
            self.assertEqual(uninterrupted["step"], plan["optimizer_step_limit"])
            if plan["max_epochs"] is not None:
                self.assertEqual(uninterrupted["epoch"], plan["max_epochs"] - 1)
                self.assertEqual(
                    uninterrupted["batch_in_epoch"],
                    plan["microbatches_per_epoch_per_rank"][-1],
                )
            self.assertEqual(uninterrupted["tokens_seen"], resumed["tokens_seen"])
            self.assertEqual(uninterrupted["batch_in_epoch"], resumed["batch_in_epoch"])
            for name, value in uninterrupted["state_dict"].items():
                torch.testing.assert_close(
                    value, resumed["state_dict"][name], rtol=0, atol=0
                )
            changed = training_config(root / "data", root / "changed", captured)
            changed.dataloader.token_budget = 13
            with self.assertRaisesRegex(ValueError, "matching batching config"):
                run_training(changed)
            changed_schedule = training_config(
                root / "data", root / "changed-plan", captured
            )
            changed_schedule.trainer.max_steps = 1
            with self.assertRaisesRegex(ValueError, "matching training plan"):
                run_training(changed_schedule)


if __name__ == "__main__":
    unittest.main()
