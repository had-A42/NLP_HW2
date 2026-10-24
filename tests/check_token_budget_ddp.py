"""Compare two-rank token-weighted training with the same merged batches on CPU.

Run explicitly: python tests/check_token_budget_ddp.py
Uses a file rendezvous to avoid the old macOS torchrun rendezvous crash.
"""

import datetime
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_training_resume import prepared_fixture, training_config

from src.model import CodeTransformer
from src.trainer.code_trainer import make_batch_sampler, run_training
from src.utils.code_runtime import load_checkpoint


def config(root, name):
    result = training_config(root / "data", root / name)
    result.model.dropout = 0
    result.model.layer_repeats = int(os.environ.get("HW2_TEST_LAYER_REPEATS", "1"))
    result.dataloader.num_workers = int(os.environ.get("HW2_TEST_NUM_WORKERS", "0"))
    return result


def double_model(**kwargs):
    # Isolate weighting correctness from Adam amplification of FP32 round-off
    # in mathematically zero attention-key-bias gradients.
    return CodeTransformer(**kwargs).double()


def rank_main(rank, root):
    root = Path(root)
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2")
    dist.init_process_group(
        "gloo",
        init_method=(root / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=60),
    )
    with patch("src.trainer.code_trainer.CodeTransformer", side_effect=double_model):
        run_training(config(root, "ddp"))


class MergedSampler:
    def __init__(self, data, settings, seed):
        self.ranks = [
            make_batch_sampler(data, settings, rank, 2, seed) for rank in range(2)
        ]

    def __len__(self):
        return len(self.ranks[0])

    def set_epoch(self, epoch, start_batch=0):
        for sampler in self.ranks:
            sampler.set_epoch(epoch, start_batch)

    def __iter__(self):
        for left, right in zip(*self.ranks):
            yield left + right


def merged_sampler(data, settings, rank, world_size, seed, shuffle=True):
    if shuffle:
        return MergedSampler(data, settings, seed)
    return make_batch_sampler(data, settings, rank, world_size, seed, shuffle=False)


def main():
    with tempfile.TemporaryDirectory(prefix="hw2-dynamic-ddp-") as temporary:
        root = Path(temporary)
        prepared_fixture(root / "data")
        mp.spawn(rank_main, args=(str(root),), nprocs=2, join=True)
        with patch(
            "src.trainer.code_trainer.make_batch_sampler", side_effect=merged_sampler
        ), patch("src.trainer.code_trainer.CodeTransformer", side_effect=double_model):
            merged_config = config(root, "merged")
            # Reference has no distributed workers. Spawn is exercised in both
            # DDP ranks; keep the single-process numerical reference synchronous.
            merged_config.dataloader.num_workers = 0
            run_training(merged_config)
        distributed = load_checkpoint(root / "ddp/last.pt")
        merged = load_checkpoint(root / "merged/last.pt")
        assert distributed["tokens_seen"] == merged["tokens_seen"]
        maximum = 0.0
        for key, weights in distributed["state_dict"].items():
            expected = merged["state_dict"][key]
            torch.testing.assert_close(weights, expected, rtol=1e-10, atol=1e-10)
            maximum = max(maximum, float((weights - expected).abs().max()))
        print(
            f"DDP token weighting passed: tokens={merged['tokens_seen']}, max_weight_difference={maximum}"
        )


if __name__ == "__main__":
    main()
