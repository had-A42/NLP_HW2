import unittest

import numpy as np
from omegaconf import OmegaConf

from datasets import Dataset
from src.datasets.code_sampler import TokenBudgetBatchSampler
from src.trainer.code_trainer import make_batch_sampler


class TokenBudgetSamplerTests(unittest.TestCase):
    def test_budget_ddp_coverage_and_equal_steps(self):
        rng = np.random.default_rng(123)
        source = rng.integers(5, 101, 257)
        target = rng.integers(8, 257, 257)
        for world in [1, 2, 3, 7]:
            samplers = [
                TokenBudgetBatchSampler(
                    source,
                    target,
                    token_budget=1024,
                    max_batch_size=13,
                    bucket_size=17,
                    rank=rank,
                    world_size=world,
                )
                for rank in range(world)
            ]
            batches = [list(s) for s in samplers]
            self.assertEqual(len({len(b) for b in batches}), 1)
            for step in zip(*batches):
                self.assertEqual(len({len(b) for b in step}), 1)
            seen = []
            for sampler, rank_batches in zip(samplers, batches):
                self.assertEqual(len(sampler), len(rank_batches))
                for batch in rank_batches:
                    self.assertLessEqual(len(batch), 13)
                    cost = len(batch) * (source[batch].max() + target[batch].max())
                    self.assertLessEqual(cost, 1024)
                    seen.extend(batch)
            expected = np.random.default_rng(42).permutation(257)
            expected = expected[: 257 - 257 % world]
            self.assertEqual(sorted(seen), sorted(expected.tolist()))

    def test_exact_budget_singletons_and_variable_batches(self):
        source = np.array([2] * 12 + [7] * 12)
        target = np.array([2] * 12 + [5] * 12)
        sampler = TokenBudgetBatchSampler(
            source,
            target,
            token_budget=12,
            max_batch_size=10,
            bucket_size=24,
            shuffle=False,
        )
        batches = list(sampler)
        self.assertEqual([len(b) for b in batches], [3] * 4 + [1] * 12)
        self.assertEqual(sorted(i for b in batches for i in b), list(range(24)))

    def test_epoch_resume_and_cached_length(self):
        source = np.arange(103) % 17 + 1
        target = np.arange(103) % 29 + 1
        sampler = TokenBudgetBatchSampler(
            source, target, token_budget=128, bucket_size=20, world_size=2
        )
        first = list(sampler)
        self.assertEqual(first, list(sampler))
        plan = sampler._plan
        self.assertEqual(len(sampler), len(first))
        self.assertIs(plan, sampler._plan)
        sampler.set_epoch(0, 3)
        self.assertEqual(list(sampler), first[3:])
        self.assertEqual(len(sampler), len(first) - 3)
        sampler.set_epoch(1)
        second = list(sampler)
        self.assertNotEqual(first, second)
        sampler.set_epoch(0)
        self.assertEqual(first, list(sampler))

    def test_input_errors_and_small_ddp_tail(self):
        with self.assertRaisesRegex(ValueError, "Row index 0"):
            TokenBudgetBatchSampler([8], [9], token_budget=16)
        for source, target in [([0], [1]), ([1], [-1]), ([1, 2], [1]), ([1.5], [2])]:
            with self.assertRaises(ValueError):
                TokenBudgetBatchSampler(source, target)
        for options in [
            dict(token_budget=0),
            dict(max_batch_size=0),
            dict(bucket_size=0),
            dict(world_size=0),
            dict(rank=2, world_size=2),
        ]:
            with self.assertRaises(ValueError):
                TokenBudgetBatchSampler([1], [1], **options)
        sampler = TokenBudgetBatchSampler([1], [1], world_size=2)
        self.assertEqual(len(sampler), 0)
        self.assertEqual(list(sampler), [])

    def test_logical_hf_indices_and_validation_tail(self):
        data = Dataset.from_dict(
            dict(source_length=[10, 1000, 20, 30], target_length=[10, 1000, 20, 30])
        )
        selected = data.select([3, 0, 2])
        config = OmegaConf.create(
            dict(batching="tokens", token_budget=80, max_batch_size=4, bucket_size=10)
        )
        sampler = make_batch_sampler(selected, config, 0, 1, 42, shuffle=False)
        self.assertEqual(sampler.source_lengths.tolist(), [30, 10, 20])
        self.assertEqual(sorted(i for b in sampler for i in b), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
