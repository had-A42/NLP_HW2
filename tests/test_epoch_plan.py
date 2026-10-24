import unittest

from omegaconf import OmegaConf

from src.trainer.code_trainer import training_plan, wall_time_limit


class VariableSampler:
    def __init__(self):
        self.epoch, self.start_batch = 1, 2

    def set_epoch(self, epoch, start_batch=0):
        self.epoch, self.start_batch = epoch, start_batch

    def __len__(self):
        return [5, 6, 7][self.epoch] - self.start_batch


class EpochPlanTests(unittest.TestCase):
    def test_disabled_wall_limit_and_invalid_values(self):
        import math

        self.assertEqual(wall_time_limit(None), math.inf)
        self.assertEqual(wall_time_limit(2), 7200)
        for value in [0, -1, True, math.nan, math.inf]:
            with self.assertRaises(ValueError):
                wall_time_limit(value)

    def test_real_epoch_lengths_and_partial_accumulation(self):
        sampler = VariableSampler()
        plan = training_plan(
            OmegaConf.create({"max_steps": None, "max_epochs": 3}), sampler, 2
        )
        self.assertEqual(plan["optimizer_step_limit"], 10)
        self.assertEqual(plan["microbatches_per_epoch_per_rank"], [5, 6, 7])
        self.assertEqual((sampler.epoch, sampler.start_batch), (1, 2))
        plan = training_plan(
            OmegaConf.create({"max_steps": 4, "max_epochs": 3}), sampler, 2
        )
        self.assertEqual(plan["optimizer_step_limit"], 4)

    def test_missing_or_invalid_limit(self):
        for settings in [
            {"max_steps": None},
            {"max_epochs": 0},
            {"max_steps": True},
            {"max_epochs": 1.5},
        ]:
            with self.assertRaises(ValueError):
                training_plan(OmegaConf.create(settings), VariableSampler(), 2)
