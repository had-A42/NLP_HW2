"""Deterministic global batches with optional local length bucketing and DDP slices."""

import math

import numpy as np
from torch.utils.data import Sampler


class CodeBatchSampler(Sampler):
    def __init__(
        self,
        lengths,
        batch_size,
        rank=0,
        world_size=1,
        seed=42,
        bucketed=False,
        bucket_batches=100,
        shuffle=True,
        drop_last=True,
    ):
        if (
            batch_size < 1
            or world_size < 1
            or not 0 <= rank < world_size
            or bucket_batches < 1
        ):
            raise ValueError("Invalid batch size or distributed rank.")
        if not drop_last and world_size > 1:
            raise ValueError("Non-dropping evaluation batches must use one rank.")
        self.lengths = np.asarray(lengths)
        self.batch_size, self.rank, self.world_size = batch_size, rank, world_size
        self.seed, self.bucketed, self.bucket_batches = seed, bucketed, bucket_batches
        self.shuffle, self.drop_last = shuffle, drop_last
        self.epoch, self.start_batch = 0, 0

    def set_epoch(self, epoch, start_batch=0):
        self.epoch, self.start_batch = epoch, start_batch

    def __len__(self):
        size = self.batch_size * self.world_size
        count = (
            len(self.lengths) // size
            if self.drop_last
            else math.ceil(len(self.lengths) / size)
        )
        return max(0, count - self.start_batch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        indices = (
            rng.permutation(len(self.lengths))
            if self.shuffle
            else np.arange(len(self.lengths))
        )
        global_size = self.batch_size * self.world_size
        if self.bucketed:
            pool = global_size * self.bucket_batches
            for start in range(0, len(indices), pool):
                block = indices[start : start + pool]
                indices[start : start + pool] = block[
                    np.argsort(self.lengths[block], kind="stable")
                ]
        total = (
            len(indices) // global_size
            if self.drop_last
            else math.ceil(len(indices) / global_size)
        )
        order = (
            rng.permutation(total)
            if self.shuffle and self.bucketed
            else np.arange(total)
        )
        for batch_idx in order[self.start_batch :]:
            start = int(batch_idx) * global_size + self.rank * self.batch_size
            batch = indices[start : start + self.batch_size]
            if len(batch):
                yield batch.tolist()


class TokenBudgetBatchSampler(Sampler):
    """Length-bucketed batches bounded by padded source+target tokens per rank.

    Every rank builds the same compact epoch plan. Sorted examples are dealt
    round-robin across ranks, so each rank gets the same number of examples
    and steps without sharing rows. At most world_size-1 rows are dropped.
    """

    def __init__(
        self,
        source_lengths,
        target_lengths,
        token_budget=16384,
        max_batch_size=256,
        bucket_size=3200,
        rank=0,
        world_size=1,
        seed=42,
        shuffle=True,
    ):
        self.source_lengths = np.asarray(source_lengths)
        self.target_lengths = np.asarray(target_lengths)
        if (
            self.source_lengths.ndim != 1
            or self.target_lengths.shape != self.source_lengths.shape
            or self.source_lengths.dtype.kind not in "iu"
            or self.target_lengths.dtype.kind not in "iu"
            or np.any(self.source_lengths <= 0)
            or np.any(self.target_lengths <= 0)
        ):
            raise ValueError(
                "Source/target lengths must be matching positive integer arrays."
            )
        if (
            min(token_budget, max_batch_size, bucket_size, world_size) < 1
            or not 0 <= rank < world_size
        ):
            raise ValueError(
                "Token budget, batch/bucket size and distributed rank are invalid."
            )
        self.source_lengths = self.source_lengths.astype(np.int64, copy=False)
        self.target_lengths = self.target_lengths.astype(np.int64, copy=False)
        self.lengths = self.source_lengths + self.target_lengths
        too_long = np.flatnonzero(self.lengths > token_budget)
        if len(too_long):
            index = int(too_long[0])
            raise ValueError(
                f"Row index {index} needs {self.lengths[index]} tokens, "
                f"exceeding token_budget={token_budget}; rows are never truncated."
            )
        self.token_budget = token_budget
        self.max_batch_size, self.bucket_size = max_batch_size, bucket_size
        self.rank, self.world_size = rank, world_size
        self.seed, self.shuffle = seed, shuffle
        self.epoch, self.start_batch = 0, 0
        self._plan = None

    def set_epoch(self, epoch, start_batch=0):
        if epoch < 0 or start_batch < 0:
            raise ValueError("Epoch and start_batch must be nonnegative.")
        if epoch != self.epoch:
            self._plan = None
        self.epoch, self.start_batch = epoch, start_batch

    def _epoch_plan(self):
        if self._plan is not None:
            return self._plan
        rng = np.random.default_rng(self.seed + self.epoch)
        indices = (
            rng.permutation(len(self.lengths))
            if self.shuffle
            else np.arange(len(self.lengths))
        )
        indices = indices[: len(indices) - len(indices) % self.world_size]
        pool = self.bucket_size * self.world_size
        bounds = []
        for start in range(0, len(indices), pool):
            end = min(start + pool, len(indices))
            block = indices[start:end]
            indices[start:end] = block[np.argsort(self.lengths[block], kind="stable")]
            cursor = start
            while cursor < end:
                largest = min(self.max_batch_size, (end - cursor) // self.world_size)
                trial = indices[cursor : cursor + largest * self.world_size].reshape(
                    largest, self.world_size
                )
                source_max = np.maximum.accumulate(self.source_lengths[trial], axis=0)
                target_max = np.maximum.accumulate(self.target_lengths[trial], axis=0)
                cost = (source_max + target_max) * np.arange(1, largest + 1)[:, None]
                size = int(np.count_nonzero(np.all(cost <= self.token_budget, axis=1)))
                stop = cursor + size * self.world_size
                bounds.append((cursor, stop))
                cursor = stop
        bounds = np.asarray(bounds, dtype=np.int64).reshape(-1, 2)
        order = rng.permutation(len(bounds)) if self.shuffle else np.arange(len(bounds))
        self._plan = indices, bounds, order
        return self._plan

    def __len__(self):
        _, bounds, _ = self._epoch_plan()
        return max(0, len(bounds) - self.start_batch)

    def __iter__(self):
        indices, bounds, order = self._epoch_plan()
        if self.start_batch > len(bounds):
            raise ValueError("start_batch exceeds the number of batches in this epoch.")
        for position in order[self.start_batch :]:
            start, stop = bounds[position]
            yield indices[start + self.rank : stop : self.world_size].tolist()
