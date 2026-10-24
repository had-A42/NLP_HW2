"""Offline access to the original HF Arrow cache; no downloads or text rewriting."""

import json
import re
from numbers import Integral
from operator import index as integer_index
from pathlib import Path

from torch.utils.data import Dataset

from datasets import Dataset as HFDataset
from datasets import concatenate_datasets
from src.utils.io_utils import ROOT_PATH


class CodeTaskDataset(Dataset):
    DEFAULT_REVISION = "b2724689ff079e9c22efd548fe134bfb29036dbb"
    CACHE_NAME = "SlavaYus___tinysms_qa_12m"
    DATASET_NAME = "tinysms_qa_12m"

    def __init__(
        self,
        data_dir=None,
        split="train",
        limit=None,
        revision=DEFAULT_REVISION,
        instance_transforms=None,
    ):
        if split not in {"train", "test"}:
            raise ValueError(
                "split must be train or test; validation is prepared separately."
            )
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, Integral)
        ):
            raise TypeError("limit must be a non-negative integer or None.")
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative.")
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("revision must be an exact dataset commit hash.")
        self.data_dir = (
            Path(data_dir if data_dir is not None else ROOT_PATH.parent / "data")
            .expanduser()
            .resolve()
        )
        self.split, self.revision = split, revision
        self.instance_transforms = instance_transforms
        fields = {"question", "code"} if split == "train" else {"question"}
        if instance_transforms and set(instance_transforms) - fields:
            raise ValueError(
                "Unsupported transform fields; row_id cannot be transformed."
            )
        root = self.data_dir / "cache" / self.CACHE_NAME / "default"
        matches = list(root.glob(f"*/{revision}/dataset_info.json"))
        if not matches:
            raise FileNotFoundError(
                f"Local prepared cache missing under {root}. CodeTaskDataset never downloads data."
            )
        if len(matches) != 1:
            raise ValueError(
                "Multiple cached versions found; use an unambiguous data_dir."
            )
        self.cache_path = matches[0].parent
        info = json.loads(matches[0].read_text()).get("splits", {}).get(split)
        if info is None:
            raise ValueError(f"Missing split {split!r} in cache metadata.")
        lengths = info.get("shard_lengths")
        prefix = f"{self.DATASET_NAME}-{split}"
        names = (
            [
                f"{prefix}-{i:05d}-of-{len(lengths):05d}.arrow"
                for i in range(len(lengths))
            ]
            if lengths
            else [f"{prefix}.arrow"]
        )
        for name in names:
            if not (self.cache_path / name).is_file():
                raise FileNotFoundError(
                    f"Incomplete local cache: missing {self.cache_path / name}"
                )
        parts = [
            HFDataset.from_file(str(self.cache_path / name), in_memory=False)
            for name in names
        ]
        if lengths and [len(p) for p in parts] != lengths:
            raise ValueError("Shard row counts do not match metadata.")
        self.dataset = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
        if len(self.dataset) != info["num_examples"]:
            raise ValueError("Split row count does not match metadata.")
        if fields - set(self.dataset.column_names):
            raise ValueError(
                f"Missing columns: {fields - set(self.dataset.column_names)}"
            )
        self._length = (
            len(self.dataset) if limit is None else min(int(limit), len(self.dataset))
        )

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        row_id = integer_index(idx)
        if row_id < 0:
            row_id += len(self)
        if not 0 <= row_id < len(self):
            raise IndexError(idx)
        row = self.dataset[row_id]
        item = {"row_id": row_id, "question": row["question"]}
        if self.split == "train":
            item["code"] = row["code"]
        for field in ("question", "code"):
            if field in item and (
                not isinstance(item[field], str) or not item[field].strip()
            ):
                raise ValueError(
                    f"Invalid {field!r} in split {self.split!r}, row_id={row_id}: expected non-empty text."
                )
        for field, transform in (self.instance_transforms or {}).items():
            item[field] = transform(item[field])
        return item
