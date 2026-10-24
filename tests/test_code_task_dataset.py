"""Behavior checks using tiny local Arrow-cache fixtures, never training data."""

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader

from src.datasets import CodeTaskDataset


class CodeTaskDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.data_dir = Path(self.temporary.name)
        self.cache = (
            self.data_dir / "cache" / CodeTaskDataset.CACHE_NAME / "default"
            / "0.0.0" / CodeTaskDataset.DEFAULT_REVISION
        )
        self.cache.mkdir(parents=True)
        self.rows = [
            {"question": "First?", "code": "def solve():\n    return 1\n"},
            {"question": "Second?", "code": "def solve():\n    return 2\n"},
            {"question": "Third?", "code": "def solve():\n    return 3\n"},
        ]
        self.write_arrow("tinysms_qa_12m-train-00000-of-00002.arrow", self.rows[:2])
        self.write_arrow("tinysms_qa_12m-train-00001-of-00002.arrow", self.rows[2:])
        self.write_arrow("tinysms_qa_12m-test.arrow", [{"question": "Test?", "code": None}])
        self.metadata = {
            "splits": {
                "train": {"num_examples": 3, "shard_lengths": [2, 1]},
                "test": {"num_examples": 1},
            }
        }
        self.write_metadata()

    def write_arrow(self, filename, rows):
        table = HFDataset.from_list(rows).data.table
        with pa.OSFile(str(self.cache / filename), "wb") as sink:
            with pa.ipc.new_stream(sink, table.schema) as writer:
                writer.write_table(table)

    def write_metadata(self):
        (self.cache / "dataset_info.json").write_text(json.dumps(self.metadata))

    def dataset(self, **kwargs):
        return CodeTaskDataset(data_dir=self.data_dir, **kwargs)

    def test_shard_order_and_text_preservation(self):
        dataset = self.dataset()
        self.assertEqual(len(dataset), 3)
        for row_id, row in enumerate(self.rows):
            self.assertEqual(dataset[row_id], {"row_id": row_id, **row})

    def test_first_n_limit_and_index_bounds(self):
        dataset = self.dataset(limit=2)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[-1]["row_id"], 1)
        for idx in [2, -3]:
            with self.assertRaises(IndexError):
                dataset[idx]
        self.assertEqual(len(self.dataset(limit=100)), 3)
        empty = self.dataset(limit=0)
        self.assertEqual(len(empty), 0)
        with self.assertRaises(IndexError):
            empty[0]

    def test_test_has_no_code_target(self):
        dataset = self.dataset(split="test")
        self.assertEqual(dataset[0], {"row_id": 0, "question": "Test?"})
        batch = next(iter(DataLoader(dataset, batch_size=1)))
        self.assertEqual(batch["question"], ["Test?"])
        self.assertNotIn("code", batch)

    def test_transforms_do_not_modify_source(self):
        dataset = self.dataset(instance_transforms={"question": str.upper})
        self.assertEqual(dataset[0]["question"], "FIRST?")
        self.assertEqual(dataset.dataset[0]["question"], "First?")
        self.assertEqual(dataset[0]["row_id"], 0)

    def test_invalid_arguments(self):
        for limit in [True, 1.5, "2"]:
            with self.assertRaises(TypeError):
                self.dataset(limit=limit)
        with self.assertRaises(ValueError):
            self.dataset(limit=-1)
        with self.assertRaises(ValueError):
            self.dataset(split="val")
        with self.assertRaises(ValueError):
            self.dataset(instance_transforms={"row_id": str})
        with self.assertRaises(ValueError):
            self.dataset(revision="main")

    def test_missing_cache_fails_without_network(self):
        with patch("socket.socket", side_effect=AssertionError("Network forbidden")):
            with self.assertRaisesRegex(FileNotFoundError, "never downloads"):
                CodeTaskDataset(data_dir=self.data_dir / "missing")

    def test_missing_shard_is_not_silently_skipped(self):
        (self.cache / "tinysms_qa_12m-train-00001-of-00002.arrow").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Incomplete local cache"):
            self.dataset()

    def test_wrong_row_count(self):
        self.metadata["splits"]["train"]["num_examples"] = 4
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, "row count"):
            self.dataset()

    def test_empty_text_reports_original_row_id(self):
        for field in ["question", "code"]:
            for value in [None, "", " \n\t"]:
                row = {**self.rows[2], field: value}
                self.write_arrow("tinysms_qa_12m-train-00001-of-00002.arrow", [row])
                with self.assertRaisesRegex(ValueError, f"{field!r}.*row_id=2"):
                    self.dataset()[2]

    def test_pickle_and_offline_reads(self):
        with patch("socket.socket", side_effect=AssertionError("Network forbidden")):
            dataset = self.dataset(limit=2)
            restored = pickle.loads(pickle.dumps(dataset))
            self.assertEqual(restored[1], {"row_id": 1, **self.rows[1]})


if __name__ == "__main__":
    unittest.main()
