"""Offline, group-disjoint preparation; no new examples or target programs."""

import hashlib
import json
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from src.datasets.code_task_dataset import CodeTaskDataset
from src.tokenization import CodeTokenizer


def is_validation(question, fraction, seed=42):
    digest = hashlib.blake2b(f"{seed}\0{question}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") < int(fraction * 2**64)


@lru_cache(maxsize=4)
def load_tokenizer(path):
    return CodeTokenizer(path)


def encode_training_batch(
    batch,
    indices,
    tokenizer_path,
    max_source_length,
    max_target_length,
    validation_fraction,
    seed,
):
    tokenizer = load_tokenizer(tokenizer_path)
    questions, codes = batch["question"], batch["code"]
    row_ids = batch.get("row_id", indices)
    for row_id, question, code in zip(row_ids, questions, codes):
        for name, value in [("question", question), ("code", code)]:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Invalid {name!r}, row_id={row_id}; expected non-empty text."
                )
    source = [ids + [tokenizer.eos_id] for ids in tokenizer.encode_batch(questions)]
    targets = [ids + [tokenizer.eos_id] for ids in tokenizer.encode_batch(codes)]
    return {
        "row_id": row_ids,
        "source_ids": source,
        "target_ids": targets,
        "source_length": [len(ids) for ids in source],
        "target_length": [len(ids) for ids in targets],
        "is_validation": [
            is_validation(q, validation_fraction, seed) for q in questions
        ],
        "too_long": [
            len(s) > max_source_length or len(t) > max_target_length
            for s, t in zip(source, targets)
        ],
    }


def deduplicate_exact_pairs(source):
    """Keep first occurrence using 128-bit fingerprints, not a set of full texts."""
    fingerprints = np.empty(len(source), dtype="V16")
    offset = 0
    for batch in source.iter(batch_size=10_000):
        for question, code in zip(batch["question"], batch["code"]):
            if (
                not isinstance(question, str)
                or not question.strip()
                or not isinstance(code, str)
                or not code.strip()
            ):
                raise ValueError(f"Invalid question/code, row_id={offset}.")
            q, c = question.encode(), code.encode()
            fingerprints[offset] = hashlib.blake2b(
                len(q).to_bytes(8, "big") + q + c, digest_size=16
            ).digest()
            offset += 1
    _, keep = np.unique(fingerprints, return_index=True)
    return source.select(np.sort(keep))


def prepare(config):
    fraction = float(config.validation_fraction)
    if not 0 <= fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1).")
    output = Path(config.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prepared data: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = CodeTaskDataset(
        config.data_dir, limit=config.source_limit, revision=config.revision
    )
    if not len(raw):
        raise ValueError("Training source is empty.")
    source = raw.dataset.select(range(len(raw)))
    source = source.add_column("row_id", np.arange(len(source), dtype=np.int64))
    original_rows = len(source)
    if config.deduplicate:
        source = deduplicate_exact_pairs(source)
    workers = int(config.num_proc) if int(config.num_proc) > 1 else None
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}-", dir=output.parent
    ) as temporary:
        staging = Path(temporary) / "result"
        staging.mkdir()
        tokenizer_path = staging / "tokenizer.json"
        rng = np.random.default_rng(config.seed)
        indices = np.sort(
            rng.choice(
                len(source),
                size=min(len(source), config.tokenizer.train_limit),
                replace=False,
            )
        )
        sample = source.select(indices)

        def training_texts():
            for row in sample:
                if (
                    not isinstance(row["question"], str)
                    or not row["question"].strip()
                    or not isinstance(row["code"], str)
                    or not row["code"].strip()
                ):
                    raise ValueError(f"Invalid question/code, row_id={row['row_id']}.")
                if not is_validation(row["question"], fraction, config.seed):
                    yield row["question"]
                    yield row["code"]

        tokenizer = CodeTokenizer.train(
            training_texts(),
            tokenizer_path,
            config.tokenizer.vocab_size,
            config.tokenizer.min_frequency,
        )
        encoded = source.map(
            encode_training_batch,
            batched=True,
            with_indices=True,
            batch_size=1000,
            num_proc=workers,
            remove_columns=source.column_names,
            cache_file_name=str(Path(temporary) / "encoded.arrow"),
            fn_kwargs={
                "tokenizer_path": str(tokenizer_path),
                "max_source_length": config.max_source_length,
                "max_target_length": config.max_target_length,
                "validation_fraction": fraction,
                "seed": config.seed,
            },
            desc="Tokenize original train",
        )
        splits = {}
        for name, desired in [("train", False), ("val", True)]:
            splits[name] = encoded.filter(
                lambda bad, val: not bad and val == desired,
                input_columns=["too_long", "is_validation"],
                num_proc=workers,
                cache_file_name=str(Path(temporary) / f"{name}-indices.arrow"),
                load_from_cache_file=False,
                desc=f"Select {name}",
            ).remove_columns(["too_long", "is_validation"])
        if not len(splits["train"]):
            raise ValueError("No training rows remain after preparation.")
        if fraction and not len(splits["val"]):
            raise ValueError(
                "Validation is empty; increase source_limit or use validation_fraction=0 for a smoke test."
            )
        test = CodeTaskDataset(config.data_dir, split="test", revision=config.revision)
        test_source = test.dataset.select_columns(["question"])
        # Keep raw test questions: inference tokenizes them inside the full timer.
        for i, question in enumerate(test_source["question"]):
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Invalid test question, row_id={i}.")
            if len(tokenizer.encode(question)) + 1 > config.max_source_length:
                raise ValueError(
                    f"Test question row_id={i} exceeds max_source_length; increase the limit."
                )
        splits["test"] = test_source.add_column(
            "row_id", np.arange(len(test), dtype=np.int64)
        )
        stats = {}
        for name, data in splits.items():
            data.save_to_disk(str(staging / name))
            stats[name] = {"rows": len(data)}
            if name != "test" and len(data):
                for field in ["source_length", "target_length"]:
                    # filter/select may keep an indices table; .data is physical
                    # storage and would include rows belonging to the other split.
                    lengths = np.fromiter(data[field], dtype=np.int64, count=len(data))
                    stats[name][field] = {
                        "sum": int(lengths.sum()),
                        "quantiles": np.quantile(
                            lengths, [0, 0.5, 0.9, 0.95, 0.99, 1]
                        ).tolist(),
                    }
        manifest = {
            "format_version": 1,
            "revision": config.revision,
            "tokenizer_sha256": tokenizer.sha256,
            "vocab_size": tokenizer.vocab_size,
            "pad_id": tokenizer.pad_id,
            "bos_id": tokenizer.bos_id,
            "eos_id": tokenizer.eos_id,
            "validation_fraction": fraction,
            "seed": config.seed,
            "max_source_length": config.max_source_length,
            "max_target_length": config.max_target_length,
            "source_rows": original_rows,
            "exact_pair_duplicates_removed": original_rows - len(source),
            "too_long_rows_dropped": len(source)
            - len(splits["train"])
            - len(splits["val"]),
            "splits": stats,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2)
        )
        OmegaConf.save(config, staging / "prepare_config.yaml", resolve=True)
        staging.rename(output)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return output
