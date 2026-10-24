"""Single-device greedy/beam inference and strict, ordered {code: ...} JSONL output."""

import ast
import json
import time
from pathlib import Path

import torch

from datasets import load_from_disk
from src.datasets.code_collate import CodeCollator
from src.model import CodeTransformer
from src.utils.code_runtime import (
    environment_report,
    load_checkpoint,
    prepared_artifacts,
)


def run_inference(config, process_started=None):
    started = time.perf_counter() if process_started is None else process_started
    if int(config.batch_size) < 1 or int(config.max_new_tokens) < 1:
        raise ValueError("batch_size and max_new_tokens must be positive.")
    if config.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("Inference device must be auto, cpu or cuda.")
    cuda = config.device != "cpu" and torch.cuda.is_available()
    if config.device == "cuda" and not cuda:
        raise RuntimeError("CUDA requested but unavailable.")
    device = torch.device("cuda:0" if cuda else "cpu")
    torch.set_num_threads(config.cpu_threads)
    t4 = cuda and "T4" in torch.cuda.get_device_name(0)
    if config.require_t4 and (not t4 or torch.cuda.device_count() != 1):
        raise RuntimeError(
            "Benchmark requires one visible NVIDIA T4; launch with CUDA_VISIBLE_DEVICES=0."
        )
    root, manifest, tokenizer = prepared_artifacts(config.prepared_dir)
    checkpoint = load_checkpoint(config.checkpoint)
    if checkpoint["tokenizer_sha256"] != tokenizer.sha256:
        raise ValueError("Checkpoint and tokenizer do not match.")
    model = CodeTransformer(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    data = load_from_disk(str(root / config.split))
    expected = int(manifest["splits"][config.split]["rows"])
    if len(data) != expected:
        raise ValueError("Inference split row count differs from the manifest.")
    if config.limit is not None:
        if config.limit < 1:
            raise ValueError("Inference limit must be positive.")
        data = data.select(range(min(int(config.limit), len(data))))
    if not len(data):
        raise ValueError("Inference dataset is empty.")
    rows = []
    raw_questions = "question" in data.column_names
    for batch in data.iter(batch_size=1000):
        encoded = (
            [
                ids + [tokenizer.eos_id]
                for ids in tokenizer.encode_batch(batch["question"])
            ]
            if raw_questions
            else batch["source_ids"]
        )
        for row_id, ids in zip(batch["row_id"], encoded):
            if len(ids) > model.max_source_length:
                raise ValueError(
                    f"Source row_id={row_id} exceeds model limit; no silent truncation."
                )
            rows.append({"row_id": int(row_id), "source_ids": ids})
    ids = [row["row_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate row IDs in inference input.")
    if config.split == "test" and sorted(ids) != list(range(len(data))):
        raise ValueError("Test row IDs must cover the original contiguous input order.")
    if config.sort_by_length:
        rows.sort(key=lambda row: len(row["source_ids"]))
    collate = CodeCollator(tokenizer.pad_id, tokenizer.bos_id)
    if cuda:
        torch.cuda.synchronize()
    generation_started = time.perf_counter()
    answers, generated_lengths, without_eos = {}, [], 0
    for start in range(0, len(rows), int(config.batch_size)):
        items = rows[start : start + int(config.batch_size)]
        batch = collate(items)
        with torch.autocast(
            device.type, dtype=torch.float16, enabled=bool(cuda and config.amp)
        ):
            output = (
                model.generate(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    max_new_tokens=int(config.max_new_tokens),
                    cache_type=config.cache_type,
                    compile_step=config.compile_step,
                    eos_check_interval=config.eos_check_interval,
                    num_beams=config.num_beams,
                    length_penalty=config.length_penalty,
                    suppress_tokens=(tokenizer.unk_id,),
                )
                .cpu()
                .tolist()
            )
        for item, generated in zip(items, output):
            if tokenizer.eos_id not in generated:
                without_eos += 1
            generated_lengths.append(
                next(
                    (
                        i + 1
                        for i, token in enumerate(generated)
                        if token == tokenizer.eos_id
                    ),
                    len(generated),
                )
            )
            answers[item["row_id"]] = tokenizer.decode(generated)
    if cuda:
        torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - generation_started
    destination = Path(config.output_path).expanduser().resolve()
    report_path = Path(config.report_path).expanduser().resolve()
    if destination == report_path or destination == Path(config.checkpoint).resolve():
        raise ValueError("Prediction/report/checkpoint paths must be different.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w") as stream:
        for row_id in sorted(ids):
            stream.write(
                json.dumps({"code": answers[row_id]}, ensure_ascii=False) + "\n"
            )
    temporary.replace(destination)
    full_seconds = time.perf_counter() - started
    parsed, with_solve = 0, 0
    for code in answers.values():
        try:
            tree = ast.parse(code)
            parsed += 1
            with_solve += int(
                any(
                    isinstance(node, ast.FunctionDef) and node.name == "solve"
                    for node in tree.body
                )
            )
        except SyntaxError:
            pass
    full_test = config.split == "test" and len(data) == expected and expected == 1919
    eligible = bool(t4 and torch.cuda.device_count() == 1 and full_test)
    report = {
        **environment_report(device),
        "visible_cuda_devices": torch.cuda.device_count(),
        "examples": len(data),
        "expected_split_examples": expected,
        "full_test": full_test,
        "tokenized_raw_questions_inside_timer": raw_questions,
        "full_seconds": full_seconds,
        "generation_seconds": generation_seconds,
        "max_seconds": config.max_seconds,
        "within_time_budget": full_seconds <= config.max_seconds,
        "eligible_single_t4_benchmark": eligible,
        "official_limit_seconds": 180,
        "passes_required_benchmark": eligible and full_seconds <= 180,
        "dtype": "fp16_autocast" if cuda and config.amp else "fp32",
        "batch_size": config.batch_size,
        "max_new_tokens": config.max_new_tokens,
        "cache_type": config.cache_type,
        "compile_step": config.compile_step,
        "sort_by_length": config.sort_by_length,
        "without_eos": without_eos,
        "num_beams": config.num_beams,
        "length_penalty": config.length_penalty,
        "generated_tokens": sum(generated_lengths),
        "syntax_parse_rate": parsed / len(data),
        "solve_function_rate": with_solve / len(data),
        "tokenizer_sha256": tokenizer.sha256,
        "checkpoint": str(Path(config.checkpoint).resolve()),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report
