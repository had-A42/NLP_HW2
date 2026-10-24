"""Compare decoding settings in fresh processes; never select using hidden labels."""

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def validate_plan(plan):
    repeats = plan.get("repeats", 2)
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("Inference repeats must be a positive integer.")
    variants = plan.get("variants", [])
    tags = set()
    for variant in variants:
        tag = variant.get("tag", "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", tag) or tag in tags:
            raise ValueError("Inference tags must be unique safe filenames.")
        tags.add(tag)
        for field in ["num_beams", "max_new_tokens", "batch_size"]:
            value = variant.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer.")
        if not math.isfinite(variant.get("length_penalty", 0.6)):
            raise ValueError("length_penalty must be finite.")
    if not variants or plan.get("default_tag") not in tags:
        raise ValueError("Set variants and a matching default_tag.")


def verify_submission(path, expected_rows=1919):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if len(rows) != expected_rows or any(
        not isinstance(row, dict)
        or set(row) != {"code"}
        or not isinstance(row["code"], str)
        for row in rows
    ):
        raise ValueError("Expected one code-only JSON object per test task.")


def run_suite(checkpoint, prepared_dir, output_dir, plan, timeout=210):
    validate_plan(plan)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[2]
    records = []
    repeats = plan.get("repeats", 2)
    for variant in plan["variants"]:
        for repeat in range(repeats):
            stem = output / f"{variant['tag']}-repeat-{repeat}"
            predictions, report = Path(f"{stem}.jsonl"), Path(f"{stem}.report.json")
            command = [
                sys.executable,
                str(project / "inference.py"),
                f"checkpoint={Path(checkpoint).resolve()}",
                f"prepared_dir={Path(prepared_dir).resolve()}",
                f"output_path={predictions}",
                f"report_path={report}",
                "device=cuda",
                "require_t4=true",
                "sort_by_length=true",
                "cache_type=static",
                "compile_step=false",
                *[
                    f"{key}={variant.get(key, 0.6)}"
                    for key in [
                        "num_beams",
                        "max_new_tokens",
                        "batch_size",
                        "length_penalty",
                    ]
                ],
            ]
            record = {**variant, "repeat": repeat, "passes": False}
            started = time.perf_counter()
            try:
                with Path(f"{stem}.stdout.log").open("w") as log:
                    subprocess.run(
                        command,
                        cwd=project,
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=timeout,
                        check=True,
                    )
                external_seconds = time.perf_counter() - started
                verify_submission(predictions)
                measured = json.loads(report.read_text())
                record.update(
                    predictions=predictions.name,
                    report=report.name,
                    seconds=external_seconds,
                    passes=bool(
                        measured["passes_required_benchmark"]
                        and external_seconds <= 180
                    ),
                    syntax_parse_rate=measured["syntax_parse_rate"],
                    without_eos=measured["without_eos"],
                    predictions_sha256=hashlib.sha256(
                        predictions.read_bytes()
                    ).hexdigest(),
                )
                measured["external_process_seconds"] = external_seconds
                measured["passes_required_benchmark"] = record["passes"]
                report.write_text(json.dumps(measured, indent=2))
            except (subprocess.SubprocessError, OSError, ValueError, KeyError) as error:
                record.update(
                    error_type=type(error).__name__,
                    seconds=time.perf_counter() - started,
                )
            records.append(record)
            (output / "runs.json").write_text(json.dumps(records, indent=2))
            print("INFERENCE_SWEEP " + json.dumps(record), flush=True)
    summaries = []
    for variant in plan["variants"]:
        runs = [r for r in records if r["tag"] == variant["tag"]]
        summaries.append(
            {
                **variant,
                "passes_all_repeats": len(runs) == repeats
                and all(r["passes"] for r in runs),
                "max_seconds": max(r["seconds"] for r in runs),
                "identical_predictions": len(
                    {r.get("predictions_sha256") for r in runs}
                )
                == 1
                and all("predictions_sha256" in r for r in runs),
                "runs": runs,
            }
        )
    default = next(v for v in summaries if v["tag"] == plan["default_tag"])
    summary = {
        "default_tag": plan["default_tag"],
        "default_passes": default["passes_all_repeats"],
        "quality_measured": False,
        "selection_policy": "Configured default; no ranking by unmeasured pass@1.",
        "variants": summaries,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
