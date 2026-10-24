"""Export a linked training checkpoint and generate ordered JSONL on one T4."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path


def verify_jsonl(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != 1919 or any(
        set(row) != {"code"} or not isinstance(row["code"], str) for row in rows
    ):
        raise ValueError(
            "Submission must contain exactly 1919 objects with only a string code field."
        )


def main(input_root=None, output_root=None):
    started = time.monotonic()
    inputs = Path(input_root or "/kaggle/input")
    working = Path(output_root or "/kaggle/working")
    checkpoints = list(inputs.rglob("best.pt")) or list(inputs.rglob("last.pt"))
    if len(checkpoints) != 1:
        raise RuntimeError(f"Expected one linked checkpoint, found {len(checkpoints)}.")
    provenances = list(inputs.rglob("code_provenance.json"))
    if len(provenances) != 1:
        raise RuntimeError("Attach the outputs of exactly one training version.")
    provenance = json.loads(provenances[0].read_text())
    sources = [
        p.parent
        for p in inputs.rglob("bundle_manifest.json")
        if json.loads(p.read_text()) == provenance
    ]
    if not sources:
        raise RuntimeError("Training output lacks its matching source bundle.")
    source = sources[0]
    project = working / "inference_project"
    project.mkdir(exist_ok=False)
    for relative, digest in provenance["sha256"].items():
        target = project / relative
        if not target.resolve().is_relative_to(project.resolve()):
            raise ValueError("Unsafe code bundle path.")
        content = (source / relative).read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f"Training source checksum mismatch: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    (project / "bundle_manifest.json").write_text(json.dumps(provenance, indent=2))
    manifests = []
    for path in inputs.rglob("manifest.json"):
        try:
            metadata = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if "tokenizer_sha256" in metadata and (path.parent / "test").is_dir():
            manifests.append(path)
    if len(manifests) != 1:
        raise RuntimeError("Attach exactly one prepared test input.")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-input",
            "--disable-pip-version-check",
            "-r",
            str(project / "requirements-kaggle.txt"),
        ],
        check=True,
    )
    bundle_path = working / "code_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in [*provenance["sha256"], "bundle_manifest.json"]:
            archive.write(project / relative, relative)
    artifact = working / "inference_artifact"
    subprocess.run(
        [
            sys.executable,
            "export_inference.py",
            "--checkpoint",
            str(checkpoints[0]),
            "--prepared-dir",
            str(manifests[0].parent),
            "--output",
            str(artifact),
            "--code-bundle",
            str(bundle_path),
        ],
        cwd=project,
        check=True,
    )
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="0",
        OMP_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONUNBUFFERED="1",
    )
    settings = globals().get("SETTINGS", {})
    if settings.get("sweep_plan"):
        plan = json.loads((project / settings["sweep_plan"]).read_text())
        sweep = working / "decoding-sweep"
        subprocess.run(
            [
                sys.executable,
                "benchmark_inference.py",
                "--checkpoint",
                str(artifact / "model.pt"),
                "--prepared-dir",
                str(artifact),
                "--output-dir",
                str(sweep),
                "--plan",
                str(project / settings["sweep_plan"]),
            ],
            cwd=project,
            env=env,
            check=True,
            timeout=3000,
        )
        summary = json.loads((sweep / "summary.json").read_text())
        for variant in summary["variants"]:
            first = variant["runs"][0]
            if "predictions" in first:
                shutil.copyfile(
                    sweep / first["predictions"],
                    working / f"predictions-{variant['tag']}.jsonl",
                )
        selected = next(
            v for v in summary["variants"] if v["tag"] == summary["default_tag"]
        )
        first = selected["runs"][0]
        if "predictions" in first:
            shutil.copyfile(sweep / first["predictions"], working / "predictions.jsonl")
            shutil.copyfile(sweep / first["report"], working / "inference_report.json")
        result = {
            "selected": selected,
            "candidates": summary["variants"],
            "repeat_passes": selected["passes_all_repeats"],
            "ready_for_submission": summary["default_passes"],
            "quality_measured": False,
            "selection_policy": summary["selection_policy"],
            "total_job_seconds": time.monotonic() - started,
        }
        (working / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
        return
    records = []

    def generate(tokens, batch, tag):
        remaining = 1100 - (time.monotonic() - started)
        if remaining < 60:
            return None
        predictions = working / f"predictions-{tag}.jsonl"
        report = working / f"report-{tag}.json"
        wall = time.perf_counter()
        try:
            subprocess.run(
                [
                    sys.executable,
                    "inference.py",
                    f"checkpoint={artifact / 'model.pt'}",
                    f"prepared_dir={artifact}",
                    f"output_path={predictions}",
                    f"report_path={report}",
                    "device=cuda",
                    "require_t4=true",
                    f"batch_size={batch}",
                    f"max_new_tokens={tokens}",
                ],
                cwd=project,
                env=env,
                check=True,
                timeout=min(240, remaining - 20),
            )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
            print(
                f"Inference candidate {tag} failed: {type(error).__name__}", flush=True
            )
            return None
        outer_seconds = time.perf_counter() - wall
        verify_jsonl(predictions)
        measured = json.loads(report.read_text())
        measured["external_process_seconds"] = outer_seconds
        measured["passes_required_benchmark"] = bool(
            measured["passes_required_benchmark"] and outer_seconds <= 180
        )
        report.write_text(json.dumps(measured, indent=2))
        record = dict(
            tag=tag,
            tokens=tokens,
            batch=batch,
            predictions=predictions.name,
            report=report.name,
            seconds=outer_seconds,
            passes=measured["passes_required_benchmark"],
        )
        records.append(record)
        return record

    generate(64, 128, "64-b128")
    generate(64, 64, "64-b64")
    eligible = [r for r in records if r["tokens"] == 64 and r["passes"]]
    candidates = eligible or [r for r in records if r["tokens"] == 64]
    if not candidates:
        raise RuntimeError("No inference candidate produced a complete submission.")
    selected = min(candidates, key=lambda r: r["seconds"])
    shutil.copyfile(working / selected["predictions"], working / "predictions.jsonl")
    shutil.copyfile(working / selected["report"], working / "inference_report.json")
    repeat = generate(64, selected["batch"], "64-repeat")
    longer = generate(128, 128, "128-b128")
    if longer:
        shutil.copyfile(
            working / longer["predictions"], working / "predictions-128.jsonl"
        )
    summary = dict(
        selected=selected,
        candidates=records,
        repeat_passes=bool(repeat and repeat["passes"]),
        ready_for_submission=bool(selected["passes"] and repeat and repeat["passes"]),
        total_job_seconds=time.monotonic() - started,
    )
    (working / "result.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
