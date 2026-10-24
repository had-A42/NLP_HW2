"""Benchmark wider beams using an unchanged, already exported scratch model."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path


def main():
    settings = globals()["SETTINGS"]
    started = time.monotonic()
    inputs, working = Path("/kaggle/input"), Path("/kaggle/working")
    models = [p for p in inputs.rglob("model.pt") if (p.parent / "tokenizer.json").is_file()]
    if len(models) != 1:
        raise RuntimeError("Attach exactly one exported inference artifact.")
    source = models[0].parent
    artifact = working / "inference_artifact"
    artifact.mkdir()
    for name in ["model.pt", "tokenizer.json", "manifest.json", "code_bundle.zip"]:
        shutil.copyfile(source / name, artifact / name)
    shutil.copytree(source / "test", artifact / "test")
    digest = hashlib.sha256((artifact / "model.pt").read_bytes()).hexdigest()
    if digest != settings["model_sha256"]:
        raise ValueError("Exported model checksum differs from the selected checkpoint.")
    project = working / "inference_project"
    project.mkdir()
    with zipfile.ZipFile(artifact / "code_bundle.zip") as archive:
        for member in archive.infolist():
            if not (project / member.filename).resolve().is_relative_to(project.resolve()):
                raise ValueError("Unsafe code archive path.")
        archive.extractall(project)
    provenance = json.loads((project / "bundle_manifest.json").read_text())
    for name, sha in provenance["sha256"].items():
        if hashlib.sha256((project / name).read_bytes()).hexdigest() != sha:
            raise ValueError("Inference code checksum mismatch.")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-input",
                    "--disable-pip-version-check", "-r", str(project / "requirements-kaggle.txt")], check=True)
    plan = settings["plan"]
    (working / "beam-plan.json").write_text(json.dumps(plan, indent=2))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="1",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    variants = []

    def candidate(spec):
        variant = {k: spec[k] for k in ["tag", "num_beams", "batch_size"]}
        variant.update(max_new_tokens=plan["max_new_tokens"], length_penalty=plan["length_penalty"])
        records = []
        for repeat in range(plan["repeats"]):
            if time.monotonic() - started > 2400:
                break
            leaf = working / "beam-runs" / f"{variant['tag']}-repeat-{repeat}"
            leaf.parent.mkdir(exist_ok=True)
            config = working / f"plan-{variant['tag']}-{repeat}.json"
            config.write_text(json.dumps({"repeats": 1, "default_tag": variant["tag"], "variants": [variant]}))
            try:
                subprocess.run([sys.executable, "benchmark_inference.py", "--checkpoint", str(artifact / "model.pt"),
                                "--prepared-dir", str(artifact), "--output-dir", str(leaf), "--plan", str(config)],
                               cwd=project, env=env, check=True, timeout=235)
                summary = json.loads((leaf / "summary.json").read_text())
                record = summary["variants"][0]["runs"][0]
                record.update(repeat=repeat, directory=str(leaf.relative_to(working)))
            except (subprocess.SubprocessError, OSError, ValueError, KeyError) as error:
                record = dict(repeat=repeat, passes=False, error_type=type(error).__name__)
            records.append(record)
            print("BEAM_RESULT " + json.dumps({**variant, **record}), flush=True)
            if not record["passes"]:
                break
        passed = len(records) == plan["repeats"] and all(r["passes"] for r in records)
        result = {**variant, "passes_all_repeats": passed, "runs": records,
                  "max_seconds": max((r.get("seconds", 235) for r in records), default=235)}
        if passed:
            first = records[0]
            output = working / f"predictions-{variant['tag']}.jsonl"
            shutil.copyfile(working / first["directory"] / first["predictions"], output)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            if len(rows) != 1919 or any(set(r) != {"code"} or not isinstance(r["code"], str) for r in rows):
                raise ValueError("Submission format mismatch.")
            result["predictions"] = output.name
        variants.append(result)
        (working / "beam-summary.json").write_text(json.dumps({"model_sha256": digest, "quality_measured": False,
                                                             "variants": variants}, indent=2))

    for spec in plan["variants"]:
        if time.monotonic() - started > 2200:
            break
        candidate(spec)
    for spec in plan.get("optional", []):
        eligible = [v for v in variants if v["num_beams"] == spec["requires_beam"] and v["passes_all_repeats"]]
        if eligible and min(v["max_seconds"] for v in eligible) <= spec["max_previous_seconds"] and time.monotonic() - started < 2100:
            candidate(spec)
    print("BEAM_SWEEP_COMPLETE " + json.dumps({"variants": len(variants), "seconds": time.monotonic() - started}), flush=True)


if __name__ == "__main__":
    main()
