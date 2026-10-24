"""Executed as a private Kaggle kernel; SETTINGS is prepended by prepare_kernel.py."""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path


def main():
    started = time.monotonic()
    settings = globals().get("SETTINGS", {})
    input_root = Path("/kaggle/input")
    archives = list(input_root.rglob("code_bundle.zip"))
    expected_source = settings.get("expected_source_manifest")
    if expected_source is not None:
        matching = []
        for path in archives:
            with zipfile.ZipFile(path) as archive:
                if json.loads(archive.read("bundle_manifest.json")) == expected_source:
                    matching.append(path)
        archives = matching[:1]
    project = Path("/kaggle/working/hw2_project")
    project.mkdir(parents=True, exist_ok=True)

    def extract(archive_path, destination):
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                if (
                    not (destination / member.filename)
                    .resolve()
                    .is_relative_to(destination.resolve())
                ):
                    raise ValueError("Unsafe archive path.")
            archive.extractall(destination)

    if len(archives) == 1:
        extract(archives[0], project)
    elif not archives:
        bundles = list(input_root.rglob("bundle_manifest.json"))
        if expected_source is not None:
            bundles = [
                p for p in bundles if json.loads(p.read_text()) == expected_source
            ][:1]
        if len(bundles) != 1:
            raise RuntimeError("Attach exactly one code bundle dataset.")
        bundle = json.loads(bundles[0].read_text())
        for relative in [*bundle["sha256"], "bundle_manifest.json"]:
            target = project / relative
            if not target.resolve().is_relative_to(project.resolve()):
                raise ValueError("Unsafe bundle path.")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(bundles[0].parent / relative, target)
    else:
        raise RuntimeError("Multiple code bundles attached; select one.")
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
    manifests = []
    for path in input_root.rglob("manifest.json"):
        try:
            manifest = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        if "tokenizer_sha256" in manifest and (path.parent / "train").is_dir():
            manifests.append(path)
    if not manifests:
        data_archives = list(input_root.rglob("prepared.zip"))
        if len(data_archives) == 1:
            prepared = Path("/kaggle/working/prepared_data")
            prepared.mkdir(exist_ok=True)
            extract(data_archives[0], prepared)
            manifests = [prepared / "manifest.json"]
    if len(manifests) != 1:
        raise RuntimeError(
            "Attach exactly one prepared data artifact with manifest.json/train/val/test/tokenizer.json."
        )
    writer_mode = next(
        (
            value.split("=", 1)[1]
            for value in reversed(settings.get("overrides", []))
            if value.startswith("writer.mode=")
        ),
        "online",
    )
    if writer_mode == "online" and not os.getenv("COMET_API_KEY"):
        from kaggle_secrets import UserSecretsClient

        try:
            os.environ["COMET_API_KEY"] = UserSecretsClient().get_secret(
                "COMET_API_KEY"
            )
        except Exception:
            raise RuntimeError(
                "Enable the COMET_API_KEY Kaggle Secret for this kernel."
            ) from None
    import torch

    if torch.cuda.device_count() != 2:
        raise RuntimeError("This training launcher requires two visible GPUs.")
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpus": [torch.cuda.get_device_name(i) for i in range(2)],
            }
        ),
        flush=True,
    )
    output = Path("/kaggle/working/artifacts")
    output.mkdir(exist_ok=True)
    shutil.copyfile(project / "bundle_manifest.json", output / "code_provenance.json")
    env = dict(
        os.environ,
        HW2_PREPARED_DIR=str(manifests[0].parent),
        HW2_RUNS_DIR=str(output),
        PYTHONUNBUFFERED="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="1",
    )
    if settings.get("preflight_script"):
        preflight_env = dict(
            env,
            CUDA_VISIBLE_DEVICES="0",
            HW2_PREFLIGHT_MODEL=json.dumps(settings["preflight_model"]),
        )
        print("Starting disposable model GPU preflight", flush=True)
        subprocess.run(
            [sys.executable, "-c", settings["preflight_script"]],
            cwd=project,
            env=preflight_env,
            check=True,
            timeout=600,
        )
    hour_limit = settings.get("max_hours", 4)
    remaining = (
        float(hour_limit) * 3600 - (time.monotonic() - started)
        if hour_limit is not None
        else None
    )
    if remaining is not None and remaining < 120:
        raise RuntimeError("Setup exhausted the job budget; training was not started.")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        "train.py",
        "-cn=code_baseline",
        *settings.get("overrides", []),
        f"trainer.max_hours={(remaining - 60) / 3600}"
        if remaining is not None
        else "trainer.max_hours=null",
    ]
    if settings.get("resume_training"):
        checkpoints = list(input_root.rglob("last.pt"))
        if len(checkpoints) != 1:
            raise RuntimeError(
                "Resume requires exactly one attached last.pt checkpoint."
            )
        command.append(f"trainer.resume_from={checkpoints[0]}")
        # Keep the best weights from the preceding session even if subsequent
        # validation never improves. The full last.pt is loaded by the trainer.
        previous_best = list(input_root.rglob("best.pt"))
        if len(previous_best) != 1:
            raise RuntimeError("Resume requires exactly one previous best.pt.")
        best_target = output / settings["resume_run_name"] / "best.pt"
        best_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(previous_best[0], best_target)
    process = subprocess.Popen(command, cwd=project, env=env, start_new_session=True)
    try:
        returncode = process.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise RuntimeError(
            "Job time budget exhausted; recover from the last completed checkpoint."
        ) from None
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


if __name__ == "__main__":
    main()
