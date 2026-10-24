"""Detached Kaggle monitor: completed training -> GPU inference -> local artifacts.

Uses the Kaggle CLI tool's Python environment. Credentials remain in Kaggle's
existing auth storage; this script never reads or prints their values.
"""

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kernel_refs import canonical_kernel_ref, versioned_kernel_ref


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", required=True, help="owner/kernel/version")
    parser.add_argument("--inference-folder", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-hours", type=float, default=8)
    parser.add_argument("--inference-timeout", type=int, default=1200)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "monitor.lock"
    if lock.exists():
        try:
            os.kill(int(lock.read_text()), 0)
        except ProcessLookupError:
            lock.unlink()
        else:
            raise RuntimeError(
                "A monitor is already running for this output directory."
            )
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    state_file = output / "monitor-state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    if state.get("training", args.training) != args.training:
        raise ValueError("Output directory belongs to a different training version.")
    state.update(training=args.training, pid=os.getpid())

    def save(**values):
        if values.get("status") not in {None, "ERROR", "api_retry"}:
            state.pop("error_type", None)
        state.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        temporary = output / "monitor-state.tmp"
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(output / "monitor-state.json")
        print(json.dumps(state), flush=True)

    def authenticated_api():
        # A long-lived SDK instance can retain an expired access token.
        api = KaggleApi()
        api.authenticate()
        return api

    deadline = (
        time.monotonic() + args.max_hours * 3600 if args.max_hours > 0 else float("inf")
    )

    def wait_for(kernel, stage):
        while time.monotonic() < deadline:
            try:
                response = authenticated_api().kernels_status(kernel)
                status = getattr(
                    response.status, "name", str(response.status).split(".")[-1]
                )
                save(stage=stage, kernel=kernel, status=status)
                if status in {
                    "COMPLETE",
                    "ERROR",
                    "CANCELLED",
                    "CANCELED",
                    "CANCEL_ACKNOWLEDGED",
                }:
                    return status
            except Exception as error:
                # Avoid including service response bodies or credentials in the log.
                save(stage=stage, status="api_retry", error_type=type(error).__name__)
            time.sleep(args.poll_seconds)
        raise TimeoutError("Monitor deadline exhausted.")

    def download(kernel, target, pattern):
        for attempt in range(5):
            try:
                authenticated_api().kernels_output(
                    kernel, str(target), file_pattern=pattern, force=True, quiet=True
                )
                return
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(20)

    def wait_for_checkpoint():
        for _ in range(30):
            page = None
            while True:
                response = authenticated_api().kernels_list_files(
                    args.training, page_token=page, page_size=200
                )
                if any(
                    item.name.endswith(("/best.pt", "/last.pt", "best.pt", "last.pt"))
                    for item in response.files
                ):
                    return
                page = response.next_page_token
                if not page:
                    break
            time.sleep(20)
        raise RuntimeError("Training completed without a published checkpoint.")

    try:
        status = wait_for(args.training, "training")
        download(
            args.training,
            output / "training",
            r"(?:metrics\.jsonl|config\.yaml|training_plan\.json|environment\.json|code_provenance\.json)$",
        )
        if status != "COMPLETE":
            save(stage="training_failed", status=status)
            return
        wait_for_checkpoint()
        folder = Path(args.inference_folder).resolve()
        metadata = json.loads((folder / "kernel-metadata.json").read_text())
        if metadata["kernel_sources"] != [args.training]:
            raise ValueError(
                "Inference input is not pinned to the monitored training version."
            )
        inference = state.get("inference")
        if not inference:
            save(stage="starting_inference", status="submitting")
            response = authenticated_api().kernels_push(
                str(folder), timeout=str(args.inference_timeout), acc="NvidiaTeslaT4"
            )
            if (
                response is None
                or response.error
                or response.invalidKernelSources
                or response.invalidDatasetSources
            ):
                raise RuntimeError(
                    "Kaggle did not accept the inference inputs or code."
                )
            version = response.versionNumber
            actual_ref = response.ref or metadata["id"]
            inference = versioned_kernel_ref(actual_ref, version) if version else canonical_kernel_ref(actual_ref)
            save(inference=inference)
        status = wait_for(inference, "inference")
        download(
            inference,
            output / "inference",
            r"(?:\.jsonl|\.json|\.zip|model\.pt|\.arrow|\.yaml)$",
        )
        generated = output / "inference/predictions.jsonl"
        result_file = output / "inference/result.json"
        if generated.is_file():
            rows = [json.loads(line) for line in generated.read_text().splitlines()]
            if len(rows) != 1919 or any(
                set(r) != {"code"} or not isinstance(r["code"], str) for r in rows
            ):
                raise ValueError("Downloaded submission failed its format check.")
            shutil.copyfile(generated, output / "predictions.jsonl")
        result = json.loads(result_file.read_text()) if result_file.is_file() else {}
        save(
            stage="complete"
            if status == "COMPLETE" and result.get("ready_for_submission")
            else "needs_review",
            status=status,
            predictions=str(output / "predictions.jsonl")
            if generated.is_file()
            else None,
            ready_for_submission=result.get("ready_for_submission", False),
        )
    except Exception as error:
        save(stage="monitor_failed", status="ERROR", error_type=type(error).__name__)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
