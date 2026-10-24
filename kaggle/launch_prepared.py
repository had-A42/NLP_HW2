"""Upload a prepared private code bundle, start training and detach monitoring."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kernel_refs import versioned_kernel_ref

PROJECT = Path(__file__).resolve().parent.parent


def authenticated_api():
    api = KaggleApi()
    api.authenticate()
    return api


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", default="kaggle/build/pre768-unlimited")
    parser.add_argument("--output", default="artifacts/pre768-3epochs")
    parser.add_argument("--relay-python", default=str(PROJECT / ".venv/bin/python"))
    args = parser.parse_args()
    build = Path(args.build).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "launch-state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    bundle_hash = hashlib.sha256(
        (build / "code_dataset/code_bundle.zip").read_bytes()
    ).hexdigest()
    if state.get("bundle_sha256", bundle_hash) != bundle_hash:
        raise ValueError(
            "This output directory belongs to a different frozen source bundle."
        )
    state["bundle_sha256"] = bundle_hash
    # Check the existing local relay setup before consuming GPU time. Never print credentials.
    subprocess.run(
        [
            args.relay_python,
            "-c",
            "import configparser,os; from pathlib import Path; import comet_ml; "
            "c=configparser.ConfigParser(interpolation=None); c.read(Path.home()/'.comet.config'); "
            "assert os.getenv('COMET_API_KEY') or c.get('comet','api_key',fallback=None), 'Configure local Comet credentials first'",
        ],
        check=True,
        env=dict(os.environ, MPLCONFIGDIR=str(output / "mpl-cache")),
    )

    def save():
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(state_path)

    metadata = json.loads((build / "code_dataset/dataset-metadata.json").read_text())
    if not state.get("dataset_uploaded"):
        response = authenticated_api().dataset_create_new(
            str(build / "code_dataset"), public=False, quiet=True, convert_to_csv=False
        )
        if getattr(response, "error", None) or response.status.lower() != "ok":
            raise RuntimeError(
                "Kaggle rejected the private dataset upload; inspect dataset status."
            )
        state["dataset_uploaded"] = True
        state["dataset_url"] = response.url
        save()
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        try:
            status = authenticated_api().dataset_status(metadata["id"])
        except Exception as error:
            # Fresh private datasets can briefly return 403/404 while ACLs propagate.
            code = getattr(getattr(error, "response", None), "status_code", None)
            if code not in {403, 404, 429, 500, 502, 503, 504}:
                raise
            print(f"Waiting for private dataset publication: HTTP {code}", flush=True)
            time.sleep(10)
            continue
        if str(status).upper() == "READY":
            break
        if str(status).upper() in {"ERROR", "FAILED"}:
            raise RuntimeError("Prepared code dataset processing failed.")
        print("Waiting for private code dataset: " + str(status), flush=True)
        time.sleep(10)
    else:
        raise TimeoutError(
            "Code dataset was not ready after ten minutes; rerun to resume."
        )
    kernel_meta = json.loads((build / "kernel/kernel-metadata.json").read_text())
    if not state.get("training"):
        response = authenticated_api().kernels_push(
            str(build / "kernel"), acc="NvidiaTeslaT4"
        )
        # Kaggle can create a version while rejecting one of its inputs. Record
        # that identity first so a retry cannot accidentally submit another job.
        if response is not None and response.versionNumber:
            actual_ref = response.ref or kernel_meta["id"]
            state["training"] = versioned_kernel_ref(actual_ref, response.versionNumber)
            state["kernel_id"] = response.kernelId
            save()
        if (
            response is None
            or response.error
            or response.invalidKernelSources
            or response.invalidDatasetSources
        ):
            if response is not None:
                (output / "push-rejection.json").write_text(
                    json.dumps(
                        {
                            "error": response.error,
                            "invalid_kernel_sources": response.invalidKernelSources,
                            "invalid_dataset_sources": response.invalidDatasetSources,
                            "version_number": response.versionNumber,
                            "ref": response.ref,
                        },
                        indent=2,
                    )
                )
            raise RuntimeError("Kaggle rejected training; no monitor started.")
        if not response.versionNumber:
            raise RuntimeError(
                "Training response has no version; inspect Kaggle before retrying."
            )
        # For a newly created notebook, Kaggle may derive the slug from its title.
    inference_meta_path = build / "inference/kernel-metadata.json"
    inference_meta = json.loads(inference_meta_path.read_text())
    inference_meta["kernel_sources"] = [state["training"]]
    inference_meta_path.write_text(json.dumps(inference_meta, indent=2))

    def detach(name, executable, script, arguments):
        pid = state.get(name + "_pid")
        if pid:
            try:
                os.kill(pid, 0)
                return
            except ProcessLookupError:
                pass
        with (output / f"{name}.log").open("a") as log:
            child = subprocess.Popen(
                [executable, "-u", str(PROJECT / "kaggle" / script), *arguments],
                cwd=PROJECT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=dict(os.environ, MPLCONFIGDIR=str(output / "mpl-cache")),
            )
        state[name + "_pid"] = child.pid
        save()

    detach(
        "monitor",
        sys.executable,
        "follow_training.py",
        [
            "--training",
            state["training"],
            "--inference-folder",
            str(build / "inference"),
            "--output",
            str(output),
            "--max-hours",
            "0",
            "--inference-timeout",
            "3600",
        ],
    )
    detach(
        "live-stream",
        sys.executable,
        "stream_training_metrics.py",
        [
            "--kernel",
            state["training"],
            "--output",
            str(output),
            "--max-hours",
            "0",
            "--capture-log",
        ],
    )
    detach(
        "comet-relay",
        args.relay_python,
        "comet_relay.py",
        [
            "--directory",
            str(output),
            "--config",
            str(build / "requested-config.json"),
            "--max-hours",
            "0",
        ],
    )
    print(
        json.dumps(
            {
                "training": state["training"],
                "artifacts": str(output),
                "monitoring_started": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
