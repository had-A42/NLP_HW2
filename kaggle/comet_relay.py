"""Forward persisted Kaggle metrics to Comet using local credentials only."""

import argparse
import configparser
import json
import os
import time
from pathlib import Path

import comet_ml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-hours", type=float, default=8)
    args = parser.parse_args()
    root = Path(args.directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text())
    credentials = configparser.ConfigParser(interpolation=None)
    credentials.read(Path.home() / ".comet.config")
    key = os.getenv("COMET_API_KEY") or credentials.get(
        "comet", "api_key", fallback=None
    )
    if not key:
        raise RuntimeError(
            "No existing local Comet credential; never put it in the kernel source."
        )
    state_path = root / "comet-relay-state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    sent = {tuple(item) for item in state.get("sent", [])}
    common = dict(
        api_key=key,
        auto_metric_logging=False,
        auto_param_logging=False,
        auto_output_logging=False,
        log_code=False,
        log_env_details=False,
        log_env_gpu=False,
        log_env_cpu=False,
        log_git_metadata=False,
        log_git_patch=False,
    )
    if state.get("experiment_key"):
        experiment = comet_ml.ExistingExperiment(
            previous_experiment=state["experiment_key"], **common
        )
    else:
        experiment = comet_ml.Experiment(
            project_name=config["writer"]["project_name"],
            workspace=config["writer"]["workspace"],
            **common
        )
        experiment.set_name(config["run_name"])
        experiment.log_parameters(config)
        experiment.log_parameter("logging_transport", "local_kaggle_sse_relay")
    state["experiment_key"] = experiment.get_key()
    state["comet_url"] = experiment.url
    deadline = (
        time.monotonic() + args.max_hours * 3600 if args.max_hours > 0 else float("inf")
    )

    def save():
        state["sent"] = sorted(sent)
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(state_path)

    save()
    print("COMET_RELAY " + state["comet_url"], flush=True)
    try:
        while time.monotonic() < deadline:
            path = root / "live-metrics.jsonl"
            added = 0
            if path.exists():
                for line in path.read_text().splitlines():
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    kind = "train" if "train_loss" in record else "val"
                    identity = (record["step"], kind)
                    if identity in sent:
                        continue
                    experiment.log_metrics(
                        {k: v for k, v in record.items() if k != "step"},
                        step=record["step"],
                    )
                    sent.add(identity)
                    state["latest_" + kind] = record
                    added += 1
            if added:
                experiment.flush()
                save()
                print(
                    json.dumps(
                        dict(
                            sent_points=len(sent),
                            latest_step=max(item[0] for item in sent),
                        )
                    ),
                    flush=True,
                )
            monitor = root / "monitor-state.json"
            if monitor.exists() and json.loads(monitor.read_text()).get("stage") in {
                "complete",
                "training_failed",
                "monitor_failed",
                "needs_review",
            }:
                break
            time.sleep(10)
    finally:
        experiment.end()
        save()


if __name__ == "__main__":
    main()
