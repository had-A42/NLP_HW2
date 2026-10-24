"""Shared artifact and logging helpers for the code-generation pipeline."""

import hashlib
import json
import os
import platform
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.tokenization import CodeTokenizer


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepared_artifacts(directory):
    root = Path(directory).expanduser().resolve()
    if not (root / "manifest.json").is_file():
        raise FileNotFoundError(
            f"Missing prepared data at {root}; run prepare_data.py first."
        )
    manifest = json.loads((root / "manifest.json").read_text())
    tokenizer = CodeTokenizer(root / "tokenizer.json")
    if tokenizer.sha256 != manifest["tokenizer_sha256"]:
        raise ValueError("Tokenizer does not match prepared-data manifest.")
    return root, manifest, tokenizer


def model_config(config, manifest):
    result = OmegaConf.to_container(config.model, resolve=True)
    result.pop("_target_", None)
    for field in [
        "vocab_size",
        "pad_id",
        "bos_id",
        "eos_id",
        "max_source_length",
        "max_target_length",
    ]:
        result[field] = manifest[field]
        config.model[field] = manifest[field]
    return result


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    # Only load our own trusted training artifacts, never downloaded foreign weights.
    return torch.load(path, map_location=device, weights_only=False)


class CodeLogger:
    def __init__(self, config, directory, experiment_key=None):
        self.directory = Path(directory)
        self.file = (self.directory / "metrics.jsonl").open("a")
        self.experiment = None
        self.experiment_key = experiment_key
        mode = config.writer.mode
        if mode == "disabled":
            return
        if mode not in {"online", "offline"}:
            raise ValueError("writer.mode must be online, offline or disabled.")
        import comet_ml

        if mode == "online":
            if not os.getenv("COMET_API_KEY"):
                raise RuntimeError(
                    "Set COMET_API_KEY via the environment or Kaggle Secrets; no interactive login in jobs."
                )
            if experiment_key:
                self.experiment = comet_ml.ExistingExperiment(
                    previous_experiment=experiment_key,
                    auto_metric_logging=False,
                    auto_param_logging=False,
                    log_code=False,
                )
            else:
                self.experiment = comet_ml.Experiment(
                    project_name=config.writer.project_name,
                    workspace=config.writer.workspace,
                    auto_metric_logging=False,
                    auto_param_logging=False,
                    log_code=False,
                )
        else:
            self.experiment = comet_ml.OfflineExperiment(
                project_name=config.writer.project_name,
                workspace=config.writer.workspace,
                offline_directory=str(self.directory / "comet"),
                auto_metric_logging=False,
                auto_param_logging=False,
                log_code=False,
            )
        self.experiment.set_name(config.run_name)
        self.experiment.log_parameters(OmegaConf.to_container(config, resolve=True))
        self.experiment_key = self.experiment.get_key()

    def log(self, step, **metrics):
        record = {"step": step, **metrics}
        self.file.write(json.dumps(record, allow_nan=False) + "\n")
        self.file.flush()
        print(json.dumps(record), flush=True)
        if self.experiment is not None:
            self.experiment.log_metrics(metrics, step=step)

    def finish(self):
        self.file.close()
        if self.experiment is not None:
            self.experiment.end()


def environment_report(device):
    result = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        result.update(
            gpu=torch.cuda.get_device_name(device),
            total_gpu_memory=torch.cuda.get_device_properties(device).total_memory,
        )
    return result
