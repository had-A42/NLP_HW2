"""Average compatible model weights; the result is one model, not an ensemble."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from src.utils.code_runtime import load_checkpoint, save_checkpoint


def average(paths, output):
    paths = [Path(p).resolve() for p in paths]
    if len(paths) < 2 or len(set(paths)) != len(paths):
        raise ValueError("Provide at least two distinct checkpoints.")
    output = Path(output).resolve()
    if output.exists() or output in paths:
        raise FileExistsError(f"Refusing to overwrite {output}")
    first = load_checkpoint(paths[0])
    for key in ["model_config", "tokenizer_sha256", "manifest_sha256"]:
        if key not in first:
            raise ValueError(f"Missing compatibility metadata: {key}")
    # Clone each key independently: tied embeddings/output may share storage.
    result = {
        k: v.float().clone() if v.is_floating_point() else v.clone()
        for k, v in first["state_dict"].items()
    }
    for path in paths[1:]:
        checkpoint = load_checkpoint(path)
        for key in ["model_config", "tokenizer_sha256", "manifest_sha256"]:
            if checkpoint[key] != first[key]:
                raise ValueError(f"Incompatible checkpoint {path}: {key}")
        if checkpoint["state_dict"].keys() != result.keys():
            raise ValueError("Checkpoint state keys do not match.")
        for key, value in checkpoint["state_dict"].items():
            if value.shape != result[key].shape:
                raise ValueError(f"Incompatible tensor {key}")
            if value.is_floating_point():
                result[key] += value.float()
            elif not torch.equal(value, result[key]):
                raise ValueError(f"Non-floating state differs: {key}")
    for key, value in result.items():
        if value.is_floating_point():
            result[key] = (value / len(paths)).to(first["state_dict"][key].dtype)
    payload = {
        key: first[key]
        for key in [
            "format_version",
            "model_config",
            "tokenizer_sha256",
            "manifest_sha256",
            "config",
        ]
    }
    payload["state_dict"] = result
    payload["averaged_from"] = [
        {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in paths
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(output, payload)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(average(args.checkpoints, args.output))


if __name__ == "__main__":
    main()
