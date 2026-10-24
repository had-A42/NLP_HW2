"""Export only model weights, tokenizer and raw test questions for fast inference."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from src.utils.code_runtime import load_checkpoint, prepared_artifacts, save_checkpoint


def export(
    checkpoint_path, prepared_dir, output_dir, code_bundle=None, dataset_id=None
):
    root, manifest, tokenizer = prepared_artifacts(prepared_dir)
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint["tokenizer_sha256"] != tokenizer.sha256:
        raise ValueError("Checkpoint/tokenizer mismatch.")
    output = Path(output_dir).resolve()
    if output.is_relative_to(root):
        raise ValueError("Export must not be inside the training input artifact.")
    output.mkdir(parents=True, exist_ok=False)
    fields = [
        "format_version",
        "state_dict",
        "model_config",
        "tokenizer_sha256",
        "manifest_sha256",
        "config",
    ]
    payload = {field: checkpoint[field] for field in fields}
    payload["source_checkpoint_sha256"] = hashlib.sha256(
        Path(checkpoint_path).read_bytes()
    ).hexdigest()
    save_checkpoint(output / "model.pt", payload)
    shutil.copyfile(root / "tokenizer.json", output / "tokenizer.json")
    shutil.copyfile(root / "manifest.json", output / "manifest.json")
    shutil.copytree(root / "test", output / "test")
    if code_bundle:
        shutil.copyfile(code_bundle, output / "code_bundle.zip")
    if dataset_id:
        (output / "dataset-metadata.json").write_text(
            json.dumps(
                {
                    "id": dataset_id,
                    "title": "NLP HW2 inference artifact",
                    "licenses": [{"name": "other"}],
                },
                indent=2,
            )
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--code-bundle")
    parser.add_argument("--dataset-id")
    args = parser.parse_args()
    print(
        export(
            args.checkpoint,
            args.prepared_dir,
            args.output,
            args.code_bundle,
            args.dataset_id,
        )
    )


if __name__ == "__main__":
    main()
