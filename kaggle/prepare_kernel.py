"""Prepare local code/input metadata and a private T4x2 kernel. Never uploads or runs it."""

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent


def package(
    output,
    owner,
    data_slug="nlp-hw2-data",
    code_slug="nlp-hw2-code",
    kernel_slug="nlp-hw2-train",
    overrides=(),
    max_hours=4,
):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    dataset = output / "code_dataset"
    kernel = output / "kernel"
    dataset.mkdir()
    kernel.mkdir()
    files = []
    for path in (PROJECT / "src").rglob("*"):
        if (
            path.is_file()
            and path.suffix in {".py", ".yaml", ".json"}
            and "__pycache__" not in path.parts
        ):
            files.append(path)
    files += [
        PROJECT / name
        for name in [
            "train.py",
            "inference.py",
            "prepare_data.py",
            "average_checkpoints.py",
            "benchmark_inference.py",
            "export_inference.py",
            "requirements-kaggle.txt",
            "LICENSE",
        ]
    ]
    hashes = {}
    with zipfile.ZipFile(
        dataset / "code_bundle.zip", "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(files):
            if path.is_symlink() or not path.resolve().is_relative_to(PROJECT):
                raise ValueError(f"External/symlink file cannot be packaged: {path}")
            relative = str(path.relative_to(PROJECT))
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            archive.write(path, relative)
        archive.writestr(
            "bundle_manifest.json", json.dumps({"sha256": hashes}, indent=2)
        )
    (dataset / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "NLP HW2 code bundle",
                "id": f"{owner}/{code_slug}",
                "licenses": [{"name": "MIT"}],
            },
            indent=2,
        )
    )
    source = (
        "SETTINGS = "
        + repr({"overrides": list(overrides), "max_hours": max_hours})
        + "\n\n"
    )
    source += (PROJECT / "kaggle/train_launcher.py").read_text()
    (kernel / "run.py").write_text(source)
    (kernel / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": f"{owner}/{kernel_slug}",
                "title": "NLP HW2 train",
                "code_file": "run.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": "true",
                "enable_gpu": "true",
                "machine_shape": "NvidiaTeslaT4",
                "enable_internet": "true",
                "dataset_sources": [f"{owner}/{code_slug}", f"{owner}/{data_slug}"],
                "competition_sources": [],
                "kernel_sources": [],
                "model_sources": [],
            },
            indent=2,
        )
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--owner", default="andrewkhoroshilov")
    parser.add_argument("--data-slug", default="nlp-hw2-data")
    parser.add_argument("--code-slug", default="nlp-hw2-code")
    parser.add_argument("--kernel-slug", default="nlp-hw2-train")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--max-hours", type=float, default=4)
    args = parser.parse_args()
    if args.max_hours <= 0:
        parser.error("--max-hours must be positive")
    result = package(
        args.output,
        args.owner,
        args.data_slug,
        args.code_slug,
        args.kernel_slug,
        args.override,
        args.max_hours,
    )
    print(result)
    print(
        "Local files prepared. Upload the private code/data datasets, then explicitly push the kernel."
    )


if __name__ == "__main__":
    main()
