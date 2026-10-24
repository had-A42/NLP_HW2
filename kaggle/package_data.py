"""Create a private-upload directory for prepared data without altering its source."""

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--owner", default="andrewkhoroshilov")
    parser.add_argument("--slug", default="nlp-hw2-data")
    args = parser.parse_args()
    root = Path(args.prepared_dir).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        hashlib.sha256((root / "tokenizer.json").read_bytes()).hexdigest()
        != manifest["tokenizer_sha256"]
    ):
        raise ValueError("Prepared tokenizer checksum mismatch.")
    output = Path(args.output).resolve()
    if output.is_relative_to(root):
        raise ValueError("Upload directory must be outside prepared source.")
    output.mkdir(parents=True, exist_ok=False)
    # Archive avoids copying/re-serializing the Arrow files; Kaggle extracts zip inputs.
    with zipfile.ZipFile(
        output / "prepared.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
    ) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                if path.is_symlink():
                    raise ValueError("Prepared data cannot contain symlinks.")
                archive.write(path, path.relative_to(root))
    (output / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "id": f"{args.owner}/{args.slug}",
                "title": "NLP HW2 prepared data",
                "licenses": [{"name": "other"}],
                "description": "Prepared solely from SlavaYus/tinysms_qa_12m; source license: other. No extra targets.",
            },
            indent=2,
        )
    )
    print(output)


if __name__ == "__main__":
    main()
