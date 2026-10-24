"""Run full inference in fresh processes on one visible T4 and collect reports."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from src.utils.inference_sweep import run_suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sort-by-length", action="store_true")
    parser.add_argument("--compile-step", action="store_true")
    parser.add_argument("--cache-type", choices=["static", "dynamic"], default="static")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--plan", help="JSON decoding sweep; overrides the legacy grid."
    )
    args = parser.parse_args()
    if args.plan:
        run_suite(
            args.checkpoint,
            args.prepared_dir,
            args.output_dir,
            json.loads(Path(args.plan).read_text()),
        )
        return
    if args.repeats < 1 or min(args.batch_sizes) < 1:
        parser.error("Repeats and batch sizes must be positive.")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    reports = []
    for batch in args.batch_sizes:
        for repeat in range(args.repeats):
            stem = output / f"batch-{batch}-repeat-{repeat}"
            command = [
                sys.executable,
                str(Path(__file__).parent / "inference.py"),
                f"checkpoint={Path(args.checkpoint).resolve()}",
                f"prepared_dir={Path(args.prepared_dir).resolve()}",
                f"output_path={stem}.jsonl",
                f"report_path={stem}.report.json",
                f"batch_size={batch}",
                f"max_new_tokens={args.max_new_tokens}",
                f"sort_by_length={str(args.sort_by_length).lower()}",
                f"compile_step={str(args.compile_step).lower()}",
                f"cache_type={args.cache_type}",
                "device=cuda",
                "require_t4=true",
            ]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
            completed = subprocess.run(command, env=env, check=False)
            if completed.returncode:
                reports.append(
                    {
                        "batch_size": batch,
                        "repeat": repeat,
                        "failed": True,
                        "returncode": completed.returncode,
                    }
                )
            else:
                report = json.loads(Path(f"{stem}.report.json").read_text())
                reports.append({"repeat": repeat, **report})
            (output / "reports.json").write_text(json.dumps(reports, indent=2))
    print(output / "reports.json")


if __name__ == "__main__":
    main()
