"""Attach to an existing inference job and download small results. Never submits jobs."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


def api():
    client = KaggleApi()
    client.authenticate()
    return client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    state = {"inference": args.kernel, "stage": "inference"}

    def save():
        (out / "monitor-state.json").write_text(json.dumps(state, indent=2))
        print(json.dumps(state), flush=True)

    while True:
        try:
            response = api().kernels_status(args.kernel)
            status = getattr(response.status, "name", str(response.status))
            state.update(status=status)
            save()
            if status in {"COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"}:
                break
        except Exception as error:
            state.update(status="api_retry", error_type=type(error).__name__)
            save()
        time.sleep(20)
    pattern = r"(?:predictions-.*\.jsonl|beam-summary\.json|beam-plan\.json|summary\.json|runs\.json|.*\.report\.json|.*\.stdout\.log)$"
    for attempt in range(12):
        try:
            api().kernels_output(args.kernel, str(out), file_pattern=pattern, force=True, quiet=True)
            if (out / "beam-summary.json").is_file():
                break
        except Exception:
            if attempt == 11:
                raise
        time.sleep(15)
    summary = json.loads((out / "beam-summary.json").read_text())
    verified = []
    for variant in summary["variants"]:
        if not variant["passes_all_repeats"]:
            continue
        prediction = out / variant["predictions"]
        rows = [json.loads(s) for s in prediction.read_text().splitlines()]
        assert len(rows) == 1919 and all(set(r) == {"code"} and isinstance(r["code"], str) for r in rows)
        assert len(variant["runs"]) == 2
        for run in variant["runs"]:
            report = json.loads((out / run["directory"] / run["report"]).read_text())
            assert report["eligible_single_t4_benchmark"] and report["visible_cuda_devices"] == 1
            assert report["examples"] == 1919 and run["seconds"] <= 180 and run["passes"]
        assert hashlib.sha256(prediction.read_bytes()).hexdigest() == variant["runs"][0]["predictions_sha256"]
        verified.append({"tag": variant["tag"], "num_beams": variant["num_beams"],
                         "batch_size": variant["batch_size"], "max_seconds": variant["max_seconds"],
                         "predictions": str(prediction), "rows": len(rows)})
    (out / "verification.json").write_text(json.dumps(verified, indent=2))
    state.update(stage="complete" if status == "COMPLETE" else "needs_review", verified_variants=len(verified))
    save()


if __name__ == "__main__":
    main()
