"""Collect live training metrics from Kaggle's SSE API without logging secrets."""

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import (
    ApiGetKernelSessionLogsStreamRequest,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-hours", type=float, default=8)
    parser.add_argument("--capture-log", action="store_true")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "live-metrics.jsonl"
    seen = set()
    if destination.exists():
        for line in destination.read_text().splitlines():
            record = json.loads(line)
            seen.add((record["step"], "train" if "train_loss" in record else "val"))
    api = KaggleApi()
    api.authenticate()
    owner, slug, version = api.parse_kernel_string(args.kernel)
    deadline = (
        time.monotonic() + args.max_hours * 3600 if args.max_hours > 0 else float("inf")
    )
    state = dict(kernel=args.kernel)

    def accept(event):
        text = event.get("data", "")
        if args.capture_log:
            safe = re.sub(
                r"(?i)((?:api[_ -]?key|access[_ -]?token|authorization)\s*[=:]\s*)[^\s,;]+",
                r"\1[REDACTED]",
                text,
            )
            with (root / "kaggle-live.log").open("a") as file:
                file.write(safe)
        link = re.search(r"https://www\.comet\.com/[^\s]+", text)
        if link:
            state["comet_url"] = link.group().strip()
        try:
            record = json.loads(text)
        except (ValueError, TypeError):
            return
        if not isinstance(record, dict) or "step" not in record:
            return
        kind = (
            "train"
            if "train_loss" in record
            else "val"
            if "val_loss" in record
            else None
        )
        if kind is None or (record["step"], kind) in seen:
            return
        seen.add((record["step"], kind))
        with destination.open("a") as file:
            file.write(json.dumps(record) + "\n")
        state["latest_" + kind] = record
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        temporary = root / "live-status.tmp"
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(root / "live-status.json")
        print(
            json.dumps(
                dict(kind=kind, step=record["step"], loss=record[kind + "_loss"])
            ),
            flush=True,
        )

    while time.monotonic() < deadline:
        try:
            api = KaggleApi()
            api.authenticate()
            request = ApiGetKernelSessionLogsStreamRequest()
            request.user_name, request.kernel_slug = owner, slug
            request.version_label = (
                api._kernel_version_label(version) if version else ""
            )
            request.wait_for_logs_url_seconds = 5
            with api.build_kaggle_client() as client:
                transport = client.kernels.kernels_api_client._client
                transport._init_session()
                original_send = transport._session.send

                def timed_send(request, **kwargs):
                    kwargs["timeout"] = (10, 40)
                    return original_send(request, **kwargs)

                transport._session.send = timed_send
                print("Connecting live log stream", flush=True)
                response = (
                    client.kernels.kernels_api_client.get_kernel_session_logs_stream(
                        request
                    )
                )
                response.encoding = "utf-8"
                print("Live log stream connected", flush=True)
                try:
                    if "application/json" in response.headers.get("Content-Type", ""):
                        for event in response.json():
                            accept(event)
                        return
                    for line in response.iter_lines(
                        chunk_size=1, decode_unicode=True
                    ):
                        if isinstance(line, bytes):
                            line = line.decode()
                        if "END_OF_LOG" in line:
                            return
                        if line.startswith("data: "):
                            accept(json.loads(line[6:]))
                        if time.monotonic() >= deadline:
                            return
                finally:
                    response.close()
        except Exception as error:
            print(
                json.dumps(dict(status="retry", error_type=type(error).__name__)),
                flush=True,
            )
            time.sleep(15)


if __name__ == "__main__":
    main()
