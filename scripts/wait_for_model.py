#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed OpenAI-compatible server readiness check")
    parser.add_argument("--url", required=True, help="e.g. http://127.0.0.1:8000/v1")
    parser.add_argument("--expected-model", required=True)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()
    if args.timeout <= 0 or args.interval <= 0:
        raise SystemExit("--timeout and --interval must be positive")
    deadline = time.monotonic() + args.timeout
    last_error = "no response"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(args.url.rstrip("/") + "/models", timeout=3) as response:
                payload = json.load(response)
            names = {item.get("id") for item in payload.get("data", [])}
            if args.expected_model in names:
                print(f"ready: {args.expected_model}")
                return
            last_error = f"available models: {sorted(name for name in names if name)}"
        except (OSError, ValueError, urllib.error.URLError) as error:
            last_error = str(error)
        time.sleep(args.interval)
    raise SystemExit(f"server readiness timeout: {last_error}")


if __name__ == "__main__":
    main()
