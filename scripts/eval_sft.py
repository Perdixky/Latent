from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from msp.eval.report import evaluate_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    args = parser.parse_args()

    with args.pred_path.open("r", encoding="utf-8-sig") as f:
        records = [json.loads(line) for line in f if line.strip()]

    metrics = evaluate_records(records)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
