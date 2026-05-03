from __future__ import annotations

from typing import Any

from msp.eval.metrics import (
    distinct_slot_ratio,
    gold_duplicate_hit_rate,
    invalid_id_rate,
    slot_coverage,
    support_precision,
    support_recall,
)
from msp.inference.parser import parse_prediction


def evaluate_records(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {
            "support_recall": 0.0,
            "support_precision": 0.0,
            "slot_coverage": 0.0,
            "gold_duplicate_hit_rate": 0.0,
            "invalid_id_rate": 0.0,
            "exact_format_rate": 0.0,
            "distinct_slot_ratio": 0.0,
        }

    totals = {
        "support_recall": 0.0,
        "support_precision": 0.0,
        "slot_coverage": 0.0,
        "gold_duplicate_hit_rate": 0.0,
        "invalid_id_rate": 0.0,
        "exact_format_rate": 0.0,
        "distinct_slot_ratio": 0.0,
    }

    for record in records:
        if "pred_chunks" in record and "slots" not in record:
            parsed = {
                "slots": [],
                "pred_chunks": record["pred_chunks"],
                "pred_chunks_with_duplicates": record["pred_chunks"],
                "invalid_ids": [],
                "exact_format": True,
            }
        else:
            parsed = parse_prediction(
                record.get("prediction_text", ""),
                set(record.get("valid_chunk_ids", [])),
            )
        gold = record.get("gold_support_chunks", [])
        totals["support_recall"] += support_recall(parsed["pred_chunks"], gold)
        totals["support_precision"] += support_precision(parsed["pred_chunks"], gold)
        totals["slot_coverage"] += slot_coverage(parsed["slots"], gold)
        totals["gold_duplicate_hit_rate"] += gold_duplicate_hit_rate(
            parsed["pred_chunks_with_duplicates"],
            gold,
        )
        totals["invalid_id_rate"] += invalid_id_rate(
            parsed["invalid_ids"],
            parsed["pred_chunks_with_duplicates"],
        )
        totals["exact_format_rate"] += 1.0 if parsed["exact_format"] else 0.0
        totals["distinct_slot_ratio"] += distinct_slot_ratio(parsed["slots"])

    return {key: value / len(records) for key, value in totals.items()}
