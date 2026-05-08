from collections import Counter
from typing import Any


def support_recall(pred_chunks: list[str], gold_chunks: list[str]) -> float:
    gold = set(gold_chunks)
    pred = set(pred_chunks)
    if not gold:
        return 0.0
    return len(gold & pred) / len(gold)


def support_precision(pred_chunks: list[str], gold_chunks: list[str]) -> float:
    gold = set(gold_chunks)
    pred = set(pred_chunks)
    if not pred:
        return 0.0
    return len(gold & pred) / len(pred)


def support_f1(pred_chunks: list[str], gold_chunks: list[str]) -> float:
    precision = support_precision(pred_chunks, gold_chunks)
    recall = support_recall(pred_chunks, gold_chunks)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def support_f2(pred_chunks: list[str], gold_chunks: list[str]) -> float:
    precision = support_precision(pred_chunks, gold_chunks)
    recall = support_recall(pred_chunks, gold_chunks)
    if precision + recall == 0:
        return 0.0
    beta_squared = 4
    return (1 + beta_squared) * precision * recall / (beta_squared * precision + recall)


def false_negative_rate(pred_chunks: list[str], gold_chunks: list[str]) -> float:
    gold = set(gold_chunks)
    if not gold:
        return 0.0
    pred = set(pred_chunks)
    return len(gold - pred) / len(gold)


def false_positive_per_example(pred_chunks: list[str], gold_chunks: list[str]) -> float:
    return float(len(set(pred_chunks) - set(gold_chunks)))


def slot_coverage(slots: list[dict[str, Any]], gold_chunks: list[str]) -> float:
    gold = set(gold_chunks)
    if not slots:
        return 0.0
    hit = 0
    for slot in slots:
        if set(slot["pivot_chunks"]) & gold:
            hit += 1
    return hit / len(slots)


def gold_duplicate_hit_rate(
    pred_chunks_with_duplicates: list[str],
    gold_chunks: list[str],
) -> float:
    gold = set(gold_chunks)
    counts = Counter(pred_chunks_with_duplicates)
    duplicated = [cid for cid, count in counts.items() if count >= 2]
    if not duplicated:
        return 0.0
    return sum(1 for cid in duplicated if cid in gold) / len(duplicated)


def invalid_id_rate(invalid_ids: list[str], pred_chunks_with_duplicates: list[str]) -> float:
    total = len(invalid_ids) + len(pred_chunks_with_duplicates)
    if total == 0:
        return 0.0
    return len(invalid_ids) / total


def distinct_slot_ratio(slots: list[dict[str, Any]]) -> float:
    if not slots:
        return 0.0
    signatures = [tuple(sorted(slot["pivot_chunks"])) for slot in slots]
    return len(set(signatures)) / len(signatures)
