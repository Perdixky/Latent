from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from msp.data.formatting import format_flat_target, format_prompt, format_target
from msp.data.schema import normalize_chunk_id


DEFAULT_INPUT_PATH = Path("data/evidence_corpus_full_shuffled.jsonl")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _split(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    n = len(records)
    if n == 0:
        return {"train": [], "valid": [], "test": []}
    if n == 1:
        return {"train": records, "valid": records, "test": records}
    if n == 2:
        return {"train": records[:1], "valid": records[1:], "test": records[1:]}

    train_end = max(1, int(n * 0.8))
    valid_end = max(train_end + 1, int(n * 0.9))
    if valid_end >= n:
        valid_end = n - 1
    return {
        "train": records[:train_end],
        "valid": records[train_end:valid_end],
        "test": records[valid_end:],
    }


def _convert_chunks(raw_chunks: list[dict[str, Any]], max_chunks: int) -> tuple[list[dict[str, str]], dict[Any, str]]:
    chunks = []
    id_map: dict[Any, str] = {}
    for raw in raw_chunks[:max_chunks]:
        cid = normalize_chunk_id(raw["chunk_id"])
        id_map[raw["chunk_id"]] = cid
        id_map[str(raw["chunk_id"])] = cid
        chunks.append({"chunk_id": cid, "text": raw["text"]})
    return chunks, id_map


def _map_ids(ids: list[Any], id_map: dict[Any, str], valid_ids: set[str]) -> list[str]:
    mapped = []
    for raw_id in ids:
        cid = id_map.get(raw_id, id_map.get(str(raw_id)))
        if cid is not None and cid in valid_ids:
            mapped.append(cid)
    return list(dict.fromkeys(mapped))


def _slots_from_example(
    ex: dict[str, Any],
    id_map: dict[Any, str],
    valid_ids: set[str],
    num_slots: int,
    dedupe_slots: bool,
) -> list[dict[str, Any]]:
    slots = []
    used: set[str] = set()
    support_groups = ex.get("support_groups") or []

    if support_groups:
        for group in support_groups:
            pivot_chunks = _map_ids(group.get("chunk_ids", []), id_map, valid_ids)
            if dedupe_slots:
                pivot_chunks = [cid for cid in pivot_chunks if cid not in used]
                used.update(pivot_chunks)
            if pivot_chunks:
                slots.append(
                    {
                        "slot_id": f"S{len(slots) + 1}",
                        "slot_name": group.get("description"),
                        "pivot_chunks": pivot_chunks,
                    }
                )
            if len(slots) >= num_slots:
                break

    if not slots:
        for cid in _map_ids(ex.get("support_ids", []), id_map, valid_ids)[:num_slots]:
            slots.append({"slot_id": f"S{len(slots) + 1}", "pivot_chunks": [cid]})

    return slots[:num_slots]


def convert_example(
    ex: dict[str, Any],
    num_slots: int,
    max_chunks: int,
    target_style: str,
    marker_style: str,
    dedupe_slots: bool,
) -> dict[str, Any]:
    chunks, id_map = _convert_chunks(ex.get("context_chunks", []), max_chunks=max_chunks)
    valid_chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    valid_set = set(valid_chunk_ids)
    gold_support_chunks = _map_ids(ex.get("support_ids", []), id_map, valid_set)
    gold_slots = _slots_from_example(ex, id_map, valid_set, num_slots, dedupe_slots)
    target = (
        format_flat_target(gold_support_chunks)
        if target_style == "flat"
        else format_target(gold_slots, num_slots=num_slots)
    )

    return {
        "id": ex.get("id", ""),
        "chunks": chunks,
        "question": ex["question"],
        "prompt": format_prompt(
            chunks,
            ex["question"],
            num_slots=num_slots,
            marker_style=marker_style,
        ),
        "target": target,
        "gold_support_chunks": gold_support_chunks,
        "gold_slots": gold_slots,
        "valid_chunk_ids": valid_chunk_ids,
        "metadata": {
            "difficulty": ex.get("difficulty"),
            "target_style": target_style,
            "marker_style": marker_style,
            "dedupe_slots": dedupe_slots,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--num_slots", type=int, default=4)
    parser.add_argument("--max_chunks", type=int, default=64)
    parser.add_argument("--target_style", choices=["slots", "flat"], default="slots")
    parser.add_argument("--marker_style", choices=["xml", "paragraph"], default="xml")
    parser.add_argument("--dedupe_slots", action="store_true")
    args = parser.parse_args()

    records = [
        convert_example(
            ex,
            num_slots=args.num_slots,
            max_chunks=args.max_chunks,
            target_style=args.target_style,
            marker_style=args.marker_style,
            dedupe_slots=args.dedupe_slots,
        )
        for ex in _load_jsonl(args.input_path)
    ]

    for split, split_records in _split(records).items():
        _write_jsonl(args.output_dir / f"{split}_sft.jsonl", split_records)

    print(f"Wrote {len(records)} processed examples to {args.output_dir}")


if __name__ == "__main__":
    main()
