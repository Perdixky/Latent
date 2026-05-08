from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from msp.data.formatting import format_document


def format_slot_query_prompt(
    chunks: list[dict[str, str]],
    question: str,
    num_slots: int = 4,
    marker_style: str = "xml",
) -> str:
    doc = format_document(chunks, marker_style=marker_style)
    task = f"""<task>
请输出最多 {num_slots} 个 evidence slot queries。
每个 slot_query 只描述需要寻找的证据类型或实体来源。
不要输出 chunk id 列表。
输出格式必须严格如下：
<slot_queries>
<slot_query id="S1">Supporting facts from entity or source</slot_query>
<slot_query id="S2">Supporting facts from another entity or source</slot_query>
</slot_queries>
</task>"""
    return f"""{doc}

<question>
{question.strip()}
</question>

{task}

"""


def format_slot_query_target(gold_slots: list[dict[str, Any]], num_slots: int | None = None) -> str:
    slots = gold_slots[:num_slots] if num_slots is not None else gold_slots
    lines = ["<slot_queries>"]
    for idx, slot in enumerate(slots, start=1):
        slot_name = slot.get("slot_name") or f"Evidence slot S{idx}"
        lines.append(f'<slot_query id="S{idx}">{slot_name}</slot_query>')
    lines.append("</slot_queries>")
    return "\n".join(lines)


def build_slot_chunk_labels(
    gold_slots: list[dict[str, Any]],
    valid_chunk_ids: list[str],
    max_slots: int | None = None,
    max_chunks: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    slot_count = len(gold_slots) if max_slots is None else max_slots
    chunk_count = len(valid_chunk_ids) if max_chunks is None else max_chunks
    labels = torch.zeros((slot_count, chunk_count), dtype=torch.float32)
    slot_mask = torch.zeros(slot_count, dtype=torch.bool)
    chunk_mask = torch.zeros(chunk_count, dtype=torch.bool)

    chunk_index = {chunk_id: idx for idx, chunk_id in enumerate(valid_chunk_ids[:chunk_count])}
    chunk_mask[: min(len(valid_chunk_ids), chunk_count)] = True

    for slot_idx, slot in enumerate(gold_slots[:slot_count]):
        slot_mask[slot_idx] = True
        for chunk_id in slot.get("pivot_chunks", []):
            idx = chunk_index.get(chunk_id)
            if idx is not None:
                labels[slot_idx, idx] = 1.0

    return labels, slot_mask, chunk_mask


def _marker_end_positions(input_ids: list[int], marker_ids: list[int]) -> list[int]:
    if not marker_ids:
        return []
    positions = []
    marker_len = len(marker_ids)
    for idx in range(0, len(input_ids) - marker_len + 1):
        if input_ids[idx : idx + marker_len] == marker_ids:
            positions.append(idx + marker_len - 1)
    return positions


def marker_end_positions(input_ids: list[int], marker_ids: list[int]) -> list[int]:
    return _marker_end_positions(input_ids, marker_ids)


def _offset_marker_end_positions(text: str, offsets: list[tuple[int, int]], marker: str) -> list[int]:
    positions = []
    start = 0
    while True:
        marker_start = text.find(marker, start)
        if marker_start == -1:
            break
        marker_end = marker_start + len(marker)
        token_position = None
        for idx, (token_start, token_end) in enumerate(offsets):
            if token_start < marker_end <= token_end:
                token_position = idx
                break
            if token_start < marker_end and token_end <= marker_end and token_end > marker_start:
                token_position = idx
        if token_position is not None:
            positions.append(token_position)
        start = marker_end
    return positions


def tokenize_with_marker_positions(
    text: str,
    tokenizer: Any,
    markers: dict[str, str],
) -> tuple[list[int], dict[str, list[int]]]:
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    except TypeError:
        encoded = tokenizer(text, add_special_tokens=False)

    input_ids = encoded["input_ids"]
    offsets = encoded.get("offset_mapping") if hasattr(encoded, "get") else None
    if offsets is not None:
        normalized_offsets = [(int(start), int(end)) for start, end in offsets]
        return input_ids, {
            name: _offset_marker_end_positions(text, normalized_offsets, marker)
            for name, marker in markers.items()
        }

    marker_positions = {}
    for name, marker in markers.items():
        marker_ids = tokenizer(marker, add_special_tokens=False)["input_ids"]
        marker_positions[name] = _marker_end_positions(input_ids, marker_ids)
    return input_ids, marker_positions


def _shift_retained_positions(positions: list[int], kept_start: int, kept_len: int) -> list[int]:
    shifted = []
    for pos in positions:
        if kept_start <= pos < kept_start + kept_len:
            shifted.append(pos - kept_start)
    return shifted


def shift_retained_positions(positions: list[int], kept_start: int, kept_len: int) -> list[int]:
    return _shift_retained_positions(positions, kept_start, kept_len)


def format_slot_scoring_input(
    chunks: list[dict[str, str]],
    question: str,
    slot_queries_text: str,
    marker_style: str = "xml",
) -> str:
    doc = format_document(chunks, marker_style=marker_style)
    return f"""{doc}

<question>
{question.strip()}
</question>

{slot_queries_text.strip()}
"""


class SlotScorerDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer: Any,
        max_length: int = 8192,
        num_slots: int | None = None,
        marker_style: str = "xml",
    ) -> None:
        self.examples: list[dict[str, Any]] = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_slots = num_slots
        self.marker_style = marker_style
        self.chunk_end_ids = tokenizer("</chunk>", add_special_tokens=False)["input_ids"]
        self.slot_query_end_ids = tokenizer("</slot_query>", add_special_tokens=False)["input_ids"]

        with Path(path).open("r", encoding="utf-8-sig") as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        gold_slots = ex.get("gold_slots", [])
        valid_chunk_ids = ex.get("valid_chunk_ids") or [chunk["chunk_id"] for chunk in ex["chunks"]]
        num_slots = self.num_slots if self.num_slots is not None else len(gold_slots)
        prompt = format_slot_query_prompt(
            ex["chunks"],
            ex["question"],
            num_slots=num_slots,
            marker_style=self.marker_style,
        )
        target = format_slot_query_target(gold_slots, num_slots=num_slots)

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        full_text = prompt + target
        full_input_ids, marker_positions = tokenize_with_marker_positions(
            full_text,
            self.tokenizer,
            {"chunk": "</chunk>", "slot_query": "</slot_query>"},
        )
        prompt_len = min(len(prompt_ids), len(full_input_ids))
        full_labels = [-100] * prompt_len + full_input_ids[prompt_len:]
        if eos_token_id is not None and (not full_input_ids or full_input_ids[-1] != eos_token_id):
            full_input_ids = full_input_ids + [eos_token_id]
            full_labels = full_labels + [eos_token_id]
        chunk_positions = marker_positions["chunk"]
        slot_positions = [pos for pos in marker_positions["slot_query"] if pos >= prompt_len]

        if len(full_input_ids) > self.max_length:
            kept_start = len(full_input_ids) - self.max_length
            input_ids = full_input_ids[kept_start:]
            labels = full_labels[kept_start:]
        else:
            kept_start = 0
            input_ids = full_input_ids
            labels = full_labels

        labels_matrix, slot_mask, chunk_mask = build_slot_chunk_labels(
            gold_slots,
            valid_chunk_ids,
            max_slots=num_slots,
            max_chunks=len(valid_chunk_ids),
        )
        chunk_positions = _shift_retained_positions(chunk_positions, kept_start, len(input_ids))
        slot_positions = _shift_retained_positions(slot_positions, kept_start, len(input_ids))

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
            "slot_positions": slot_positions[: labels_matrix.shape[0]],
            "chunk_positions": chunk_positions[: labels_matrix.shape[1]],
            "slot_chunk_labels": labels_matrix.tolist(),
            "slot_mask": slot_mask.tolist(),
            "chunk_mask": chunk_mask.tolist(),
            "chunk_ids": valid_chunk_ids,
            "id": ex.get("id", str(idx)),
        }


class SlotScorerCollator:
    def __init__(self, tokenizer: Any, label_pad_token_id: int = -100) -> None:
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            {
                "input_ids": [f["input_ids"] for f in features],
                "attention_mask": [f["attention_mask"] for f in features],
            },
            padding=True,
            return_tensors="pt",
        )
        max_len = batch["input_ids"].shape[1]
        batch["labels"] = torch.tensor(
            [
                f["labels"] + [self.label_pad_token_id] * (max_len - len(f["labels"]))
                for f in features
            ],
            dtype=torch.long,
        )

        max_slots = max(len(f["slot_chunk_labels"]) for f in features)
        max_chunks = max(len(f["slot_chunk_labels"][0]) if f["slot_chunk_labels"] else 0 for f in features)
        slot_labels = torch.zeros((len(features), max_slots, max_chunks), dtype=torch.float32)
        slot_chunk_mask = torch.zeros((len(features), max_slots, max_chunks), dtype=torch.bool)
        slot_positions = torch.zeros((len(features), max_slots), dtype=torch.long)
        chunk_positions = torch.zeros((len(features), max_chunks), dtype=torch.long)
        slot_position_mask = torch.zeros((len(features), max_slots), dtype=torch.bool)
        chunk_position_mask = torch.zeros((len(features), max_chunks), dtype=torch.bool)

        for batch_idx, feature in enumerate(features):
            labels = torch.tensor(feature["slot_chunk_labels"], dtype=torch.float32)
            slots, chunks = labels.shape
            slot_labels[batch_idx, :slots, :chunks] = labels

            slot_mask = torch.tensor(feature["slot_mask"], dtype=torch.bool)
            chunk_mask = torch.tensor(feature["chunk_mask"], dtype=torch.bool)
            positions_slots = feature["slot_positions"][:slots]
            positions_chunks = feature["chunk_positions"][:chunks]
            slot_positions[batch_idx, : len(positions_slots)] = torch.tensor(positions_slots, dtype=torch.long)
            chunk_positions[batch_idx, : len(positions_chunks)] = torch.tensor(positions_chunks, dtype=torch.long)
            slot_position_mask[batch_idx, : len(positions_slots)] = True
            chunk_position_mask[batch_idx, : len(positions_chunks)] = True

            valid_slot_mask = slot_mask[:slots] & slot_position_mask[batch_idx, :slots]
            valid_chunk_mask = chunk_mask[:chunks] & chunk_position_mask[batch_idx, :chunks]
            slot_chunk_mask[batch_idx, :slots, :chunks] = valid_slot_mask[:, None] & valid_chunk_mask[None, :]

        batch["slot_positions"] = slot_positions
        batch["chunk_positions"] = chunk_positions
        batch["slot_position_mask"] = slot_position_mask
        batch["chunk_position_mask"] = chunk_position_mask
        batch["slot_chunk_labels"] = slot_labels
        batch["slot_chunk_mask"] = slot_chunk_mask
        return batch
