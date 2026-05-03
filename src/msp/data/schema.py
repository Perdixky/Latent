from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    chunk_id: str
    text: str
    start_token: int | None = None
    end_token: int | None = None


@dataclass
class Slot:
    slot_id: str
    pivot_chunks: list[str]
    slot_name: str | None = None


@dataclass
class ProcessedExample:
    id: str
    chunks: list[Chunk]
    question: str
    prompt: str
    target: str
    gold_support_chunks: list[str]
    gold_slots: list[Slot]
    valid_chunk_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_chunk_id(raw_id: Any, width: int = 3) -> str:
    if isinstance(raw_id, str):
        stripped = raw_id.strip()
        if stripped.startswith("C") and stripped[1:].isdigit():
            return f"C{int(stripped[1:]):0{width}d}"
        if stripped.isdigit():
            return f"C{int(stripped) + 1:0{width}d}"
        raise ValueError(f"Unsupported chunk id: {raw_id!r}")
    if isinstance(raw_id, int):
        return f"C{raw_id + 1:0{width}d}"
    raise ValueError(f"Unsupported chunk id: {raw_id!r}")


def chunk_to_dict(chunk: Chunk) -> dict[str, str]:
    return {"chunk_id": chunk.chunk_id, "text": chunk.text}


def slot_to_dict(slot: Slot) -> dict[str, Any]:
    record: dict[str, Any] = {
        "slot_id": slot.slot_id,
        "pivot_chunks": slot.pivot_chunks,
    }
    if slot.slot_name is not None:
        record["slot_name"] = slot.slot_name
    return record
