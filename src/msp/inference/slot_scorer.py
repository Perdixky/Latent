from __future__ import annotations

import re
from typing import Any

import torch


SLOT_QUERY_RE = re.compile(
    r'<slot_query\s+id="(?P<slot_id>S\d+)">\s*(?P<text>.*?)\s*</slot_query>',
    re.DOTALL,
)


def parse_slot_queries(text: str) -> list[dict[str, str]]:
    return [
        {"slot_id": match.group("slot_id"), "text": match.group("text").strip()}
        for match in SLOT_QUERY_RE.finditer(text)
    ]


def normalize_slot_queries_text(text: str) -> str:
    if "<slot_queries>" in text and "</slot_queries>" in text:
        start = text.find("<slot_queries>")
        end = text.find("</slot_queries>") + len("</slot_queries>")
        return text[start:end]
    queries = parse_slot_queries(text)
    if not queries:
        return "<slot_queries>\n</slot_queries>"
    lines = ["<slot_queries>"]
    for idx, query in enumerate(queries, start=1):
        lines.append(f'<slot_query id="S{idx}">{query["text"]}</slot_query>')
    lines.append("</slot_queries>")
    return "\n".join(lines)


def format_scored_prediction(
    slot_ids: list[str],
    chunk_ids: list[str],
    probabilities: torch.Tensor,
    threshold: float,
) -> str:
    lines = []
    for slot_idx, slot_id in enumerate(slot_ids):
        selected = [
            chunk_id
            for chunk_id, prob in zip(chunk_ids, probabilities[slot_idx].detach().cpu().tolist())
            if prob >= threshold
        ]
        lines.append(f'<slot id="{slot_id}">[{", ".join(selected)}]</slot>')
    lines.append("</answer>")
    return "\n".join(lines)


def slot_ids_from_queries(slot_queries: list[dict[str, Any]]) -> list[str]:
    return [f"S{idx}" for idx, _ in enumerate(slot_queries, start=1)]
