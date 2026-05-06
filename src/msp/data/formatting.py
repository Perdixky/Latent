from __future__ import annotations

from typing import Any


def _chunk_number(chunk_id: str) -> int:
    if chunk_id.startswith("C") and chunk_id[1:].isdigit():
        return int(chunk_id[1:])
    return 0


def format_document(
    chunks: list[dict[str, str]],
    marker_style: str = "xml",
) -> str:
    if marker_style not in {"xml", "paragraph"}:
        raise ValueError("marker_style must be 'xml' or 'paragraph'")

    if marker_style == "paragraph":
        parts = ["<document>"]
        for idx, ch in enumerate(chunks, start=1):
            parts.append(f"Paragraph {idx}:")
            parts.append(ch["text"].strip())
        parts.append("</document>")
        return "\n".join(parts)

    parts = ["<document>"]
    for ch in chunks:
        parts.append(f'<chunk id="{ch["chunk_id"]}">')
        parts.append(ch["text"].strip())
        parts.append("</chunk>")
    parts.append("</document>")
    return "\n".join(parts)


def format_prompt(
    chunks: list[dict[str, str]],
    question: str,
    num_slots: int = 4,
    marker_style: str = "xml",
) -> str:
    doc = format_document(chunks, marker_style=marker_style)
    task = f"""<task>
请输出最多 {num_slots} 个 evidence slots。
每个 slot 只输出 pivot chunk ids。
允许不同 slot 重复使用同一个 chunk。
输出格式必须严格如下：
<slot id="S1">[C001, C002]</slot>
<slot id="S2">[C003]</slot>
...
</task>"""
    return f"""{doc}

<question>
{question.strip()}
</question>

{task}

<answer>
"""


def format_target(gold_slots: list[dict[str, Any]], num_slots: int | None = None) -> str:
    slots = gold_slots[:num_slots] if num_slots is not None else gold_slots
    lines = []
    for i, slot in enumerate(slots, start=1):
        chunks = ", ".join(slot["pivot_chunks"])
        lines.append(f'<slot id="S{i}">[{chunks}]</slot>')
    lines.append("</answer>")
    return "\n".join(lines)


def format_flat_target(gold_support_chunks: list[str]) -> str:
    return f'<answer>[{", ".join(gold_support_chunks)}]</answer>'
