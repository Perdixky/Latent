from __future__ import annotations

from typing import Any

from msp.data.schema import Chunk


def make_chunk_id(index: int, width: int = 3) -> str:
    return f"C{index:0{width}d}"


def chunk_by_tokens(
    text: str,
    tokenizer: Any,
    max_tokens: int = 256,
    stride: int = 0,
) -> list[Chunk]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if stride < 0:
        raise ValueError("stride cannot be negative")
    if stride >= max_tokens:
        raise ValueError("stride must be smaller than max_tokens")

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks: list[Chunk] = []
    start = 0
    idx = 1

    while start < len(token_ids):
        end = min(start + max_tokens, len(token_ids))
        sub_ids = token_ids[start:end]
        chunk_text = tokenizer.decode(sub_ids)
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(idx),
                text=chunk_text,
                start_token=start,
                end_token=end,
            )
        )
        idx += 1
        if end == len(token_ids):
            break
        start = end - stride if stride > 0 else end

    return chunks
