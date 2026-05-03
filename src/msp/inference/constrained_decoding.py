from __future__ import annotations

import re
from collections.abc import Iterable

import torch
from transformers import LogitsProcessor


CHUNK_PREFIX_RE = re.compile(r"C\d*$")
COMPLETE_CHUNK_RE = re.compile(r"^C\d{3,6}$")


class ChunkIdConstrainedLogitsProcessor(LogitsProcessor):
    """Constrain generated chunk ids to the ids present in the current document.

    The processor is intentionally local: it only masks logits when the generated
    suffix is currently a chunk-id prefix, or when a candidate token itself is a
    full chunk id. A complete valid id can still continue if that continuation is
    a prefix of a longer valid id, so documents may safely contain both C001 and
    C0018. That keeps XML tags and punctuation unconstrained while preventing
    invalid ids such as C088 in the id position.
    """

    def __init__(
        self,
        tokenizer,
        valid_chunk_ids: Iterable[str],
        prompt_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.valid_chunk_ids = set(valid_chunk_ids)
        self.prompt_length = prompt_length
        self.vocab_size = len(tokenizer)
        self._token_text_cache: dict[int, str] = {}

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        filtered = scores.clone()
        for row in range(input_ids.shape[0]):
            generated_ids = input_ids[row, self.prompt_length :]
            generated_text = self._decode(generated_ids.tolist())
            allowed = self._allowed_token_ids(generated_text)
            if allowed is not None:
                mask = torch.full_like(filtered[row], float("-inf"))
                mask[list(allowed)] = filtered[row, list(allowed)]
                filtered[row] = mask
        return filtered

    def _allowed_token_ids(self, generated_text: str) -> set[int] | None:
        prefix_match = CHUNK_PREFIX_RE.search(generated_text)
        prefix = prefix_match.group(0) if prefix_match else ""

        allowed: set[int] = set()
        restrict = bool(prefix)
        for token_id in range(self.vocab_size):
            token_text = self._token_text(token_id)
            if not token_text:
                continue

            if prefix:
                if prefix in self.valid_chunk_ids and not token_text[0].isdigit():
                    allowed.add(token_id)
                    continue
                candidate = prefix + token_text
                if self._can_continue_chunk(candidate):
                    allowed.add(token_id)
            elif COMPLETE_CHUNK_RE.match(token_text):
                restrict = True
                if token_text in self.valid_chunk_ids:
                    allowed.add(token_id)

        if not restrict:
            return None
        return allowed

    def _can_continue_chunk(self, candidate: str) -> bool:
        if any(cid.startswith(candidate) for cid in self.valid_chunk_ids):
            return True
        if candidate in self.valid_chunk_ids:
            return True
        if not re.fullmatch(r"C\d+", candidate):
            return False
        return any(cid.startswith(candidate) for cid in self.valid_chunk_ids)

    def _token_text(self, token_id: int) -> str:
        if token_id not in self._token_text_cache:
            self._token_text_cache[token_id] = self._decode([token_id])
        return self._token_text_cache[token_id]

    def _decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
