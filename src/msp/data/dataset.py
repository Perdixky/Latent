from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


class SlotPivotSFTDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer: Any,
        max_length: int = 8192,
    ) -> None:
        self.examples: list[dict[str, Any]] = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with Path(path).open("r", encoding="utf-8-sig") as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        prompt_ids = self.tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(ex["target"], add_special_tokens=False)["input_ids"]

        input_ids, labels = self.truncate_prompt_and_target(
            prompt_ids,
            target_ids,
            self.max_length,
        )
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "id": ex.get("id", str(idx)),
        }

    @staticmethod
    def truncate_prompt_and_target(
        prompt_ids: list[int],
        target_ids: list[int],
        max_length: int,
    ) -> tuple[list[int], list[int]]:
        if max_length <= 0:
            raise ValueError("max_length must be positive")

        if len(target_ids) >= max_length:
            kept_target = target_ids[-max_length:]
            return kept_target, kept_target.copy()

        prompt_budget = max_length - len(target_ids)
        kept_prompt = prompt_ids[-prompt_budget:] if prompt_budget > 0 else []
        input_ids = kept_prompt + target_ids
        labels = [-100] * len(kept_prompt) + target_ids
        return input_ids, labels
