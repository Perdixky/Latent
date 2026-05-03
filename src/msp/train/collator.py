from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class CausalLMCollator:
    tokenizer: Any
    label_pad_token_id: int = -100

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]

        batch = self.tokenizer.pad(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            padding=True,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for lab in labels:
            pad_len = max_len - len(lab)
            padded_labels.append(lab + [self.label_pad_token_id] * pad_len)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch
