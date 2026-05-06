from __future__ import annotations

import torch
from transformers import StoppingCriteria


def trim_after_stop_text(text: str, stop_text: str = "</answer>") -> str:
    end = text.find(stop_text)
    if end == -1:
        return text
    return text[: end + len(stop_text)]


class StopOnTextCriteria(StoppingCriteria):
    def __init__(self, tokenizer, prompt_length: int, stop_text: str = "</answer>") -> None:
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.stop_text = stop_text

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:  # pyright: ignore[reportIncompatibleMethodOverride]
        for row in range(input_ids.shape[0]):
            generated_ids = input_ids[row, self.prompt_length :]
            generated_text = self.tokenizer.decode(
                generated_ids.tolist(),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if self.stop_text not in generated_text:
                return False
        return True
