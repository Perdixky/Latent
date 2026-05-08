from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _gather_positions(hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    batch_size, _, hidden_size = hidden_states.shape
    safe_positions = positions.clamp(min=0, max=hidden_states.shape[1] - 1)
    gather_index = safe_positions.unsqueeze(-1).expand(batch_size, safe_positions.shape[1], hidden_size)
    return hidden_states.gather(1, gather_index)


class SlotChunkScorer(nn.Module):
    def __init__(self, hidden_size: int, scorer_dim: int | None = None, pos_weight: float = 1.0) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.scorer_dim = scorer_dim or hidden_size
        self.slot_projection = nn.Linear(hidden_size, self.scorer_dim, bias=False)
        self.chunk_projection = nn.Linear(hidden_size, self.scorer_dim, bias=False)
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)), persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        slot_positions: torch.Tensor,
        chunk_positions: torch.Tensor,
        slot_mask: torch.Tensor | None = None,
        chunk_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        slot_hidden = _gather_positions(hidden_states, slot_positions)
        chunk_hidden = _gather_positions(hidden_states, chunk_positions)
        slot_repr = self.slot_projection(slot_hidden)
        chunk_repr = self.chunk_projection(chunk_hidden)
        scores = torch.einsum("bsd,bcd->bsc", slot_repr, chunk_repr) / math.sqrt(self.scorer_dim)
        if slot_mask is not None and chunk_mask is not None:
            scores = scores.masked_fill(~(slot_mask[:, :, None] & chunk_mask[:, None, :]), 0.0)
        return scores

    def bce_loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        pos_weight: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pos_weight is None:
            pos_weight_tensor = self.pos_weight.to(scores.device)
        else:
            pos_weight_tensor = torch.as_tensor(pos_weight, device=scores.device, dtype=scores.dtype)
        raw_loss = F.binary_cross_entropy_with_logits(
            scores,
            labels.to(dtype=scores.dtype),
            reduction="none",
            pos_weight=pos_weight_tensor,
        )
        masked_loss = raw_loss * mask.to(dtype=raw_loss.dtype)
        denom = mask.sum().clamp_min(1).to(dtype=raw_loss.dtype)
        return masked_loss.sum() / denom


def _hidden_size_from_config(config: Any) -> int:
    for attr in ("hidden_size", "d_model", "n_embd"):
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)
    raise ValueError("Could not infer hidden size from model config")


class SlotScorerForCausalLM(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        scorer_dim: int | None = None,
        alpha: float = 1.0,
        beta: float = 1.0,
        pos_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.alpha = alpha
        self.beta = beta
        hidden_size = _hidden_size_from_config(base_model.config)
        self.slot_chunk_scorer = SlotChunkScorer(hidden_size, scorer_dim=scorer_dim, pos_weight=pos_weight)

    @property
    def config(self) -> Any:
        return self.base_model.config

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return self.base_model.generate(*args, **kwargs)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        slot_positions: torch.Tensor | None = None,
        chunk_positions: torch.Tensor | None = None,
        slot_chunk_labels: torch.Tensor | None = None,
        slot_chunk_mask: torch.Tensor | None = None,
        slot_position_mask: torch.Tensor | None = None,
        chunk_position_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )
        slot_query_loss = outputs.loss if labels is not None else None
        result: dict[str, torch.Tensor] = {"logits": outputs.logits}
        if slot_query_loss is not None:
            result["slot_query_loss"] = slot_query_loss

        if (
            slot_positions is not None
            and chunk_positions is not None
            and slot_position_mask is not None
            and chunk_position_mask is not None
        ):
            hidden_states = outputs.hidden_states[-1]
            scores = self.slot_chunk_scorer(
                hidden_states,
                slot_positions,
                chunk_positions,
                slot_position_mask,
                chunk_position_mask,
            )
            result["slot_chunk_scores"] = scores
            if slot_chunk_labels is not None and slot_chunk_mask is not None:
                bce_loss = self.slot_chunk_scorer.bce_loss(scores, slot_chunk_labels, slot_chunk_mask)
                result["slot_chunk_bce_loss"] = bce_loss
                if slot_query_loss is not None:
                    result["loss"] = self.alpha * slot_query_loss + self.beta * bce_loss
                else:
                    result["loss"] = bce_loss
        elif slot_query_loss is not None:
            result["loss"] = slot_query_loss

        return result

    def save_pretrained(self, save_directory: str | Path, **kwargs: Any) -> None:
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        if hasattr(self.base_model, "save_pretrained"):
            self.base_model.save_pretrained(path, **kwargs)
        torch.save(
            {
                "slot_chunk_scorer": self.slot_chunk_scorer.state_dict(),
                "alpha": self.alpha,
                "beta": self.beta,
            },
            path / "slot_scorer.pt",
        )
