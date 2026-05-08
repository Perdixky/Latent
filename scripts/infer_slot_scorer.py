from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList

import _bootstrap  # noqa: F401

from msp.data.slot_scorer import (
    format_slot_query_prompt,
    format_slot_scoring_input,
    shift_retained_positions,
    tokenize_with_marker_positions,
)
from msp.inference.slot_scorer import (
    format_scored_prediction,
    normalize_slot_queries_text,
    parse_slot_queries,
    slot_ids_from_queries,
)
from msp.inference.stopping import StopOnTextCriteria, trim_after_stop_text
from msp.modeling.slot_scorer import SlotScorerForCausalLM


def _load_base_model(model_path: str, model_kwargs: dict):
    adapter_config = Path(model_path) / "adapter_config.json"
    if not adapter_config.exists():
        return AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    from peft import PeftConfig, PeftModel

    peft_cfg = PeftConfig.from_pretrained(model_path)
    base_name = peft_cfg.base_model_name_or_path
    if base_name is None:
        raise ValueError(
            f"adapter_config.json at {model_path} has no base_model_name_or_path; "
            "cannot locate base model for inference."
        )
    base = AutoModelForCausalLM.from_pretrained(base_name, **model_kwargs)
    model = PeftModel.from_pretrained(base, model_path)
    return model.merge_and_unload()  # pyright: ignore[reportCallIssue]


def _load_slot_scorer_model(model_path: str, model_kwargs: dict) -> SlotScorerForCausalLM:
    base_model = _load_base_model(model_path, model_kwargs)
    state_path = Path(model_path) / "slot_scorer.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing slot scorer state: {state_path}")
    state = torch.load(state_path, map_location="cpu")
    model = SlotScorerForCausalLM(
        base_model,
        alpha=state.get("alpha", 1.0),
        beta=state.get("beta", 1.0),
    )
    model.slot_chunk_scorer.load_state_dict(state["slot_chunk_scorer"])
    return model


def _tokenize_with_positions(
    text: str,
    tokenizer,
    max_input_tokens: int | None,
) -> tuple[dict[str, torch.Tensor], list[int], list[int]]:
    all_ids, marker_positions = tokenize_with_marker_positions(
        text,
        tokenizer,
        {"chunk": "</chunk>", "slot_query": "</slot_query>"},
    )
    chunk_positions = marker_positions["chunk"]
    slot_positions = marker_positions["slot_query"]
    if max_input_tokens is not None and len(all_ids) > max_input_tokens:
        kept_start = len(all_ids) - max_input_tokens
        all_ids = all_ids[kept_start:]
        chunk_positions = shift_retained_positions(chunk_positions, kept_start, len(all_ids))
        slot_positions = shift_retained_positions(slot_positions, kept_start, len(all_ids))
    tensors = {
        "input_ids": torch.tensor([all_ids], dtype=torch.long),
        "attention_mask": torch.ones((1, len(all_ids)), dtype=torch.long),
    }
    return tensors, slot_positions, chunk_positions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--max_input_tokens", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num_slots", type=int, default=4)
    parser.add_argument("--device_map", type=str, default=None)
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--tokenizer_use_fast", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=args.tokenizer_use_fast)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {}
    if args.dtype == "float32":
        model_kwargs["torch_dtype"] = torch.float32
    elif args.dtype == "float16":
        model_kwargs["torch_dtype"] = torch.float16
    elif args.dtype == "bfloat16":
        model_kwargs["torch_dtype"] = torch.bfloat16
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    model = _load_slot_scorer_model(args.model_path, model_kwargs)
    model.eval()

    with args.input_path.open("r", encoding="utf-8-sig") as f:
        examples = [json.loads(line) for line in f if line.strip()]

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as out:
        for ex in tqdm(examples):
            prompt = format_slot_query_prompt(ex["chunks"], ex["question"], num_slots=args.num_slots)
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=args.max_input_tokens is not None,
                max_length=args.max_input_tokens,
            ).to(model.device)
            prompt_width = inputs["input_ids"].shape[1]
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    stopping_criteria=StoppingCriteriaList(
                        [
                            StopOnTextCriteria(
                                tokenizer=tokenizer,
                                prompt_length=prompt_width,
                                stop_text="</slot_queries>",
                            )
                        ]
                    ),
                )
            generated = trim_after_stop_text(
                tokenizer.decode(
                    output_ids[0][prompt_width:],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                stop_text="</slot_queries>",
            )
            slot_queries_text = normalize_slot_queries_text(generated)
            slot_queries = parse_slot_queries(slot_queries_text)
            if not slot_queries:
                prediction_text = "</answer>"
            else:
                scoring_text = format_slot_scoring_input(ex["chunks"], ex["question"], slot_queries_text)
                score_inputs, slot_positions, chunk_positions = _tokenize_with_positions(
                    scoring_text,
                    tokenizer,
                    args.max_input_tokens,
                )
                retained_chunk_ids = ex["valid_chunk_ids"][-len(chunk_positions) :] if chunk_positions else []
                score_inputs = {key: value.to(model.device) for key, value in score_inputs.items()}
                with torch.no_grad():
                    outputs = model(
                        **score_inputs,
                        slot_positions=torch.tensor([slot_positions], dtype=torch.long, device=model.device),
                        chunk_positions=torch.tensor([chunk_positions], dtype=torch.long, device=model.device),
                        slot_position_mask=torch.ones((1, len(slot_positions)), dtype=torch.bool, device=model.device),
                        chunk_position_mask=torch.ones((1, len(chunk_positions)), dtype=torch.bool, device=model.device),
                    )
                probabilities = torch.sigmoid(outputs["slot_chunk_scores"][0, : len(slot_queries), : len(retained_chunk_ids)])
                prediction_text = format_scored_prediction(
                    slot_ids_from_queries(slot_queries),
                    retained_chunk_ids,
                    probabilities,
                    threshold=args.threshold,
                )

            out.write(
                json.dumps(
                    {
                        "id": ex["id"],
                        "prediction_text": prediction_text,
                        "slot_queries_text": slot_queries_text,
                        "gold_support_chunks": ex["gold_support_chunks"],
                        "gold_slots": ex.get("gold_slots", []),
                        "valid_chunk_ids": ex["valid_chunk_ids"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
