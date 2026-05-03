from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import LogitsProcessorList
from transformers import AutoModelForCausalLM, AutoTokenizer

import _bootstrap  # noqa: F401

from msp.inference.constrained_decoding import ChunkIdConstrainedLogitsProcessor


def _load_model(model_path: str, model_kwargs: dict):
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--max_input_tokens", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--device_map", type=str, default=None)
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--tokenizer_use_fast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--constrained_chunk_ids", action=argparse.BooleanOptionalAction, default=True)
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
    model = _load_model(args.model_path, model_kwargs)
    model.eval()

    with args.input_path.open("r", encoding="utf-8-sig") as f:
        examples = [json.loads(line) for line in f if line.strip()]

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as out:
        for ex in tqdm(examples):
            inputs = tokenizer(
                ex["prompt"],
                return_tensors="pt",
                truncation=args.max_input_tokens is not None,
                max_length=args.max_input_tokens,
            ).to(model.device)
            logits_processor = None
            if args.constrained_chunk_ids:
                logits_processor = LogitsProcessorList(
                    [
                        ChunkIdConstrainedLogitsProcessor(
                            tokenizer=tokenizer,
                            valid_chunk_ids=set(ex["valid_chunk_ids"]),
                            prompt_length=inputs["input_ids"].shape[1],
                        )
                    ]
                )
            with torch.no_grad():
                output_ids = model.generate(  # pyright: ignore[reportAttributeAccessIssue]
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    logits_processor=logits_processor,
                )
            generated = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=False,
            )
            out.write(
                json.dumps(
                    {
                        "id": ex["id"],
                        "prediction_text": generated,
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
