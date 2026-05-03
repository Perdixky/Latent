from __future__ import annotations

import argparse
import inspect

import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

import _bootstrap  # noqa: F401

from msp.data.dataset import SlotPivotSFTDataset
from msp.train.collator import CausalLMCollator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name"],
        use_fast=cfg.get("tokenizer_use_fast", True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = cfg["max_length"]

    model_kwargs = {}
    if cfg.get("bf16", True):
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif cfg.get("fp16", False):
        model_kwargs["torch_dtype"] = torch.float16
    if cfg.get("device_map") is not None:
        model_kwargs["device_map"] = cfg.get("device_map")
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], **model_kwargs)

    if cfg.get("use_lora", True):
        lora_kwargs = {
            "r": cfg.get("lora_r", 16),
            "lora_alpha": cfg.get("lora_alpha", 32),
            "lora_dropout": cfg.get("lora_dropout", 0.05),
            "bias": "lora_only",
            "task_type": "CAUSAL_LM",
        }
        if cfg.get("target_modules") is not None:
            lora_kwargs["target_modules"] = cfg["target_modules"]
        if cfg.get("modules_to_save") is not None:
            lora_kwargs["modules_to_save"] = cfg["modules_to_save"]
        model = get_peft_model(model, LoraConfig(**lora_kwargs))
        model.print_trainable_parameters()

    train_ds = SlotPivotSFTDataset(cfg["train_path"], tokenizer, max_length=cfg["max_length"])
    valid_ds = SlotPivotSFTDataset(cfg["valid_path"], tokenizer, max_length=cfg["max_length"])

    training_kwargs = {
        "output_dir": cfg["output_dir"],
        "seed": cfg.get("seed", 42),
        "per_device_train_batch_size": cfg["per_device_train_batch_size"],
        "per_device_eval_batch_size": cfg["per_device_eval_batch_size"],
        "gradient_accumulation_steps": cfg["gradient_accumulation_steps"],
        "learning_rate": cfg["learning_rate"],
        "weight_decay": cfg.get("weight_decay", 0.01),
        "lr_scheduler_type": cfg.get("lr_scheduler_type", "cosine"),
        "warmup_ratio": cfg.get("warmup_ratio", 0.03),
        "optim": cfg.get("optim", "adamw_torch"),
        "num_train_epochs": cfg["num_train_epochs"],
        "logging_steps": cfg.get("logging_steps", 20),
        "eval_strategy": "steps",
        "eval_steps": cfg.get("eval_steps", 500),
        "save_strategy": "steps",
        "save_steps": cfg.get("save_steps", 500),
        "save_total_limit": cfg.get("save_total_limit", 2),
        "load_best_model_at_end": cfg.get("load_best_model_at_end", True),
        "metric_for_best_model": cfg.get("metric_for_best_model", "eval_loss"),
        "greater_is_better": cfg.get("greater_is_better", False),
        "dataloader_num_workers": cfg.get("dataloader_num_workers", 4),
        "label_smoothing_factor": cfg.get("label_smoothing_factor", 0.0),
        "bf16": cfg.get("bf16", True),
        "fp16": cfg.get("fp16", False),
        "gradient_checkpointing": cfg.get("gradient_checkpointing", False),
        "report_to": cfg.get("report_to", "none"),
        "remove_unused_columns": False,
    }
    if cfg.get("max_steps") is not None:
        training_kwargs["max_steps"] = cfg["max_steps"]
    training_args = TrainingArguments(**training_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": valid_ds,
        "data_collator": CausalLMCollator(tokenizer),
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(cfg["output_dir"])


if __name__ == "__main__":
    main()
