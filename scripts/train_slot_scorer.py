from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer import TRAINING_ARGS_NAME

import _bootstrap  # noqa: F401

from msp.data.slot_scorer import SlotScorerCollator, SlotScorerDataset
from msp.modeling.slot_scorer import SlotScorerForCausalLM


class SlotScorerTrainer(Trainer):
    def _save(self, output_dir: str | None = None, state_dict=None) -> None:  # noqa: ANN001
        output_path = Path(output_dir or self.args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        model = self.model_wrapped if self.is_deepspeed_enabled else self.model
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(output_path)
        torch.save(state_dict or model.state_dict(), output_path / "pytorch_model.bin")

        processing_class = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        if processing_class is not None and hasattr(processing_class, "save_pretrained"):
            processing_class.save_pretrained(output_path)
        torch.save(self.args, output_path / TRAINING_ARGS_NAME)


def _resolve_resume_checkpoint(cfg: dict) -> str | None:
    resume_cfg = cfg.get("resume_from_checkpoint")
    if resume_cfg in (None, False, "false", "False", "none", "None"):
        return None
    if resume_cfg in (True, "true", "True", "auto", "latest"):
        output_dir = Path(cfg["output_dir"])
        if not output_dir.exists():
            return None
        return get_last_checkpoint(str(output_dir))
    return str(resume_cfg)


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
    base_model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], **model_kwargs)

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
        base_model = get_peft_model(base_model, LoraConfig(**lora_kwargs))
        base_model.print_trainable_parameters()

    model = SlotScorerForCausalLM(
        base_model,
        scorer_dim=cfg.get("scorer_dim"),
        alpha=cfg.get("slot_query_loss_weight", 1.0),
        beta=cfg.get("slot_chunk_bce_loss_weight", 1.0),
        pos_weight=cfg.get("pos_weight", 1.0),
    )

    train_ds = SlotScorerDataset(
        cfg["train_path"],
        tokenizer,
        max_length=cfg["max_length"],
        num_slots=cfg.get("num_slots"),
        marker_style=cfg.get("marker_style", "xml"),
    )
    valid_ds = SlotScorerDataset(
        cfg["valid_path"],
        tokenizer,
        max_length=cfg["max_length"],
        num_slots=cfg.get("num_slots"),
        marker_style=cfg.get("marker_style", "xml"),
    )

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
        "data_collator": SlotScorerCollator(tokenizer),
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = SlotScorerTrainer(**trainer_kwargs)
    resume_checkpoint = _resolve_resume_checkpoint(cfg)
    if resume_checkpoint is not None:
        print(f"Resuming slot scorer training from checkpoint: {resume_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(cfg["output_dir"])
    model.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    trainer.save_state()


if __name__ == "__main__":
    main()
