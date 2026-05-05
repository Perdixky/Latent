# Mamba Slot Pivot

V1 explicit slot generation pipeline for chunk-level evidence selection.

## Environment

```powershell
uv venv --python 3.10
.venv\Scripts\python -m ensurepip
uv pip install -r requirements.txt
```

## V1 Flow

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HOME = "D:\Code\Latent\.hf_cache"
.venv\Scripts\python scripts\prepare_data.py --input_path data\evidence_corpus_full_shuffled.jsonl --output_dir data\processed --num_slots 4
.venv\Scripts\python scripts\train_sft.py --config configs\sft_mamba.yaml
.venv\Scripts\python scripts\infer_sft.py --model_path outputs\mamba_slot_pivot_sft --input_path data\processed\test_sft.jsonl --output_path outputs\mamba_slot_pivot_sft\predictions.jsonl
.venv\Scripts\python scripts\eval_sft.py --pred_path outputs\mamba_slot_pivot_sft\predictions.jsonl --output_path outputs\mamba_slot_pivot_sft\metrics.json
```

Training uses `state-spaces/mamba-130m-hf` by default and may require model download plus GPU memory.
The main SFT config saves full Trainer checkpoints every `save_steps` and resumes from the latest checkpoint when `resume_from_checkpoint: auto` is set.

## Downloaded Tiny Smoke Test

Use this to verify the full train -> infer -> eval path with a very small downloaded model:

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HOME = "D:\Code\Latent\.hf_cache"
.venv\Scripts\python scripts\train_sft.py --config configs\sft_tiny_download_smoke.yaml
.venv\Scripts\python scripts\infer_sft.py --model_path outputs\tiny_download_sft --input_path data\processed\test_sft.jsonl --output_path outputs\tiny_download_sft\predictions.jsonl --max_input_tokens 900 --max_new_tokens 32 --dtype float32 --no-tokenizer_use_fast
.venv\Scripts\python scripts\eval_sft.py --pred_path outputs\tiny_download_sft\predictions.jsonl --output_path outputs\tiny_download_sft\metrics.json
```

Or run the one-sample smoke script:

```powershell
.\scripts\run_tiny_download_smoke.ps1
```
