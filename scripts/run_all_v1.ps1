$ErrorActionPreference = "Stop"

$Python = ".\.venv\Scripts\python.exe"
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HOME = "D:\Code\Latent\.hf_cache"

& $Python scripts\prepare_data.py `
  --input_path data\evidence_corpus_full_shuffled.jsonl `
  --output_dir data\processed `
  --num_slots 4 `
  --max_chunks 64

& $Python scripts\train_sft.py `
  --config configs\sft_mamba.yaml

& $Python scripts\infer_sft.py `
  --model_path outputs\mamba_slot_pivot_sft `
  --input_path data\processed\test_sft.jsonl `
  --output_path outputs\mamba_slot_pivot_sft\predictions.jsonl `
  --max_new_tokens 256

& $Python scripts\eval_sft.py `
  --pred_path outputs\mamba_slot_pivot_sft\predictions.jsonl `
  --output_path outputs\mamba_slot_pivot_sft\metrics.json
