$ErrorActionPreference = "Stop"

$Python = ".\.venv\Scripts\python.exe"
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HOME = "D:\Code\Latent\.hf_cache"

& $Python scripts\prepare_data.py `
  --input_path data\evidence_corpus_full_shuffled.jsonl `
  --output_dir data\processed_tiny_smoke `
  --num_slots 4 `
  --max_chunks 8

Get-Content -Path data\processed_tiny_smoke\test_sft.jsonl -TotalCount 1 |
  Set-Content -Path data\processed_tiny_smoke\smoke_test_sft.jsonl -Encoding UTF8

& $Python scripts\train_sft.py `
  --config configs\sft_tiny_download_smoke.yaml

& $Python scripts\infer_sft.py `
  --model_path outputs\tiny_download_sft `
  --input_path data\processed_tiny_smoke\smoke_test_sft.jsonl `
  --output_path outputs\tiny_download_sft\smoke_predictions.jsonl `
  --max_input_tokens 900 `
  --max_new_tokens 16 `
  --dtype float32 `
  --no-tokenizer_use_fast

& $Python scripts\eval_sft.py `
  --pred_path outputs\tiny_download_sft\smoke_predictions.jsonl `
  --output_path outputs\tiny_download_sft\smoke_metrics.json
