from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordLevelTrainer
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast


SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<document>",
    "</document>",
    "<question>",
    "</question>",
    "<task>",
    "</task>",
    "<answer>",
    "</answer>",
    "<chunk>",
    "</chunk>",
]


def train_tokenizer(corpus_paths: list[Path]) -> PreTrainedTokenizerFast:
    tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Split(pattern=r"\s+", behavior="removed")
    trainer = WordLevelTrainer(special_tokens=SPECIAL_TOKENS, min_frequency=1)
    tokenizer.train([str(path) for path in corpus_paths], trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="$A",
        special_tokens=[],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
        additional_special_tokens=SPECIAL_TOKENS[4:],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--n_embd", type=int, default=32)
    parser.add_argument("--n_layer", type=int, default=1)
    parser.add_argument("--n_head", type=int, default=2)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = train_tokenizer(args.corpus)
    tokenizer.save_pretrained(args.output_dir)

    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_positions=512,
        bos_token_id=tokenizer.bos_token_id,  # pyright: ignore[reportArgumentType]
        eos_token_id=tokenizer.eos_token_id,  # pyright: ignore[reportArgumentType]
        pad_token_id=tokenizer.pad_token_id,  # pyright: ignore[reportArgumentType]
    )
    model = GPT2LMHeadModel(config)
    model.save_pretrained(args.output_dir)
    print(f"Wrote tiny model to {args.output_dir}")


if __name__ == "__main__":
    main()
