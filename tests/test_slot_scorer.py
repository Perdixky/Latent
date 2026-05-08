import json
import math
import re
import tempfile
import unittest
from pathlib import Path

import torch

import tests._path  # noqa: F401

from msp.eval.metrics import (
    false_negative_rate,
    false_positive_per_example,
    support_f1,
    support_f2,
)
from msp.inference.parser import parse_prediction


class WhitespaceTokenizer:
    eos_token_id = 0
    pad_token_id = 99

    def __init__(self):
        self.vocab = {"<eos>": 0, "<pad>": 99}

    def __call__(self, text, add_special_tokens=False):
        ids = []
        tokens = [
            part
            for raw in text.split()
            for part in re.split(r"(</chunk>|</slot_query>)", raw)
            if part
        ]
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab) + 1
            ids.append(self.vocab[token])
        return {"input_ids": ids}

    def pad(self, encoded, padding=True, return_tensors=None):
        max_len = max(len(ids) for ids in encoded["input_ids"])
        padded_ids = []
        masks = []
        for ids in encoded["input_ids"]:
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [self.pad_token_id] * pad_len)
            masks.append([1] * len(ids) + [0] * pad_len)
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(padded_ids, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        return {"input_ids": padded_ids, "attention_mask": masks}


class SlotScorerTest(unittest.TestCase):
    def test_slot_query_target_uses_names_without_chunk_ids(self):
        from msp.data.slot_scorer import format_slot_query_target

        target = format_slot_query_target(
            [
                {
                    "slot_id": "S9",
                    "slot_name": "Supporting facts from Valley of Blood",
                    "pivot_chunks": ["C015", "C018"],
                },
                {
                    "slot_id": "S2",
                    "slot_name": None,
                    "pivot_chunks": ["C021"],
                },
            ]
        )

        self.assertEqual(
            target,
            "<slot_queries>\n"
            '<slot_query id="S1">Supporting facts from Valley of Blood</slot_query>\n'
            '<slot_query id="S2">Evidence slot S2</slot_query>\n'
            "</slot_queries>",
        )
        self.assertNotIn("C015", target)
        self.assertNotIn("<slot id=", target)

    def test_slot_chunk_label_matrix_aligns_slots_and_chunks(self):
        from msp.data.slot_scorer import build_slot_chunk_labels

        labels, slot_mask, chunk_mask = build_slot_chunk_labels(
            gold_slots=[
                {"slot_id": "S1", "pivot_chunks": ["C002"]},
                {"slot_id": "S2", "pivot_chunks": ["C001", "C003"]},
            ],
            valid_chunk_ids=["C001", "C002", "C003", "C004"],
            max_slots=3,
            max_chunks=5,
        )

        self.assertEqual(
            labels.tolist(),
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        )
        self.assertEqual(slot_mask.tolist(), [True, True, False])
        self.assertEqual(chunk_mask.tolist(), [True, True, True, True, False])

    def test_dataset_and_collator_pad_variable_slots_and_chunks(self):
        from msp.data.slot_scorer import SlotScorerCollator, SlotScorerDataset

        records = [
            {
                "id": "a",
                "chunks": [
                    {"chunk_id": "C001", "text": "One"},
                    {"chunk_id": "C002", "text": "Two"},
                ],
                "question": "Q?",
                "gold_slots": [{"slot_name": "First", "pivot_chunks": ["C002"]}],
                "gold_support_chunks": ["C002"],
                "valid_chunk_ids": ["C001", "C002"],
            },
            {
                "id": "b",
                "chunks": [{"chunk_id": "C001", "text": "Only"}],
                "question": "Q?",
                "gold_slots": [
                    {"slot_name": "A", "pivot_chunks": ["C001"]},
                    {"slot_name": "B", "pivot_chunks": []},
                ],
                "gold_support_chunks": ["C001"],
                "valid_chunk_ids": ["C001"],
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.jsonl"
            path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            tokenizer = WhitespaceTokenizer()
            ds = SlotScorerDataset(path, tokenizer, max_length=256)
            batch = SlotScorerCollator(tokenizer)([ds[0], ds[1]])

        self.assertEqual(batch["slot_chunk_labels"].shape, (2, 2, 2))
        self.assertEqual(batch["slot_chunk_mask"].shape, (2, 2, 2))
        self.assertTrue(batch["slot_chunk_mask"][0, 0, 1])
        self.assertFalse(batch["slot_chunk_mask"][0, 1, 1])
        self.assertFalse(batch["slot_chunk_mask"][1, 0, 1])
        self.assertEqual(batch["chunk_position_mask"].tolist(), [[True, True], [True, False]])
        self.assertEqual(batch["slot_position_mask"].tolist(), [[True, False], [True, True]])

    def test_slot_chunk_scorer_shape_and_masked_bce_loss(self):
        from msp.modeling.slot_scorer import SlotChunkScorer

        scorer = SlotChunkScorer(hidden_size=4, scorer_dim=4)
        hidden_states = torch.arange(2 * 6 * 4, dtype=torch.float32).reshape(2, 6, 4)
        slot_positions = torch.tensor([[1, 2], [3, 0]])
        chunk_positions = torch.tensor([[4, 5, 0], [1, 2, 0]])
        slot_mask = torch.tensor([[True, True], [True, False]])
        chunk_mask = torch.tensor([[True, True, False], [True, True, False]])
        labels = torch.zeros((2, 2, 3), dtype=torch.float32)
        labels[0, 0, 1] = 1.0

        scores = scorer(hidden_states, slot_positions, chunk_positions, slot_mask, chunk_mask)
        loss = scorer.bce_loss(scores, labels, slot_mask[:, :, None] & chunk_mask[:, None, :])

        self.assertEqual(scores.shape, (2, 2, 3))
        self.assertTrue(math.isfinite(loss.item()))

    def test_scored_prediction_is_compatible_with_existing_parser(self):
        from msp.inference.slot_scorer import format_scored_prediction

        text = format_scored_prediction(
            slot_ids=["S1", "S2"],
            chunk_ids=["C001", "C002", "C003"],
            probabilities=torch.tensor(
                [
                    [0.10, 0.90, 0.20],
                    [0.80, 0.30, 0.70],
                ]
            ),
            threshold=0.5,
        )
        parsed = parse_prediction(text, {"C001", "C002", "C003"})

        self.assertEqual(parsed["slots"][0]["pivot_chunks"], ["C002"])
        self.assertEqual(parsed["slots"][1]["pivot_chunks"], ["C001", "C003"])
        self.assertTrue(parsed["exact_format"])

    def test_recall_oriented_metrics(self):
        self.assertEqual(support_f1(["C001"], ["C001", "C002"]), 2 / 3)
        self.assertAlmostEqual(support_f2(["C001"], ["C001", "C002"]), 5 / 9)
        self.assertEqual(false_negative_rate(["C001"], ["C001", "C002"]), 0.5)
        self.assertEqual(false_positive_per_example(["C001", "C003"], ["C001"]), 1.0)


if __name__ == "__main__":
    unittest.main()
