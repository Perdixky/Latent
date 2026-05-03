import unittest

import tests._path  # noqa: F401

import torch

from msp.inference.constrained_decoding import ChunkIdConstrainedLogitsProcessor


class FakeTokenizer:
    def __init__(self):
        self.id_to_text = {
            0: "C",
            1: "0",
            2: "1",
            3: "2",
            4: "8",
            5: ",",
            6: "C001",
            7: "C088",
            8: "</answer>",
        }

    def __len__(self):
        return len(self.id_to_text)

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(self.id_to_text[int(token_id)] for token_id in token_ids)


class ConstrainedDecodingTest(unittest.TestCase):
    def test_allows_only_valid_continuations_for_partial_chunk_id(self):
        tokenizer = FakeTokenizer()
        processor = ChunkIdConstrainedLogitsProcessor(
            tokenizer=tokenizer,
            valid_chunk_ids={"C001", "C002"},
            prompt_length=0,
        )
        input_ids = torch.tensor([[0, 1, 1]])  # C00
        scores = torch.zeros((1, len(tokenizer)))

        filtered = processor(input_ids, scores)

        self.assertFalse(torch.isneginf(filtered[0, 2]))  # 1 -> C001
        self.assertFalse(torch.isneginf(filtered[0, 3]))  # 2 -> C002
        self.assertTrue(torch.isneginf(filtered[0, 4]))  # 8 -> C008

    def test_blocks_single_token_invalid_chunk_id(self):
        tokenizer = FakeTokenizer()
        processor = ChunkIdConstrainedLogitsProcessor(
            tokenizer=tokenizer,
            valid_chunk_ids={"C001", "C002"},
            prompt_length=0,
        )
        input_ids = torch.tensor([[]], dtype=torch.long)
        scores = torch.zeros((1, len(tokenizer)))

        filtered = processor(input_ids, scores)

        self.assertFalse(torch.isneginf(filtered[0, 6]))  # C001
        self.assertTrue(torch.isneginf(filtered[0, 7]))  # C088

    def test_allows_delimiter_after_complete_valid_chunk_id(self):
        tokenizer = FakeTokenizer()
        processor = ChunkIdConstrainedLogitsProcessor(
            tokenizer=tokenizer,
            valid_chunk_ids={"C001", "C002"},
            prompt_length=0,
        )
        input_ids = torch.tensor([[0, 1, 1, 2]])  # C001
        scores = torch.zeros((1, len(tokenizer)))

        filtered = processor(input_ids, scores)

        self.assertFalse(torch.isneginf(filtered[0, 5]))  # comma delimiter
        self.assertTrue(torch.isneginf(filtered[0, 4]))  # digit would create C0018

    def test_allows_digit_after_complete_id_when_longer_valid_id_exists(self):
        tokenizer = FakeTokenizer()
        processor = ChunkIdConstrainedLogitsProcessor(
            tokenizer=tokenizer,
            valid_chunk_ids={"C001", "C0018"},
            prompt_length=0,
        )
        input_ids = torch.tensor([[0, 1, 1, 2]])  # C001
        scores = torch.zeros((1, len(tokenizer)))

        filtered = processor(input_ids, scores)

        self.assertFalse(torch.isneginf(filtered[0, 4]))  # 8 -> C0018
        self.assertFalse(torch.isneginf(filtered[0, 5]))  # comma can still end C001


if __name__ == "__main__":
    unittest.main()
