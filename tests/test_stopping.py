import unittest

import tests._path  # noqa: F401

import torch

from msp.inference.stopping import StopOnTextCriteria, trim_after_stop_text


class FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(chr(int(token_id)) for token_id in token_ids)


class StoppingTest(unittest.TestCase):
    def test_trim_after_stop_text_keeps_answer_close(self):
        text = '<slot id="S1">[C001]</slot>\n</answer>\n</task>'

        self.assertEqual(
            trim_after_stop_text(text, "</answer>"),
            '<slot id="S1">[C001]</slot>\n</answer>',
        )

    def test_stop_on_text_criteria_checks_only_generated_suffix(self):
        tokenizer = FakeTokenizer()
        criteria = StopOnTextCriteria(tokenizer, prompt_length=2, stop_text="</answer>")
        generated = [ord(ch) for ch in "</answer>"]
        input_ids = torch.tensor([[1, 2, *generated]], dtype=torch.long)
        scores = torch.zeros((1, 10))

        self.assertTrue(criteria(input_ids, scores))


if __name__ == "__main__":
    unittest.main()
