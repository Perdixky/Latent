import unittest

import tests._path  # noqa: F401

from msp.inference.parser import parse_prediction


class ParserTest(unittest.TestCase):
    def test_parse_slots_and_duplicates(self):
        parsed = parse_prediction(
            '<slot id="S1">[C001, C999]</slot>\n'
            '<slot id="S2">[C001, C002]</slot>\n'
            "</answer>",
            {"C001", "C002"},
        )

        self.assertEqual(parsed["pred_chunks"], ["C001", "C002"])
        self.assertEqual(parsed["pred_chunks_with_duplicates"], ["C001", "C001", "C002"])
        self.assertEqual(parsed["invalid_ids"], ["C999"])
        self.assertTrue(parsed["exact_format"])

    def test_missing_answer_close_is_not_exact_format(self):
        parsed = parse_prediction('<slot id="S1">[C001]</slot>', {"C001"})

        self.assertFalse(parsed["exact_format"])
        self.assertEqual(parsed["slots"][0]["pivot_chunks"], ["C001"])


if __name__ == "__main__":
    unittest.main()
