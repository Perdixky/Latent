import unittest

import tests._path  # noqa: F401

from msp.data.formatting import (
    format_document,
    format_flat_target,
    format_prompt,
    format_target,
)
from msp.data.dataset import SlotPivotSFTDataset


class FormattingTest(unittest.TestCase):
    def test_xml_prompt_and_slot_target(self):
        chunks = [
            {"chunk_id": "C001", "text": "First evidence."},
            {"chunk_id": "C002", "text": "Second evidence."},
        ]

        prompt = format_prompt(chunks, "What happened?", num_slots=2)
        target = format_target(
            [
                {"slot_id": "S1", "pivot_chunks": ["C001"]},
                {"slot_id": "S2", "pivot_chunks": ["C001", "C002"]},
            ]
        )

        self.assertIn('<chunk id="C001">', prompt)
        self.assertIn("<question>\nWhat happened?\n</question>", prompt)
        self.assertTrue(prompt.endswith("<answer>\n"))
        self.assertEqual(
            target,
            '<slot id="S1">[C001]</slot>\n'
            '<slot id="S2">[C001, C002]</slot>\n'
            "</answer>",
        )

    def test_paragraph_marker_ablation(self):
        doc = format_document(
            [{"chunk_id": "C001", "text": "Text"}],
            marker_style="paragraph",
        )

        self.assertIn("Paragraph 1:", doc)
        self.assertNotIn('<chunk id="C001">', doc)

    def test_flat_target(self):
        self.assertEqual(
            format_flat_target(["C003", "C001", "C003"]),
            "<answer>[C003, C001, C003]</answer>",
        )

    def test_sft_dataset_truncation_keeps_target_labels(self):
        input_ids, labels = SlotPivotSFTDataset.truncate_prompt_and_target(
            prompt_ids=list(range(10)),
            target_ids=[100, 101],
            max_length=5,
        )

        self.assertEqual(input_ids, [7, 8, 9, 100, 101])
        self.assertEqual(labels, [-100, -100, -100, 100, 101])


if __name__ == "__main__":
    unittest.main()
