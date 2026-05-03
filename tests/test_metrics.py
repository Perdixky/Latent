import unittest

import tests._path  # noqa: F401

from msp.eval.metrics import (
    distinct_slot_ratio,
    gold_duplicate_hit_rate,
    invalid_id_rate,
    slot_coverage,
    support_precision,
    support_recall,
)


class MetricsTest(unittest.TestCase):
    def test_support_metrics(self):
        self.assertEqual(support_recall(["C001"], []), 0.0)
        self.assertEqual(support_precision([], ["C001"]), 0.0)
        self.assertEqual(support_recall(["C001", "C003"], ["C001", "C002"]), 0.5)
        self.assertEqual(support_precision(["C001", "C003"], ["C001", "C002"]), 0.5)

    def test_slot_metrics(self):
        slots = [
            {"slot_id": "S1", "pivot_chunks": ["C001"]},
            {"slot_id": "S2", "pivot_chunks": ["C001"]},
            {"slot_id": "S3", "pivot_chunks": ["C003"]},
        ]

        self.assertAlmostEqual(slot_coverage(slots, ["C001", "C002"]), 2 / 3)
        self.assertEqual(gold_duplicate_hit_rate(["C001", "C001", "C003"], ["C001"]), 1.0)
        self.assertEqual(invalid_id_rate(["C999"], ["C001", "C002"]), 1 / 3)
        self.assertAlmostEqual(distinct_slot_ratio(slots), 2 / 3)


if __name__ == "__main__":
    unittest.main()
