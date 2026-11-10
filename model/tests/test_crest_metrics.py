"""Tests for quality-weighted and binary event-count crest metrics."""

import unittest

import numpy as np

from metrics.crest import (
    extract_event_features,
    maximum_mutual_coverage_assignment,
    raw_mask_coverage_matrices,
    ratios,
    soft_crest_counts,
)


class CrestMetricTests(unittest.TestCase):
    def test_detection_assignment_prioritizes_match_count(self):
        coverage = np.asarray([[0.9, 0.8], [0.8, 0.0]], dtype=np.float32)
        self.assertEqual(
            maximum_mutual_coverage_assignment(coverage, coverage),
            [(0, 1), (1, 0)],
        )

    def test_fragment_has_quality_credit_but_is_not_a_detection(self):
        target = np.zeros((50, 50), dtype=bool)
        prediction = np.zeros_like(target)
        target[5:45, 20] = True
        prediction[5:15, 20] = True

        counts = soft_crest_counts(prediction, target)
        result = ratios(counts)

        self.assertEqual(counts["crest_detection_tp"], 0)
        self.assertEqual(counts["crest_unmatched_predicted"], 1)
        self.assertEqual(counts["crest_unmatched_labeled"], 1)
        self.assertGreater(counts["crest_quality_sum"], 0)
        self.assertEqual(result["crest_detection_precision"], 0.0)
        self.assertEqual(result["crest_detection_recall"], 0.0)
        self.assertEqual(result["crest_unmatched_prediction_rate"], 1.0)
        self.assertEqual(result["crest_missed_crest_rate"], 1.0)

    def test_identical_crest_is_one_detection(self):
        target = np.zeros((32, 32), dtype=bool)
        target[5:25, 12] = True
        counts = soft_crest_counts(target, target)
        result = ratios(counts)

        self.assertEqual(counts["crest_detection_tp"], 1)
        self.assertEqual(counts["crest_unmatched_predicted"], 0)
        self.assertEqual(counts["crest_unmatched_labeled"], 0)
        self.assertEqual(result["crest_detection_f1"], 1.0)

    def test_tolerance_match_without_raw_overlap_is_a_detection(self):
        target = np.zeros((32, 32), dtype=bool)
        prediction = np.zeros_like(target)
        target[5:25, 12] = True
        prediction[5:25, 14] = True

        counts = soft_crest_counts(prediction, target)

        self.assertGreater(counts["crest_quality_sum"], 0)
        self.assertEqual(counts["crest_detection_tp"], 1)

        predicted = extract_event_features(prediction)
        labeled = extract_event_features(target)
        prediction_support, target_coverage = raw_mask_coverage_matrices(
            predicted, labeled
        )
        self.assertEqual(
            maximum_mutual_coverage_assignment(
                prediction_support, target_coverage
            ),
            [(0, 0)],
        )

    def test_fragment_fails_mutual_mask_coverage(self):
        target = np.zeros((50, 50), dtype=bool)
        prediction = np.zeros_like(target)
        target[5:45, 20] = True
        prediction[5:15, 20] = True
        prediction_support, target_coverage = raw_mask_coverage_matrices(
            extract_event_features(prediction),
            extract_event_features(target),
        )
        self.assertGreaterEqual(prediction_support[0, 0], 0.5)
        self.assertLess(target_coverage[0, 0], 0.5)
        self.assertEqual(
            maximum_mutual_coverage_assignment(
                prediction_support, target_coverage
            ),
            [],
        )

if __name__ == "__main__":
    unittest.main()
