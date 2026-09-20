"""Synthetic mathematical oracles independent of the actual final predictions."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from offline_math import assess, endpoints, fixed_fusion, paired_bootstrap, MARGINS


def brute_endpoints(y, p):
    matrix = np.zeros((120, 120), dtype=np.int64)
    np.add.at(matrix, (y, p), 1)
    scores = []
    for k in range(120):
        den = matrix[k, :].sum() + matrix[:, k].sum()
        scores.append(2*matrix[k, k]/den if den else 0.)
    return np.array([(y == p).mean(), (y[y >= 60] == p[y >= 60]).mean(), sum(scores)/120])


class OfflineMathTests(unittest.TestCase):
    def test_bootstrap_against_literal_sample_replication(self):
        rng = np.random.default_rng(190906)
        n = 480
        setups = np.repeat(np.arange(1, 33, 2), 30)
        labels = np.tile(np.arange(120), 4)
        reference = labels.copy(); candidate = labels.copy()
        rm = rng.random(n) < .18; cm = rng.random(n) < .20
        reference[rm] = rng.integers(0, 120, rm.sum())
        candidate[cm] = rng.integers(0, 120, cm.sum())
        order = np.arange(1, 33, 2)
        draws = rng.integers(0, 16, size=(73, 16))
        result = paired_bootstrap(labels, reference, candidate, setups, draws, order)
        expected = []
        for row in draws:
            idx = np.concatenate([np.flatnonzero(setups == order[s]) for s in row])
            expected.append(brute_endpoints(labels[idx], candidate[idx]) - brute_endpoints(labels[idx], reference[idx]))
        # Python's sequential sum of 120 fractions and NumPy's reduction differ
        # by up to 9e-16 here. 2e-15 covers roundoff, far below scientific precision.
        np.testing.assert_allclose(result, expected, atol=2e-15, rtol=0)
        np.testing.assert_allclose(endpoints(labels, reference), brute_endpoints(labels, reference), atol=2e-15, rtol=0)

    def test_gate_strict_boundary_and_p_values(self):
        delta = np.array(MARGINS)
        result = assess(delta, np.tile(delta, (10000, 1)))
        self.assertTrue(all(not x["passed"] and x["one_sided_lcb"] == x["margin"] for x in result))
        self.assertTrue(all(x["margin_null_p"] == 1. for x in result))
        zero = assess(np.zeros(3), np.zeros((10000, 3)))
        self.assertTrue(all(x["passed"] and x["margin_null_p"] == 1/10001 for x in zero))

    def test_rejects_zero_support_and_wrong_labels(self):
        y = np.array([0, 60]); s = np.array([1, 3])
        with self.assertRaisesRegex(ValueError, "zero A61"):
            paired_bootstrap(y, y, y, s, np.array([[0, 0]]), s)
        with self.assertRaises(ValueError):
            endpoints(np.array([120]), np.array([120]))

    def test_macro_f1_retains_all_120_classes(self):
        self.assertAlmostEqual(endpoints(np.array([60]), np.array([60]))[2], 1/120)

    def test_shared_fusion_can_change_decision_without_second_forward(self):
        shared = np.zeros((2, 120), dtype=np.float32)
        shared[:, 0] = 1
        fourth = np.zeros_like(shared); fourth[:, 1] = 10
        full, stream3 = fixed_fusion([shared, shared, shared, fourth])
        np.testing.assert_array_equal(full.argmax(1), [1, 1])
        np.testing.assert_array_equal(stream3.argmax(1), [0, 0])
        self.assertEqual(full.dtype, np.float32)
        with self.assertRaises(ValueError):
            fixed_fusion([shared.astype(np.float64)] * 4)


if __name__ == "__main__":
    unittest.main()
