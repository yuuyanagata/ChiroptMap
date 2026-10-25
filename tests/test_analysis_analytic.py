from dataclasses import replace
import math
from pathlib import Path
import sys
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[1]

from test_analysis_stream import synthetic_wfx
from chiroptmap.analysis.analytic import axis_integrals, analytic_moments, primitive_operator_block


class AnalyticAuditTests(unittest.TestCase):
    def test_axis_s_overlap_position_and_derivative(self):
        alpha, beta = np.array([.7]), np.array([1.3])
        a, b = np.array([-.4]), np.array([.6])
        s, r, d = axis_integrals(alpha, beta, a, b, np.array([0]), np.array([0]))
        p = alpha[0] + beta[0]
        center = (alpha[0] * a[0] + beta[0] * b[0]) / p
        overlap = math.sqrt(math.pi / p) * math.exp(-alpha[0] * beta[0] / p * (a[0] - b[0]) ** 2)
        self.assertAlmostEqual(s[0, 0], overlap)
        self.assertAlmostEqual(r[0, 0], center * overlap)
        self.assertAlmostEqual(d[0, 0], 4 * alpha[0] * beta[0] / p * (a[0] - b[0]) * overlap)

    def test_s_px_moments_and_normalized_overlap(self):
        wfx = synthetic_wfx()
        c, i0 = .7, (math.pi / 2) ** 1.5
        results, _ = analytic_moments(
            wfx, [1], [2], [np.array([[-2 * c]])], [np.array([[-c]])], block_size=1, progress=False)
        np.testing.assert_allclose(results[0]["EDTM"], [-c * i0 / 2, 0, 0], atol=1e-14)
        np.testing.assert_allclose(results[0]["MDTM"], [0, 0, -.7 * c * i0], atol=1e-14)
        np.testing.assert_allclose(results[0]["integrated_magnetic_current_proxy"], [c * i0, 0, 0], atol=1e-14)
        norm = (2 / math.pi) ** .75
        wfx = replace(wfx, mo_coefficients=np.diag([norm, 2 * norm]))
        _, diagnostics = analytic_moments(
            wfx, [1], [2], [np.array([[-2 * c]])], [np.array([[-c]])], block_size=2, progress=False)
        self.assertLess(diagnostics["probe_orthonormality_maximum_error"], 1e-14)

    def test_operator_symmetry_and_antisymmetry_for_d_primitives(self):
        alpha = np.array([.7, 1.2, .4, 1.8])
        centers = np.array([[.3, -.4, .7], [-.3, .2, .5], [0, .1, -.4], [.7, 0, .2]])
        angular = np.array([[2, 0, 0], [0, 1, 1], [1, 0, 0], [0, 0, 2]])
        operators = primitive_operator_block(alpha, centers, angular, 0, 4)
        for name, op in operators.items():
            expected = op.T if name.startswith("EDTM") or name == "overlap" else -op.T
            np.testing.assert_allclose(op, expected, atol=3e-15)


if __name__ == "__main__":
    unittest.main()
