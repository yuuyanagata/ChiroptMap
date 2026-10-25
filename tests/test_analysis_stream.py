"""Independent analytic and original-implementation regression tests."""
from dataclasses import replace
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[1]

from chiroptmap.wfx import WFXWavefunction, evaluate_mos
from chiroptmap.analysis.stream import (
    Grid, fields_from_orbitals, integrate, load_regions, main,
    nto_decomposition, snapped_grid, voronoi_regions, becke_atomic_weights,
    contracted_wavefunction, fields_from_contracted,
)


def synthetic_wfx():
    return WFXWavefunction(
        title="Unnormalized analytic s/px test", atomic_numbers=np.array([2]),
        nuclear_charges=np.array([2.]), coordinates_bohr=np.array([[0., .7, 0.]]),
        n_electrons=2, primitive_centers=np.array([0, 0]),
        primitive_types=np.array([1, 2]), primitive_exponents=np.ones(2),
        occupations=np.array([2., 0.]), energies=np.array([-.5, .2]),
        spin_types=("Alpha and Beta", "Alpha and Beta"), mo_numbers=(1, 2),
        mo_coefficients=np.eye(2))


class StreamingTests(unittest.TestCase):
    def test_snapped_boxes_are_nested(self):
        coords = np.array([[-3.17, 1.18, -2.21], [5.33, 7.77, 1.11]])
        grids = [snapped_grid(coords, h, 8, .4) for h in (.4, .2, .1)]
        for g in grids[1:]:
            np.testing.assert_allclose(g.lower, grids[0].lower)
            np.testing.assert_allclose(g.upper, grids[0].upper)
        np.testing.assert_allclose(
            grids[0].lower + np.array([3, 4, 5]) * .4,
            grids[2].lower + np.array([12, 16, 20]) * .1)
        with self.assertRaises(ValueError):
            snapped_grid(coords, .3, 8, .4)

    def test_blocks_c_order_and_bounded(self):
        g = Grid(np.array([-1., 2., 3.]), np.array([.1, .2, .3]), (2, 3, 4), "test")
        blocks = list(g.blocks(5))
        self.assertTrue(all(len(b) <= 5 for b in blocks))
        actual = np.vstack(blocks)
        expected = np.array([g.lower + np.array([i, j, k]) * g.steps
                             for i in range(2) for j in range(3) for k in range(4)])
        np.testing.assert_allclose(actual, expected)

    def test_field_algebra_matches_original_antisymmetrized_expression(self):
        rng = np.random.default_rng(2)
        points = rng.normal(size=(17, 3))
        po, pv = rng.normal(size=(17, 2)), rng.normal(size=(17, 3))
        go = tuple(rng.normal(size=(17, 2)) for _ in range(3))
        gv = tuple(rng.normal(size=(17, 3)) for _ in range(3))
        ew, mw = rng.normal(size=(2, 3)), rng.normal(size=(2, 3))
        origin = np.array([.2, -.3, .5])
        e, m, tau, j = fields_from_orbitals(points, po, pv, go, gv, ew, mw, origin)
        expected_e, expected_m = np.zeros_like(e), np.zeros_like(m)
        for i in range(2):
            for a in range(3):
                expected_e += ew[i, a] * (points - origin) * (po[:, i] * pv[:, a])[:, None]
                expected_m += mw[i, a] * (
                    np.cross(points - origin, np.column_stack([g[:, i] for g in go])) * pv[:, a, None]
                    - po[:, i, None] * np.cross(points - origin, np.column_stack([g[:, a] for g in gv])))
        np.testing.assert_allclose(e, expected_e, atol=2e-14)
        np.testing.assert_allclose(m, expected_m, atol=2e-14)
        np.testing.assert_allclose(m, np.cross(points - origin, j), atol=2e-14)

    def test_analytic_gaussian_integrals_origin_and_phase(self):
        wfx = synthetic_wfx()
        grid = Grid(wfx.coordinates_bohr[0] - 5, np.repeat(.25, 3), (41, 41, 41), "analytic")
        c = .7
        i0 = (math.pi / 2) ** 1.5
        reference_e = np.array([c * i0 / 2, 0., 0.])
        reference_m = np.array([0., 0., .7 * c * i0])
        shifted_origin = np.array([.1, .2, .3])
        report = integrate(
            wfx, [1], [2], np.array([[-2 * c]]), np.array([[-c]]),
            reference_e, reference_m, grid, shifted_origin, ["all"], np.array([0]),
            max_points=101, primitive_chunk=1, progress=False)
        self.assertEqual(report["global_phase_factor"], -1)
        for field in ("EDTM", "MDTM"):
            self.assertLess(report["reference_validation_at_zero_origin"][field]["absolute_vector_error"], 2e-12)
            self.assertLess(report["totals"][field]["region_additivity_absolute_error"], 2e-12)
            self.assertLess(abs(report["regions"]["all"][field]["projection_balance_residual"]), 2e-12)
        self.assertLess(report["origin_shift_check"]["maximum_absolute_residual"], 2e-12)
        expected_m = np.array([0., .3 * c * i0, (.7 - .2) * c * i0])
        np.testing.assert_allclose(report["totals"]["MDTM"]["vector"], expected_m, atol=2e-12)

    def test_voronoi_tie_and_region_partition(self):
        atoms = np.array([[-1., 0., 0.], [1., 0., 0.]])
        points = np.array([[-.5, 0., 0.], [0., 0., 0.], [.5, 0., 0.]])
        np.testing.assert_array_equal(voronoi_regions(points, atoms, np.array([0, 1])), [0, 0, 1])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "regions.json"
            path.write_text('{"atom_index_base":1,"regions":{"a":[1],"b":[1,2]}}')
            with self.assertRaises(ValueError):
                load_regions(path, 2)

    def test_nto_full_reconstruction_no_threshold(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(5, 8))
        report, u, v = nto_decomposition(x)
        np.testing.assert_allclose((u * report["singular_values"]) @ v.T, x, atol=5e-14)
        self.assertLess(report["full_reconstruction_frobenius_relative_error"], 3e-15)
        self.assertAlmostEqual(sum(report["normalized_weights"]), 1.)

    def test_becke_unity_rotation_permutation_and_nuclear_limits(self):
        rng = np.random.default_rng(20)
        points, atoms = rng.normal(size=(30, 3)), rng.normal(size=(6, 3))
        weights = becke_atomic_weights(points, atoms)
        self.assertTrue(np.all(weights >= 0))
        np.testing.assert_allclose(weights.sum(axis=1), 1., atol=5e-16)
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        np.testing.assert_allclose(
            becke_atomic_weights(points @ rotation, atoms @ rotation), weights, atol=3e-15)
        permutation = np.array([4, 2, 5, 1, 0, 3])
        np.testing.assert_allclose(
            becke_atomic_weights(points, atoms[permutation]), weights[:, permutation], atol=3e-15)
        np.testing.assert_allclose(becke_atomic_weights(atoms, atoms), np.eye(6), atol=3e-15)
        with self.assertRaises(ValueError):
            becke_atomic_weights(points, np.zeros((2, 3)))

    def test_exact_contraction_arbitrary_operator_weights(self):
        rng = np.random.default_rng(9)
        wfx = synthetic_wfx()
        wfx = replace(
            wfx, mo_numbers=(1, 2, 3, 4), mo_coefficients=rng.normal(size=(4, 2)),
            occupations=np.array([2., 2., 0., 0.]), energies=np.array([-.6, -.5, .2, .4]),
            spin_types=("Alpha and Beta",) * 4)
        points, origin = rng.normal(size=(29, 3)), rng.normal(size=3)
        ew, mw = rng.normal(size=(2, 2)), rng.normal(size=(2, 2))
        phi, gradients = evaluate_mos(wfx, [1, 2, 3, 4], points)
        expected = fields_from_orbitals(
            points, phi[:, :2], phi[:, 2:], tuple(g[:, :2] for g in gradients),
            tuple(g[:, 2:] for g in gradients), ew, mw, origin)
        contracted, reuse = contracted_wavefunction(wfx, [1, 2], [3, 4], ew, mw)
        self.assertFalse(reuse)
        actual = fields_from_contracted(
            points, *evaluate_mos(contracted, contracted.mo_numbers, points), 2, reuse, origin)
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, atol=3e-15)

    def test_becke_additivity_and_incomplete_pilot_status(self):
        wfx = synthetic_wfx()
        wfx = replace(wfx, atomic_numbers=np.array([2, 1]), nuclear_charges=np.array([2., 1.]),
                      coordinates_bohr=np.array([[0., .7, 0.], [2., -.1, .1]]))
        grid = Grid(np.repeat(-2., 3), np.ones(3), (5, 5, 5), "small")
        report = integrate(
            wfx, [1], [2], np.array([[-1.4]]), np.array([[-.7]]), np.array([1., 0., 0.]),
            np.array([0., 0., 1.]), grid, np.zeros(3), ["one", "two"], np.array([0, 1]),
            max_points=21, partition="becke", evaluation="contracted", progress=False, pilot_points=57)
        self.assertFalse(report["is_complete_grid"])
        self.assertEqual(report["processed_points"], 57)
        self.assertLess(report["maximum_partition_unity_residual"], 5e-16)
        for field in ("EDTM", "MDTM"):
            self.assertLess(report["totals"][field]["region_additivity_absolute_error"], 5e-16)
        self.assertEqual(report["regions"]["one"]["assigned_grid_points"], None)

    def test_streamed_cli_matches_original_generator(self):
        """The published generator is only run on a tiny 7x7x7 test grid."""
        from chiroptmap.wfx_cubes import main as original
        from chiroptmap.cubeio import read_cube
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wfx_path, log_path = root / "test.wfx", root / "test.log"
            wfx_path.write_text("""<Title>analytic test</Title>
<Number of Nuclei>1</Number of Nuclei>
<Number of Electrons>2</Number of Electrons>
<Atomic Numbers>2</Atomic Numbers>
<Nuclear Charges>2.0</Nuclear Charges>
<Nuclear Cartesian Coordinates>0 .7 0</Nuclear Cartesian Coordinates>
<Number of Primitives>2</Number of Primitives>
<Primitive Centers>1 1</Primitive Centers>
<Primitive Types>1 2</Primitive Types>
<Primitive Exponents>1 1</Primitive Exponents>
<Molecular Orbital Occupation Numbers>2 0</Molecular Orbital Occupation Numbers>
<Molecular Orbital Energies>-.5 .2</Molecular Orbital Energies>
<Molecular Orbital Spin Types>
Alpha and Beta
Alpha and Beta
</Molecular Orbital Spin Types>
<Molecular Orbital Primitive Coefficients>
<MO Number>1</MO Number>
1 0
<MO Number>2</MO Number>
0 1
</Molecular Orbital Primitive Coefficients>
""")
            log_path.write_text(""" #p HF/STO-3G TDA=(Singlets,NStates=1) IOp(9/40=4)

 Ground to excited state transition electric dipole moments (Au):
       state X Y Z Dip. S. Osc.
         1 1.0000 0.0000 0.0000 1.0000 0.1000
 Ground to excited state transition magnetic dipole moments (Au):
       state X Y Z
         1 0.0000 0.0000 1.0000

 Excited State   1:      Singlet-A      1.0000 eV  1239.84 nm  f=0.1000
       1 -> 2          0.70000000
""")
            original([
                "--log", str(log_path), "--wfx", str(wfx_path), "--state", "1",
                "--outdir", str(root / "old"), "--spacing", "1", "--padding", "3",
                "--density-mode", "validation", "--phase-align", "edtm", "--keep-components",
                "--skip-geometry-check", "--no-cpl-csv", "--min-abs-coeff", "0",
            ])
            cube_path = root / "old/test_S1_validation_EDTM_total.cube"
            report = main([
                "--repo", str(REPO), "--engine", "gaussian", "--output", str(log_path),
                "--wfx", str(wfx_path), "--reference-cube", str(cube_path),
                "--skip-geometry-check", "--max-points", "7", "--result", str(root / "new.json"),
                "--nto-json", str(root / "nto.json"),
            ])
            for field in ("EDTM", "MDTM"):
                old = []
                for component in "xyz":
                    cube = read_cube(root / f"old/test_S1_validation_{field}_{component}.cube", np.float64)
                    old.append(float(cube.data.sum()))
                np.testing.assert_allclose(report["analysis"]["totals"][field]["vector"], old, atol=1e-6)
            self.assertEqual(report["memory_model"]["full_grid_arrays"], 0)
            self.assertEqual(report["grid"]["points"], 343)


if __name__ == "__main__":
    unittest.main()
