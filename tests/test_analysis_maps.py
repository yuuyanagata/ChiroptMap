"""Small synthetic tests for streaming cube output and NTO semantics."""
from dataclasses import replace
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[1]

from chiroptmap.cubeio import read_cube
from chiroptmap.wfx import evaluate_mos
from test_analysis_stream import synthetic_wfx
from chiroptmap.analysis.stream import Grid, main as analyze
from chiroptmap.analysis.maps import CubeStream, main, selected_ntos


class MapExportTests(unittest.TestCase):
    def test_cube_readback_order_header_and_partial_block(self):
        wfx = synthetic_wfx()
        grid = Grid(np.array([-1., -2., -3.]), np.array([.2, .3, .4]), (2, 3, 5), "test")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.cube"
            with CubeStream(path, grid, wfx, "synthetic scalar", "Bohr") as writer:
                for block in (np.arange(7), np.arange(7, 17), np.arange(17, 30)):
                    writer.write(block / 7.)
            parsed = read_cube(path, np.float64)
            self.assertEqual(parsed.data.shape, (2, 3, 5))
            np.testing.assert_allclose(parsed.data.ravel(), np.arange(30) / 7., atol=5e-11)
            np.testing.assert_allclose(parsed.origin_bohr, grid.lower)
            self.assertFalse(path.with_name("test.cube.partial").exists())

    def test_incomplete_cube_not_promoted(self):
        wfx = synthetic_wfx()
        grid = Grid(np.zeros(3), np.ones(3), (2, 2, 2), "test")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "incomplete.cube"
            with self.assertRaises(ValueError):
                with CubeStream(path, grid, wfx, "test", "test") as writer:
                    writer.write([1., 2.])
            self.assertFalse(path.exists())
            self.assertTrue(path.with_name("incomplete.cube.partial").exists())

    def test_normalized_nto_orbitals_and_full_reconstruction(self):
        wfx = synthetic_wfx()
        normalization = (2 / math.pi) ** .75
        wfx = replace(wfx, mo_coefficients=np.diag([normalization, 2 * normalization]))
        nto_wfx, meta = selected_ntos(wfx, [1], [2], np.array([[-.8]]), [1])
        self.assertLess(meta["full_X_reconstruction_relative_error"], 1e-15)
        self.assertLess(meta["hole_MO_mixing_orthonormality_max_error"], 1e-15)
        grid = Grid(wfx.coordinates_bohr[0] - 5, np.repeat(.25, 3), (41, 41, 41), "analytic")
        overlap = np.zeros((2, 2))
        for points in grid.blocks(1001):
            phi, _ = evaluate_mos(nto_wfx, nto_wfx.mo_numbers, points)
            overlap += phi.T @ phi * grid.volume
        np.testing.assert_allclose(overlap, np.eye(2), atol=2e-14)
        # The singular value is metadata, not an orbital amplitude scaling.
        self.assertAlmostEqual(meta["selected_pairs"][0]["singular_value"], .8)

    def test_cli_separates_full_linear_map_and_nto_orbitals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wfx, log = root / "input.wfx", root / "input.log"
            wfx.write_text("""<Title>synthetic canonical</Title>
<Number of Nuclei>1</Number of Nuclei>
<Number of Electrons>2</Number of Electrons>
<Atomic Numbers>2</Atomic Numbers>
<Nuclear Charges>2</Nuclear Charges>
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
            log.write_text(""" #p HF/STO-3G TDA=(Singlets,NStates=1) IOp(9/40=4)

 Ground to excited state transition electric dipole moments (Au):
       state X Y Z Dip. S. Osc.
         1 1.0000 0.0000 0.0000 1.0000 0.1000
 Ground to excited state transition magnetic dipole moments (Au):
       state X Y Z
         1 0.0000 0.0000 1.0000

 Excited State   1:      Singlet-A      1.0000 eV 1239.84 nm f=0.1000
       1 -> 2          0.70000000
""")
            common = [
                "--repo", str(REPO), "--engine", "gaussian", "--output", str(log),
                "--wfx", str(wfx), "--skip-geometry-check", "--bounds", "-2", "-2", "-2", "2", "2", "2",
                "--spacing", "1", "--max-points", "9",
            ]
            analysis_path = root / "analysis.json"
            analyzed = analyze(common + ["--result", str(analysis_path)])
            exported = main(common + ["--outdir", str(root / "maps"), "--analysis-result", str(analysis_path)])
            self.assertEqual(exported["status"], "complete")
            self.assertEqual(len(exported["files"]), 4)
            for field in ("EDTM", "MDTM"):
                cube = read_cube(root / f"maps/selected_state_{field}_linear_projection.cube", np.float64)
                self.assertAlmostEqual(
                    float(cube.data.sum()), analyzed["analysis"]["totals"][field]["signed_projection"], places=9)
                self.assertLess(abs(exported["projection_integral_differences_from_matching_analysis"][field]), 1e-13)
            nto_cube = read_cube(root / "maps/selected_state_NTO01_hole_orbital.cube", np.float64)
            self.assertEqual(nto_cube.data.shape, (5, 5, 5))
            self.assertIn("ORBITAL", nto_cube.comment1)
            with self.assertRaises(FileExistsError):
                main(common + ["--outdir", str(root / "maps"), "--analysis-result", str(analysis_path)])
            pilot_path = root / "pilot.json"
            analyze(common + ["--pilot-points", "10", "--result", str(pilot_path)])
            with self.assertRaises(ValueError):
                main(common + ["--outdir", str(root / "bad"), "--analysis-result", str(pilot_path)])


if __name__ == "__main__":
    unittest.main()
