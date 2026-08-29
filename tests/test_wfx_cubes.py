import csv
from dataclasses import replace
import math
from pathlib import Path

import numpy as np

from chiroptmap.cubeio import read_cube
from chiroptmap.gaussian import parse_excited_state_amplitudes
from chiroptmap.wfx import (
    evaluate_mos,
    read_wfx,
    require_restricted_canonical_mos,
)
from chiroptmap.wfx_cubes import main


def _write_synthetic_wfx(path: Path) -> None:
    path.write_text(
        """<Title>
Synthetic canonical SCF WFX
</Title>
<Number of Nuclei>
1
</Number of Nuclei>
<Number of Electrons>
2
</Number of Electrons>
<Atomic Numbers>
1
</Atomic Numbers>
<Nuclear Charges>
1.0
</Nuclear Charges>
<Nuclear Cartesian Coordinates>
0.0 0.0 0.0
</Nuclear Cartesian Coordinates>
<Number of Primitives>
2
</Number of Primitives>
<Primitive Centers>
1 1
</Primitive Centers>
<Primitive Types>
1 2
</Primitive Types>
<Primitive Exponents>
1.0 1.0
</Primitive Exponents>
<Molecular Orbital Occupation Numbers>
2.0 0.0
</Molecular Orbital Occupation Numbers>
<Molecular Orbital Energies>
-0.5 0.2
</Molecular Orbital Energies>
<Molecular Orbital Spin Types>
Alpha and Beta
Alpha and Beta
</Molecular Orbital Spin Types>
<Molecular Orbital Primitive Coefficients>
<MO Number>
1
</MO Number>
1.0 0.0
<MO Number>
2
</MO Number>
0.0 1.0
</Molecular Orbital Primitive Coefficients>
""",
        encoding="utf-8",
    )


def _write_synthetic_log(path: Path) -> None:
    path.write_text(
        """ #p HF/STO-3G TDA=(Singlets,NStates=1) IOp(9/40=4)

 Ground to excited state transition electric dipole moments (Au):
       state          X           Y           Z        Dip. S.      Osc.
         1         1.0000      0.0000      0.0000      1.0000      0.1000
 Ground to excited state transition magnetic dipole moments (Au):
       state          X           Y           Z
         1         0.0000      0.0000      1.0000

 Excited State   1:      Singlet-A      1.0000 eV  1239.84 nm  f=0.1000
       1 -> 2          0.70710678
       1 <- 2          0.10000000
""",
        encoding="utf-8",
    )


def test_wfx_parser_and_analytic_primitive_gradient(tmp_path):
    wfx_path = tmp_path / "canonical.wfx"
    _write_synthetic_wfx(wfx_path)
    wfx = read_wfx(wfx_path)

    assert wfx.n_nuclei == 1
    assert wfx.n_primitives == 2
    assert wfx.mo_coefficients.shape == (2, 2)

    points = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    phi, (gx, gy, gz) = evaluate_mos(wfx, [1, 2], points, primitive_chunk=1)
    exponential = math.exp(-0.25)

    assert np.allclose(phi[0], [1.0, 0.0])
    assert np.allclose(phi[1], [exponential, 0.5 * exponential])
    assert np.allclose(gx[0], [0.0, 1.0])
    assert math.isclose(gx[1, 1], 0.5 * exponential)
    assert np.allclose(gy, 0.0)
    assert np.allclose(gz, 0.0)


def test_parses_forward_and_reverse_response_amplitudes(tmp_path):
    log_path = tmp_path / "tda.out"
    _write_synthetic_log(log_path)
    forward, reverse = parse_excited_state_amplitudes(log_path.read_text(), 1)
    assert math.isclose(forward[(1, 2)], 0.70710678)
    assert math.isclose(reverse[(1, 2)], 0.1)


def test_rejects_post_scf_natural_orbitals(tmp_path):
    wfx_path = tmp_path / "canonical.wfx"
    _write_synthetic_wfx(wfx_path)
    canonical = read_wfx(wfx_path)
    natural = replace(
        canonical,
        occupations=np.asarray([1.8, 0.2]),
        energies=np.zeros(2),
    )

    try:
        require_restricted_canonical_mos(natural, [1, 2])
    except ValueError as exc:
        assert "post-SCF natural orbitals" in str(exc)
    else:
        raise AssertionError("A post-SCF natural-orbital WFX was not rejected.")


def test_cli_generates_edtm_and_mdtm_cubes_without_mo_cubes(tmp_path):
    wfx_path = tmp_path / "canonical.wfx"
    log_path = tmp_path / "tda.out"
    outdir = tmp_path / "cubes"
    _write_synthetic_wfx(wfx_path)
    _write_synthetic_log(log_path)

    main(
        [
            "--log",
            str(log_path),
            "--wfx",
            str(wfx_path),
            "--state",
            "1",
            "--outdir",
            str(outdir),
            "--spacing",
            "1.0",
            "--padding",
            "1.0",
            "--x-block",
            "1",
            "--primitive-chunk",
            "1",
            "--skip-geometry-check",
        ]
    )

    edtm_path = outdir / "tda_S1_contribution_map_EDTM_total.cube"
    mdtm_path = outdir / "tda_S1_contribution_map_MDTM_total.cube"
    assert edtm_path.is_file()
    assert mdtm_path.is_file()
    edtm = read_cube(edtm_path, np.float64)
    mdtm = read_cube(mdtm_path, np.float64)
    assert edtm.data.shape == (3, 3, 3)
    assert mdtm.data.shape == (3, 3, 3)
    assert np.isfinite(edtm.data).all()
    assert np.isfinite(mdtm.data).all()
    assert np.max(np.abs(edtm.data)) > 0.0

    validation_path = outdir / "tda_S1_contribution_map_moment_validation.csv"
    with validation_path.open(encoding="utf-8", newline="") as handle:
        quantities = {row["quantity"] for row in csv.DictReader(handle)}
    assert quantities == {"EDTM", "MDTM"}

    settings = (outdir / "tda_S1_contribution_map_settings.txt").read_text(
        encoding="utf-8"
    )
    assert "validation_edtm_factor=-2.0" in settings
    assert "validation_mdtm_factor=-1.0" in settings

    comparison_outdir = tmp_path / "reference_grid_cubes"
    main(
        [
            "--log",
            str(log_path),
            "--wfx",
            str(wfx_path),
            "--state",
            "1",
            "--outdir",
            str(comparison_outdir),
            "--reference-cube",
            str(edtm_path),
            "--density-mode",
            "validation",
            "--response-convention",
            "xplus_yminus",
            "--spacing",
            "9.0",
            "--x-block",
            "1",
            "--primitive-chunk",
            "1",
            "--skip-geometry-check",
        ]
    )
    reference_result = read_cube(
        comparison_outdir / "tda_S1_validation_EDTM_total.cube", np.float64
    )
    assert reference_result.data.shape == edtm.data.shape
    assert np.allclose(reference_result.origin_bohr, edtm.origin_bohr)
    assert np.allclose(reference_result.vx_bohr, edtm.vx_bohr)
