import json
from pathlib import Path

import numpy as np

from chiroptmap.orca_json import read_orca_json_wavefunction


def _write_occupied_wfx(path: Path, coefficients: np.ndarray) -> None:
    coefficient_blocks = []
    for number, row in enumerate(coefficients, start=1):
        coefficient_blocks.extend(
            [
                "<MO Number>",
                str(number),
                "</MO Number>",
                " ".join(f"{value:.16g}" for value in row),
            ]
        )
    path.write_text(
        """<Title>
Synthetic occupied-only ORCA WFX
</Title>
<Number of Nuclei>
1
</Number of Nuclei>
<Number of Electrons>
6
</Number of Electrons>
<Atomic Numbers>
6
</Atomic Numbers>
<Nuclear Charges>
6.0
</Nuclear Charges>
<Nuclear Cartesian Coordinates>
0.0 0.0 0.0
</Nuclear Cartesian Coordinates>
<Number of Primitives>
5
</Number of Primitives>
<Primitive Centers>
1 1 1 1 1
</Primitive Centers>
<Primitive Types>
1 1 2 3 4
</Primitive Types>
<Primitive Exponents>
1.5 0.4 0.8 0.8 0.8
</Primitive Exponents>
<Molecular Orbital Occupation Numbers>
2.0 2.0 2.0
</Molecular Orbital Occupation Numbers>
<Molecular Orbital Energies>
-0.7 -0.4 -0.2
</Molecular Orbital Energies>
<Molecular Orbital Spin Types>
Alpha and Beta
Alpha and Beta
Alpha and Beta
</Molecular Orbital Spin Types>
<Molecular Orbital Primitive Coefficients>
"""
        + "\n".join(coefficient_blocks)
        + "\n</Molecular Orbital Primitive Coefficients>\n",
        encoding="utf-8",
    )


def test_orca_json_wfx_bridge_recovers_shell_local_transforms(tmp_path):
    json_mo_coefficients = np.asarray(
        [
            [1.0, 1.0, 0.0, 0.0],
            [2.0, 0.0, 1.0, 0.0],
            [3.0, 0.0, 0.0, 1.0],
            [4.0, 1.0, 2.0, 3.0],
        ]
    )
    s_transform = np.asarray([[0.5, -0.25]])
    p_transform = np.asarray(
        [
            [1.0, 2.0, 0.0],
            [0.0, -1.0, 3.0],
            [2.0, 0.0, 0.5],
        ]
    )
    expected_primitive_coefficients = np.hstack(
        [
            json_mo_coefficients[:, :1] @ s_transform,
            json_mo_coefficients[:, 1:] @ p_transform,
        ]
    )

    wfx_path = tmp_path / "occupied.wfx"
    _write_occupied_wfx(wfx_path, expected_primitive_coefficients[:3])

    molecule = {
        "Multiplicity": 1,
        "CoordinateUnits": "bohr",
        "Atoms": [
            {
                "Coords": [0.0, 0.0, 0.0],
                "Basis": [
                    {"Shell": "s", "Exponents": [1.5, 0.4]},
                    {"Shell": "p", "Exponents": [0.8]},
                ],
            }
        ],
        "MolecularOrbitals": {
            "MOs": [
                {
                    "MOCoefficients": row.tolist(),
                    "Occupancy": 2.0 if index < 3 else 0.0,
                    "OrbitalEnergy": -0.7 + 0.3 * index,
                }
                for index, row in enumerate(json_mo_coefficients)
            ]
        },
        "STEOM": {
            "OrbWin": [[0, 2, 3, 3]],
            "SingletRoots": [
                {
                    "IRoot": 1,
                    "Energy": 0.1,
                    "Xia": [[[0.11], [-0.22], [0.33]]],
                }
            ],
        },
    }
    json_path = tmp_path / "wavefunction.json"
    json_path.write_text(json.dumps({"Molecule": molecule}), encoding="utf-8")

    bridge = read_orca_json_wavefunction(json_path, wfx_path)

    np.testing.assert_allclose(
        bridge.wfx.mo_coefficients,
        expected_primitive_coefficients,
        rtol=0.0,
        atol=1.0e-13,
    )
    assert bridge.diagnostics.n_shells == 2
    assert bridge.diagnostics.n_primitives == 5
    assert bridge.diagnostics.maximum_occupied_reconstruction_error < 1.0e-13
    assert bridge.diagnostics.maximum_geometry_difference_bohr == 0.0
    assert bridge.roots[0].occupied_mos == (1, 2, 3)
    assert bridge.roots[0].virtual_mos == (4,)
    np.testing.assert_allclose(bridge.roots[0].x_amplitudes[:, 0], [0.11, -0.22, 0.33])
