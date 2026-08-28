"""ORCA JSON/WFX bridge for direct real-space canonical-MO evaluation.

ORCA's ``orca_2aim`` WFX contains occupied orbitals only, while the JSON
export contains all canonical orbitals in the spherical AO basis.  This module
derives the local AO-to-WFX-primitive transformation independently for every
contracted shell from the occupied orbitals shared by both files, then applies
that transformation to all JSON orbitals.  The resulting ``WFXWavefunction``
can be evaluated by ChiroptMap's existing analytical primitive code without
creating intermediate MO cube files.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .constants import BOHR_PER_ANGSTROM
from .wfx import WFXWavefunction, primitive_angular_powers, read_wfx


_SHELL_L = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}


@dataclass(frozen=True)
class OrcaJsonBridgeDiagnostics:
    n_atoms: int
    n_aos: int
    n_mos: int
    n_occupied: int
    n_primitives: int
    n_shells: int
    maximum_occupied_reconstruction_error: float
    maximum_geometry_difference_bohr: float


@dataclass(frozen=True)
class OrcaTdaRoot:
    state: int
    energy_hartree: float
    occupied_mos: Tuple[int, ...]
    virtual_mos: Tuple[int, ...]
    x_amplitudes: np.ndarray


@dataclass(frozen=True)
class OrcaJsonWavefunction:
    wfx: WFXWavefunction
    roots: Tuple[OrcaTdaRoot, ...]
    diagnostics: OrcaJsonBridgeDiagnostics


def _shell_sizes(shell_name: str, n_exponents: int) -> Tuple[int, int]:
    try:
        angular_momentum = _SHELL_L[shell_name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported ORCA JSON shell type: {shell_name!r}") from exc
    n_pure = 2 * angular_momentum + 1
    n_cartesian = (angular_momentum + 1) * (angular_momentum + 2) // 2
    return n_pure, n_cartesian * n_exponents


def _load_json(path: Path) -> Dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or "Molecule" not in data:
        raise ValueError("The ORCA JSON does not contain a Molecule object.")
    return data


def _parse_roots(molecule: Dict[str, object]) -> Tuple[OrcaTdaRoot, ...]:
    steom = molecule.get("STEOM")
    if not isinstance(steom, dict):
        raise ValueError("The ORCA JSON does not contain the excited-state root data.")
    orbital_windows = steom.get("OrbWin")
    if not isinstance(orbital_windows, list) or not orbital_windows:
        raise ValueError("The ORCA JSON does not contain an excited-state orbital window.")
    window = orbital_windows[0]
    if not isinstance(window, list) or len(window) != 4:
        raise ValueError("Unexpected ORCA OrbWin representation.")
    occupied = tuple(range(int(window[0]) + 1, int(window[1]) + 2))
    virtual = tuple(range(int(window[2]) + 1, int(window[3]) + 2))

    roots_json = steom.get("SingletRoots")
    if not isinstance(roots_json, list) or not roots_json:
        raise ValueError("The ORCA JSON contains no singlet TDA roots.")
    roots: List[OrcaTdaRoot] = []
    for root_json in roots_json:
        if not isinstance(root_json, dict):
            raise ValueError("Malformed ORCA excited-state root.")
        xia_outer = root_json.get("Xia")
        if not isinstance(xia_outer, list) or len(xia_outer) != 1:
            raise ValueError("Unexpected ORCA Xia representation.")
        amplitudes = np.asarray(xia_outer[0], dtype=np.float64)
        if amplitudes.shape != (len(occupied), len(virtual)):
            raise ValueError(
                f"ORCA root {root_json.get('IRoot')} has Xia shape {amplitudes.shape}; "
                f"expected {(len(occupied), len(virtual))}."
            )
        roots.append(
            OrcaTdaRoot(
                state=int(root_json["IRoot"]),
                energy_hartree=float(root_json["Energy"]),
                occupied_mos=occupied,
                virtual_mos=virtual,
                x_amplitudes=amplitudes,
            )
        )
    return tuple(roots)


def read_orca_json_wavefunction(json_path: Path, occupied_wfx_path: Path) -> OrcaJsonWavefunction:
    """Combine an ORCA JSON export and occupied-only WFX into all canonical MOs."""

    data = _load_json(Path(json_path))
    molecule = data["Molecule"]
    if not isinstance(molecule, dict):
        raise ValueError("Malformed ORCA Molecule object.")
    if int(molecule.get("Multiplicity", 0)) != 1:
        raise ValueError("Only restricted closed-shell ORCA JSON data are currently supported.")

    atoms = molecule.get("Atoms")
    molecular_orbitals = molecule.get("MolecularOrbitals")
    if not isinstance(atoms, list) or not isinstance(molecular_orbitals, dict):
        raise ValueError("The ORCA JSON is missing atoms or molecular orbitals.")
    mos_json = molecular_orbitals.get("MOs")
    if not isinstance(mos_json, list) or not mos_json:
        raise ValueError("The ORCA JSON contains no molecular orbitals.")

    mo_coefficients = np.asarray([mo["MOCoefficients"] for mo in mos_json], dtype=np.float64)
    occupations = np.asarray([mo["Occupancy"] for mo in mos_json], dtype=np.float64)
    energies = np.asarray([mo["OrbitalEnergy"] for mo in mos_json], dtype=np.float64)
    n_mos, n_aos = mo_coefficients.shape
    if n_mos != n_aos:
        raise ValueError(f"Expected a square canonical-MO coefficient matrix, got {mo_coefficients.shape}.")
    n_occupied = int(np.count_nonzero(occupations > 1.0))
    if n_occupied == 0 or not np.allclose(occupations[:n_occupied], 2.0, atol=1.0e-8):
        raise ValueError("Unexpected ORCA restricted-orbital occupations.")
    if not np.allclose(occupations[n_occupied:], 0.0, atol=1.0e-8):
        raise ValueError("Fractionally occupied ORCA orbitals are not supported.")

    occupied_wfx = read_wfx(Path(occupied_wfx_path))
    if occupied_wfx.n_mos != n_occupied:
        raise ValueError(
            f"ORCA WFX has {occupied_wfx.n_mos} occupied MOs, but JSON has {n_occupied}."
        )
    if occupied_wfx.n_nuclei != len(atoms):
        raise ValueError("ORCA JSON and WFX atom counts differ.")

    coordinate_units = str(molecule.get("CoordinateUnits", "")).lower()
    json_coordinates = np.asarray([atom["Coords"] for atom in atoms], dtype=np.float64)
    if coordinate_units.startswith("ang"):
        json_coordinates *= BOHR_PER_ANGSTROM
    elif coordinate_units not in {"au", "a.u.", "bohr", "bohrs"}:
        raise ValueError(f"Unsupported ORCA coordinate unit: {molecule.get('CoordinateUnits')!r}")
    geometry_difference = float(np.max(np.abs(json_coordinates - occupied_wfx.coordinates_bohr)))
    if geometry_difference > 2.0e-5:
        raise ValueError(
            f"ORCA JSON/WFX geometries differ by {geometry_difference:.3e} bohr."
        )

    all_primitive_coefficients = np.empty(
        (n_mos, occupied_wfx.n_primitives), dtype=np.float64
    )
    ao_cursor = 0
    primitive_cursor = 0
    shell_count = 0
    maximum_error = 0.0
    for atom_index, atom in enumerate(atoms):
        basis = atom.get("Basis")
        if not isinstance(basis, list):
            raise ValueError(f"Atom {atom_index} has no orbital basis in the ORCA JSON.")
        for shell in basis:
            shell_name = str(shell["Shell"]).lower()
            exponents = np.asarray(shell["Exponents"], dtype=np.float64)
            n_pure, n_primitive_functions = _shell_sizes(shell_name, exponents.size)
            ao_slice = slice(ao_cursor, ao_cursor + n_pure)
            primitive_slice = slice(
                primitive_cursor, primitive_cursor + n_primitive_functions
            )
            if ao_slice.stop > n_aos or primitive_slice.stop > occupied_wfx.n_primitives:
                raise ValueError("ORCA basis-shell traversal exceeded the AO or WFX dimensions.")

            centers = occupied_wfx.primitive_centers[primitive_slice]
            if not np.all(centers == atom_index):
                raise ValueError(
                    f"WFX primitive centers do not match JSON atom {atom_index}, shell {shell_name}."
                )
            primitive_exponents = occupied_wfx.primitive_exponents[primitive_slice]
            angular_momenta = np.asarray(
                [sum(primitive_angular_powers(int(value))) for value in occupied_wfx.primitive_types[primitive_slice]],
                dtype=np.int64,
            )
            expected_l = _SHELL_L[shell_name]
            if not np.all(angular_momenta == expected_l):
                raise ValueError(
                    f"WFX primitive types do not match JSON shell {shell_name} on atom {atom_index}."
                )
            n_cartesian = n_primitive_functions // exponents.size
            expected_exponents = np.repeat(exponents, n_cartesian)
            if not np.allclose(
                np.sort(primitive_exponents),
                np.sort(expected_exponents),
                rtol=2.0e-10,
                atol=1.0e-12,
            ):
                raise ValueError(
                    f"WFX/JSON primitive exponents differ on atom {atom_index}, shell {shell_name}."
                )

            occupied_ao = mo_coefficients[:n_occupied, ao_slice]
            occupied_primitive = occupied_wfx.mo_coefficients[:, primitive_slice]
            transform, _residuals, rank, _singular = np.linalg.lstsq(
                occupied_ao, occupied_primitive, rcond=None
            )
            if rank != n_pure:
                raise ValueError(
                    f"Could not determine a full-rank AO/primitive transform for atom {atom_index}, "
                    f"shell {shell_name}: rank={rank}, expected={n_pure}."
                )
            reconstructed = occupied_ao @ transform
            shell_error = float(np.max(np.abs(reconstructed - occupied_primitive)))
            maximum_error = max(maximum_error, shell_error)
            all_primitive_coefficients[:, primitive_slice] = (
                mo_coefficients[:, ao_slice] @ transform
            )
            ao_cursor = ao_slice.stop
            primitive_cursor = primitive_slice.stop
            shell_count += 1

    if ao_cursor != n_aos or primitive_cursor != occupied_wfx.n_primitives:
        raise ValueError(
            f"Incomplete ORCA basis traversal: AO {ao_cursor}/{n_aos}, "
            f"primitives {primitive_cursor}/{occupied_wfx.n_primitives}."
        )
    # ORCA's WFX is an ASCII export with lower decimal precision than its JSON
    # MO coefficients.  A few 1e-9 of absolute reconstruction noise is normal.
    if maximum_error > 1.0e-7:
        raise ValueError(
            f"ORCA occupied-MO primitive reconstruction error is too large: {maximum_error:.3e}."
        )

    all_mo_wfx = replace(
        occupied_wfx,
        occupations=occupations,
        energies=energies,
        spin_types=tuple("Alpha and Beta" for _ in range(n_mos)),
        mo_numbers=tuple(range(1, n_mos + 1)),
        mo_coefficients=all_primitive_coefficients,
    )
    roots = _parse_roots(molecule)
    diagnostics = OrcaJsonBridgeDiagnostics(
        n_atoms=len(atoms),
        n_aos=n_aos,
        n_mos=n_mos,
        n_occupied=n_occupied,
        n_primitives=occupied_wfx.n_primitives,
        n_shells=shell_count,
        maximum_occupied_reconstruction_error=maximum_error,
        maximum_geometry_difference_bohr=geometry_difference,
    )
    return OrcaJsonWavefunction(wfx=all_mo_wfx, roots=roots, diagnostics=diagnostics)
