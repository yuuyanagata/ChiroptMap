"""WFX parsing and direct molecular-orbital grid evaluation.

WFX molecular-orbital coefficients multiply *unnormalized* Cartesian Gaussian
primitives.  The coefficients stored in the file therefore already contain the
primitive normalization and contraction factors needed to evaluate an MO as

    phi(r) = sum_p c_p x_p**lx y_p**ly z_p**lz exp(-alpha_p r_p**2).

The transition-density workflow requires a restricted, canonical SCF WFX that
also contains the virtual orbitals.  A post-SCF TD/TDA WFX contains natural
orbitals and is deliberately rejected because Gaussian's printed ``i -> a``
coefficients refer to the canonical ground-state MO basis.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from .constants import BOHR_PER_ANGSTROM
from .models import Cube


_FLOAT_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?")


@dataclass(frozen=True)
class WFXWavefunction:
    """The WFX fields needed for direct real-space MO evaluation."""

    title: str
    atomic_numbers: np.ndarray
    nuclear_charges: np.ndarray
    coordinates_bohr: np.ndarray
    n_electrons: int
    primitive_centers: np.ndarray
    primitive_types: np.ndarray
    primitive_exponents: np.ndarray
    occupations: np.ndarray
    energies: np.ndarray
    spin_types: Tuple[str, ...]
    mo_numbers: Tuple[int, ...]
    mo_coefficients: np.ndarray

    @property
    def n_nuclei(self) -> int:
        return int(self.atomic_numbers.size)

    @property
    def n_primitives(self) -> int:
        return int(self.primitive_exponents.size)

    @property
    def n_mos(self) -> int:
        return int(len(self.mo_numbers))


def _section(text: str, tag: str, *, required: bool = True) -> Optional[str]:
    match = re.search(
        rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1)
    if required:
        raise ValueError(f"Required WFX section <{tag}> was not found.")
    return None


def _numbers(section: str, dtype: np.dtype) -> np.ndarray:
    tokens = _FLOAT_RE.findall(section)
    if not tokens:
        return np.empty(0, dtype=dtype)
    cleaned = [token.replace("D", "E").replace("d", "e") for token in tokens]
    return np.asarray(cleaned, dtype=dtype)


def read_wfx(path: Path) -> WFXWavefunction:
    """Read a Gaussian/AIM-style WFX file without an XML dependency."""

    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="ignore")

    title = (_section(text, "Title", required=False) or path.stem).strip()
    n_nuclei = int(_numbers(_section(text, "Number of Nuclei") or "", np.int64)[0])
    n_primitives = int(_numbers(_section(text, "Number of Primitives") or "", np.int64)[0])
    n_electrons = int(_numbers(_section(text, "Number of Electrons") or "", np.int64)[0])

    atomic_numbers = _numbers(_section(text, "Atomic Numbers") or "", np.int64)
    nuclear_charges = _numbers(_section(text, "Nuclear Charges") or "", np.float64)
    coordinates = _numbers(_section(text, "Nuclear Cartesian Coordinates") or "", np.float64)
    primitive_centers = _numbers(_section(text, "Primitive Centers") or "", np.int64) - 1
    primitive_types = _numbers(_section(text, "Primitive Types") or "", np.int64)
    primitive_exponents = _numbers(_section(text, "Primitive Exponents") or "", np.float64)
    occupations = _numbers(_section(text, "Molecular Orbital Occupation Numbers") or "", np.float64)
    energies = _numbers(_section(text, "Molecular Orbital Energies") or "", np.float64)

    spin_section = _section(text, "Molecular Orbital Spin Types") or ""
    spin_types = tuple(line.strip() for line in spin_section.splitlines() if line.strip())

    if atomic_numbers.size != n_nuclei or nuclear_charges.size != n_nuclei:
        raise ValueError("WFX nuclear arrays do not match <Number of Nuclei>.")
    if coordinates.size != 3 * n_nuclei:
        raise ValueError("WFX nuclear coordinate count is inconsistent.")
    if not (
        primitive_centers.size
        == primitive_types.size
        == primitive_exponents.size
        == n_primitives
    ):
        raise ValueError("WFX primitive arrays do not match <Number of Primitives>.")
    if np.any(primitive_centers < 0) or np.any(primitive_centers >= n_nuclei):
        raise ValueError("WFX contains an invalid primitive center index.")

    mo_section = _section(text, "Molecular Orbital Primitive Coefficients") or ""
    mo_re = re.compile(
        r"<MO Number>\s*(\d+)\s*</MO Number>\s*(.*?)"
        r"(?=<MO Number>|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    mo_numbers = []
    mo_coefficients = []
    for match in mo_re.finditer(mo_section):
        number = int(match.group(1))
        coeffs = _numbers(match.group(2), np.float64)
        if coeffs.size != n_primitives:
            raise ValueError(
                f"MO {number} has {coeffs.size} primitive coefficients; expected {n_primitives}."
            )
        mo_numbers.append(number)
        mo_coefficients.append(coeffs)

    if not mo_coefficients:
        raise ValueError("No MO primitive-coefficient blocks were parsed from the WFX file.")
    if len(set(mo_numbers)) != len(mo_numbers):
        raise ValueError("Duplicate <MO Number> values were found in the WFX file.")

    coeff_matrix = np.vstack(mo_coefficients)
    n_mos = len(mo_numbers)
    if occupations.size != n_mos or energies.size != n_mos or len(spin_types) != n_mos:
        raise ValueError(
            "WFX MO occupations, energies, spin types, and coefficient blocks have different lengths."
        )

    return WFXWavefunction(
        title=title,
        atomic_numbers=atomic_numbers,
        nuclear_charges=nuclear_charges,
        coordinates_bohr=coordinates.reshape((n_nuclei, 3)),
        n_electrons=n_electrons,
        primitive_centers=primitive_centers,
        primitive_types=primitive_types,
        primitive_exponents=primitive_exponents,
        occupations=occupations,
        energies=energies,
        spin_types=spin_types,
        mo_numbers=tuple(mo_numbers),
        mo_coefficients=coeff_matrix,
    )


_LOW_L_POWERS = {
    1: (0, 0, 0),
    2: (1, 0, 0),
    3: (0, 1, 0),
    4: (0, 0, 1),
    5: (2, 0, 0),
    6: (0, 2, 0),
    7: (0, 0, 2),
    8: (1, 1, 0),
    9: (1, 0, 1),
    10: (0, 1, 1),
    11: (3, 0, 0),
    12: (0, 3, 0),
    13: (0, 0, 3),
    14: (2, 1, 0),
    15: (2, 0, 1),
    16: (0, 2, 1),
    17: (1, 2, 0),
    18: (1, 0, 2),
    19: (0, 1, 2),
    20: (1, 1, 1),
    21: (4, 0, 0),
    22: (0, 4, 0),
    23: (0, 0, 4),
    24: (3, 1, 0),
    25: (3, 0, 1),
    26: (1, 3, 0),
    27: (0, 3, 1),
    28: (1, 0, 3),
    29: (0, 1, 3),
    30: (2, 2, 0),
    31: (2, 0, 2),
    32: (0, 2, 2),
    33: (2, 1, 1),
    34: (1, 2, 1),
    35: (1, 1, 2),
}


def primitive_angular_powers(type_number: int) -> Tuple[int, int, int]:
    """Map a WFX primitive type number to Cartesian powers ``(lx, ly, lz)``."""

    if type_number in _LOW_L_POWERS:
        return _LOW_L_POWERS[type_number]
    if type_number < 1:
        raise ValueError(f"Invalid WFX primitive type: {type_number}")

    # WFX defines H and higher shells by IX=0..L, IY=0..L-IX,
    # IZ=L-IX-IY.  Types 1..35 cover S through G.
    offset = 35
    angular_momentum = 5
    while True:
        count = (angular_momentum + 1) * (angular_momentum + 2) // 2
        if type_number <= offset + count:
            position = type_number - offset - 1
            cursor = 0
            for ix in range(angular_momentum + 1):
                for iy in range(angular_momentum - ix + 1):
                    if cursor == position:
                        return ix, iy, angular_momentum - ix - iy
                    cursor += 1
            break
        offset += count
        angular_momentum += 1
    raise ValueError(f"Could not decode WFX primitive type: {type_number}")


def require_restricted_canonical_mos(
    wfx: WFXWavefunction,
    needed_mos: Sequence[int],
    *,
    occupation_tolerance: float = 1.0e-5,
) -> None:
    """Reject post-SCF natural-orbital WFX data and missing virtual orbitals."""

    if wfx.n_electrons % 2:
        raise ValueError("Only closed-shell restricted WFX data are currently supported.")
    if any(spin.lower() != "alpha and beta" for spin in wfx.spin_types):
        raise ValueError("Only restricted 'Alpha and Beta' spatial orbitals are supported.")

    n_occ = wfx.n_electrons // 2
    if wfx.n_mos <= n_occ:
        raise ValueError(
            "The WFX contains no virtual orbitals. Generate a separate ground-state SCF WFX "
            "with Output=WFX and IOp(99/18=-1)."
        )

    expected = np.zeros(wfx.n_mos, dtype=float)
    expected[:n_occ] = 2.0
    occupation_error = float(np.max(np.abs(wfx.occupations - expected)))
    energies_are_zero = bool(np.all(np.abs(wfx.energies) < 1.0e-12))
    if occupation_error > occupation_tolerance or energies_are_zero:
        raise ValueError(
            "This WFX contains post-SCF natural orbitals (fractional occupations and/or zero "
            "orbital energies), but the Gaussian excitation coefficients refer to canonical "
            "ground-state MOs. Generate a separate ground-state SCF WFX at the identical "
            "geometry/method/basis with 'Output=WFX IOp(99/18=-1)', and pass that file here."
        )

    available = set(wfx.mo_numbers)
    missing = sorted(set(int(mo) for mo in needed_mos) - available)
    if missing:
        raise ValueError(
            f"The WFX is missing required canonical MO numbers: {missing}. "
            "Regenerate it with IOp(99/18=-1)."
        )


def build_cube_grid(
    wfx: WFXWavefunction,
    *,
    padding_bohr: float = 6.0,
    spacing_bohr: float = 0.20,
    dtype: np.dtype = np.float64,
) -> Cube:
    """Create an axis-aligned Gaussian-cube grid around the WFX nuclei."""

    if padding_bohr <= 0.0:
        raise ValueError("padding_bohr must be positive.")
    if spacing_bohr <= 0.0:
        raise ValueError("spacing_bohr must be positive.")

    lower = np.min(wfx.coordinates_bohr, axis=0) - padding_bohr
    upper = np.max(wfx.coordinates_bohr, axis=0) + padding_bohr
    counts = np.ceil((upper - lower) / spacing_bohr).astype(int) + 1
    counts = np.maximum(counts, 2)
    nx, ny, nz = (int(value) for value in counts)

    atoms_block = [
        f"{int(z):5d} {float(charge): .6f} {coord[0]: .6f} {coord[1]: .6f} {coord[2]: .6f}"
        for z, charge, coord in zip(wfx.atomic_numbers, wfx.nuclear_charges, wfx.coordinates_bohr)
    ]

    return Cube(
        comment1="WFX direct-grid template",
        comment2="Coordinates and axis vectors are in Bohr",
        natoms_raw=wfx.n_nuclei,
        natoms=wfx.n_nuclei,
        nvals=None,
        val_ids=None,
        origin_bohr=lower.astype(float),
        nx=nx,
        ny=ny,
        nz=nz,
        vx_bohr=np.array([spacing_bohr, 0.0, 0.0]),
        vy_bohr=np.array([0.0, spacing_bohr, 0.0]),
        vz_bohr=np.array([0.0, 0.0, spacing_bohr]),
        atoms_block=atoms_block,
        data=np.zeros((nx, ny, nz), dtype=dtype),
        unit_scale_to_bohr=1.0,
    )


def evaluate_mos(
    wfx: WFXWavefunction,
    mo_numbers: Sequence[int],
    points_bohr: np.ndarray,
    *,
    dtype: np.dtype = np.float64,
    primitive_chunk: int = 64,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Evaluate selected MOs and their analytic gradients at Cartesian points."""

    if primitive_chunk < 1:
        raise ValueError("primitive_chunk must be at least 1.")
    points = np.asarray(points_bohr, dtype=dtype)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_bohr must have shape (n_points, 3).")

    row_for_mo = {number: row for row, number in enumerate(wfx.mo_numbers)}
    try:
        rows = [row_for_mo[int(number)] for number in mo_numbers]
    except KeyError as exc:
        raise ValueError(f"MO {exc.args[0]} is not present in the WFX file.") from exc

    selected_coefficients = np.asarray(wfx.mo_coefficients[rows, :], dtype=dtype)
    n_points = points.shape[0]
    n_mos = len(rows)
    phi = np.zeros((n_points, n_mos), dtype=dtype)
    gx = np.zeros_like(phi)
    gy = np.zeros_like(phi)
    gz = np.zeros_like(phi)

    powers = np.asarray(
        [primitive_angular_powers(int(value)) for value in wfx.primitive_types],
        dtype=np.int64,
    )
    centers = np.asarray(wfx.coordinates_bohr[wfx.primitive_centers], dtype=dtype)
    exponents = np.asarray(wfx.primitive_exponents, dtype=dtype)

    for p0 in range(0, wfx.n_primitives, primitive_chunk):
        p1 = min(wfx.n_primitives, p0 + primitive_chunk)
        center = centers[p0:p1]
        alpha = exponents[p0:p1][None, :]
        angular = powers[p0:p1]
        lx = angular[:, 0][None, :]
        ly = angular[:, 1][None, :]
        lz = angular[:, 2][None, :]

        dx = points[:, 0, None] - center[None, :, 0]
        dy = points[:, 1, None] - center[None, :, 1]
        dz = points[:, 2, None] - center[None, :, 2]
        exponential = np.exp(-alpha * (dx * dx + dy * dy + dz * dz))

        px = np.power(dx, lx)
        py = np.power(dy, ly)
        pz = np.power(dz, lz)
        basis = px * py * pz * exponential

        px_prev = np.where(lx > 0, lx * np.power(dx, np.maximum(lx - 1, 0)), 0.0)
        py_prev = np.where(ly > 0, ly * np.power(dy, np.maximum(ly - 1, 0)), 0.0)
        pz_prev = np.where(lz > 0, lz * np.power(dz, np.maximum(lz - 1, 0)), 0.0)

        basis_gx = px_prev * py * pz * exponential - 2.0 * alpha * dx * basis
        basis_gy = px * py_prev * pz * exponential - 2.0 * alpha * dy * basis
        basis_gz = px * py * pz_prev * exponential - 2.0 * alpha * dz * basis

        coefficients = selected_coefficients[:, p0:p1].T
        phi += basis @ coefficients
        gx += basis_gx @ coefficients
        gy += basis_gy @ coefficients
        gz += basis_gz @ coefficients

    return phi, (gx, gy, gz)


def parse_last_standard_orientation(log_text: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return the final Gaussian Standard-orientation atoms and Bohr coordinates."""

    starts = [match.start() for match in re.finditer(r"Standard orientation:", log_text)]
    if not starts:
        return None
    tail = log_text[starts[-1]:]
    rows = []
    row_re = re.compile(
        r"^\s*\d+\s+(\d+)\s+\d+\s+"
        r"([+-]?\d*\.\d+(?:[EeDd][+-]?\d+)?)\s+"
        r"([+-]?\d*\.\d+(?:[EeDd][+-]?\d+)?)\s+"
        r"([+-]?\d*\.\d+(?:[EeDd][+-]?\d+)?)\s*$",
        flags=re.MULTILINE,
    )
    for match in row_re.finditer(tail):
        rows.append(
            (
                int(match.group(1)),
                float(match.group(2).replace("D", "E")),
                float(match.group(3).replace("D", "E")),
                float(match.group(4).replace("D", "E")),
            )
        )
        # Stop before a later unrelated numeric table. A standard-orientation
        # atom table is contiguous; the first non-row after rows is a separator.
        after = tail[match.end():]
        if after.lstrip().startswith("---"):
            break
    if not rows:
        return None
    atomic_numbers = np.asarray([row[0] for row in rows], dtype=np.int64)
    coordinates_bohr = BOHR_PER_ANGSTROM * np.asarray([row[1:] for row in rows], dtype=float)
    return atomic_numbers, coordinates_bohr
