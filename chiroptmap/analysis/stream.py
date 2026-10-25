#!/usr/bin/env python3
"""Shared bounded-grid integrators, partitions and TDA NTO decomposition.

Parsers, AO/primitive transforms and linear excitation-weight conventions
are shared with the direct cube commands. The state-fields interface adds
same-origin analytic reference axes and multi-origin integration to these
numerical primitives.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Iterator

import numpy as np


@dataclass
class Grid:
    lower: np.ndarray
    steps: np.ndarray
    counts: tuple[int, int, int]
    source: str

    @property
    def upper(self) -> np.ndarray:
        return self.lower + self.steps * (np.asarray(self.counts) - 1)

    @property
    def volume(self) -> float:
        return float(np.prod(self.steps))

    @property
    def size(self) -> int:
        return math.prod(self.counts)

    def blocks(self, max_points: int) -> Iterator[np.ndarray]:
        """C-order/z-fastest blocks; never allocate a whole grid or x slab."""
        if max_points < 1:
            raise ValueError("max_points must be positive")
        ny, nz = self.counts[1:]
        for start in range(0, self.size, max_points):
            linear = np.arange(start, min(start + max_points, self.size), dtype=np.int64)
            ijk = np.column_stack((linear // (ny * nz), linear // nz % ny, linear % nz))
            yield self.lower + ijk * self.steps

    def metadata(self) -> dict:
        return {
            "lower_bohr": self.lower.tolist(), "upper_bohr": self.upper.tolist(),
            "steps_bohr": self.steps.tolist(), "counts": list(self.counts),
            "points": self.size, "voxel_volume_bohr3": self.volume, "source": self.source,
            "quadrature": "equal-weight Cartesian point sum, including both endpoints",
        }


def snapped_grid(coordinates: np.ndarray, spacing: float, padding: float,
                 snap: float, bounds: list[float] | None = None) -> Grid:
    if not np.isfinite([spacing, padding, snap]).all() or spacing <= 0 or padding < 0 or snap <= 0:
        raise ValueError("spacing and snap must be positive; padding must be nonnegative")
    if bounds is None:
        lower = np.floor((np.min(coordinates, axis=0) - padding) / snap) * snap
        upper = np.ceil((np.max(coordinates, axis=0) + padding) / snap) * snap
        source = "nuclear bounding box snapped outwards on absolute zero-anchored lattice"
    else:
        if len(bounds) != 6 or not np.isfinite(bounds).all():
            raise ValueError("Bounds must contain six finite coordinates")
        lower, upper = np.asarray(bounds[:3], float), np.asarray(bounds[3:], float)
        source = "explicit fixed physical bounds"
    intervals = (upper - lower) / spacing
    if np.any(intervals <= 0) or not np.allclose(intervals, np.rint(intervals), atol=1e-8, rtol=0):
        raise ValueError("Every box extent must be positive and an integer multiple of spacing")
    return Grid(lower, np.repeat(float(spacing), 3),
                tuple(int(x) + 1 for x in np.rint(intervals)), source)


def cube_header_grid(path: Path, wfx) -> Grid:
    """Read only header/atom records; never allocate reference-cube values."""
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        next(handle); next(handle)
        head = next(handle).split()
        natoms = abs(int(head[0]))
        lower = np.asarray(head[1:4], float)
        counts, axes = [], []
        for _ in range(3):
            row = next(handle).split()
            count = int(row[0])
            if count <= 0:
                raise ValueError("Reference header must use positive counts and Bohr axes")
            counts.append(count)
            axes.append([float(x) for x in row[1:4]])
        atoms = np.asarray([[float(x) for x in next(handle).split()[:5]]
                            for _ in range(natoms)])
    axes = np.asarray(axes)
    steps = np.diag(axes)
    if not np.allclose(axes, np.diag(steps), atol=1e-12, rtol=0) or np.any(steps <= 0):
        raise ValueError("Only positive axis-aligned reference grids are supported")
    if natoms != wfx.n_nuclei or not np.array_equal(atoms[:, 0], wfx.atomic_numbers):
        raise ValueError("Reference cube atom identities differ from WFX")
    if np.max(np.abs(atoms[:, 2:5] - wfx.coordinates_bohr)) > 2e-5:
        raise ValueError("Reference cube coordinates differ from WFX")
    return Grid(lower, steps, tuple(counts), str(path.resolve()))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_regions(path: Path | None, n_atoms: int) -> tuple[list[str], np.ndarray, dict]:
    if path is None:
        labels = [f"atom_{i + 1}" for i in range(n_atoms)]
        return labels, np.arange(n_atoms), {
            "definition": "one region per atom; not a chemically assigned fragment decomposition",
            "atom_index_base": 1,
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("atom_index_base") != 1 or not isinstance(data.get("regions"), dict):
        raise ValueError("Region JSON needs atom_index_base: 1 and regions: {name:[atoms]}")
    labels = list(data["regions"])
    if not labels:
        raise ValueError("Region list is empty")
    mapping = np.full(n_atoms, -1, dtype=int)
    for region, label in enumerate(labels):
        atoms = data["regions"][label]
        if not atoms:
            raise ValueError(f"Empty region: {label}")
        for atom in atoms:
            if type(atom) is not int or not 1 <= atom <= n_atoms:
                raise ValueError(f"Invalid atom index {atom} in {label}")
            if mapping[atom - 1] != -1:
                raise ValueError(f"Atom {atom} occurs in more than one region")
            mapping[atom - 1] = region
    if np.any(mapping < 0):
        raise ValueError("Every atom must belong to exactly one region")
    return labels, mapping, {"definition": data, "source_sha256": sha256(path)}


def voronoi_regions(points: np.ndarray, atoms: np.ndarray,
                    atom_to_region: np.ndarray) -> np.ndarray:
    """Equal-distance ties go to the first atom (deterministic, documented)."""
    best = np.full(len(points), np.inf)
    nearest = np.zeros(len(points), dtype=np.int64)
    for atom, center in enumerate(atoms):
        distance = np.sum((points - center) ** 2, axis=1)
        use = distance < best
        best[use] = distance[use]
        nearest[use] = atom
    return atom_to_region[nearest]


def becke_atomic_weights(points: np.ndarray, atoms: np.ndarray) -> np.ndarray:
    """Equal-radius Becke weights: three iterations of f(mu)=1.5mu-.5mu^3.

    No heteroatomic radius adjustment is applied. The weights are rotationally
    covariant in continuous space; any sampled-grid integral still has ordinary
    Cartesian quadrature/orientation dependence.
    """
    distances = np.linalg.norm(points[:, None, :] - atoms[None, :, :], axis=2)
    weights = np.ones_like(distances)
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            separation = float(np.linalg.norm(atoms[i] - atoms[j]))
            if separation < 1e-10:
                raise ValueError("Becke partition cannot distinguish coincident nuclei")
            switch = np.clip((distances[:, i] - distances[:, j]) / separation, -1, 1)
            for _ in range(3):
                switch = 1.5 * switch - .5 * switch ** 3
            switch = .5 * (1 - switch)
            weights[:, i] *= switch
            weights[:, j] *= 1 - switch
    normalization = weights.sum(axis=1)
    if np.any(normalization <= 0) or not np.isfinite(normalization).all():
        raise ValueError("Becke weight normalization failed")
    return weights / normalization[:, None]


def contracted_wavefunction(wfx, occupied, virtual, ew, mw):
    """Exact linear combinations of virtual primitive coefficients, no rank cutoff."""
    indices = {number: i for i, number in enumerate(wfx.mo_numbers)}
    occ_coeff = wfx.mo_coefficients[[indices[n] for n in occupied]]
    virt_coeff = wfx.mo_coefficients[[indices[n] for n in virtual]]
    magnetic = mw @ virt_coeff
    # TDA factors obey We=2*Wm. Require near-roundoff agreement and record reuse.
    reuse = np.allclose(ew, 2 * mw, rtol=5e-15, atol=0)
    coeff = np.vstack((occ_coeff, magnetic)) if reuse else np.vstack((occ_coeff, magnetic, ew @ virt_coeff))
    n = len(coeff)
    transformed = replace(
        wfx, title=wfx.title + " [exact weighted virtual combinations]",
        mo_coefficients=coeff, mo_numbers=tuple(range(1, n + 1)),
        occupations=np.zeros(n), energies=np.zeros(n),
        spin_types=tuple("Alpha and Beta" for _ in range(n)))
    # These combinations are not normalized canonical MOs; only evaluate_mos is
    # used on the transformed object, not canonical-orbital checks or NTO parsing.
    return transformed, bool(reuse)


def fields_from_contracted(points, phi, gradients, n_occ, reuse, origin):
    occupied, magnetic = phi[:, :n_occ], phi[:, n_occ:2 * n_occ]
    electric = 2 * magnetic if reuse else phi[:, 2 * n_occ:3 * n_occ]
    tau = np.einsum("pi,pi->p", occupied, electric)
    current = np.column_stack([
        np.einsum("pi,pi->p", g[:, :n_occ], magnetic)
        - np.einsum("pi,pi->p", occupied, g[:, n_occ:2 * n_occ])
        for g in gradients
    ])
    displaced = points - origin
    return displaced * tau[:, None], np.cross(displaced, current), tau, current


def peak_rss_bytes():
    """Peak process working set/RSS; not an incremental array-memory estimate."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes
            class Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ]
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            kernel = ctypes.windll.kernel32
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = (
                wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD)
            ctypes.windll.psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
            return int(counters.PeakWorkingSetSize) if ok else None
        import resource
        result = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(result if sys.platform == "darwin" else result * 1024)
    except (ImportError, AttributeError, OSError):
        return None


def fields_from_orbitals(points: np.ndarray, phi_occ: np.ndarray,
                         phi_virt: np.ndarray, grad_occ: tuple,
                         grad_virt: tuple, electric_weights: np.ndarray,
                         magnetic_weights: np.ndarray, origin: np.ndarray):
    """Return E=(r-O)*tau, M=(r-O) cross j, tau and real magnetic-current proxy j.

    j=sum Wm[(grad phi_i)*phi_a-phi_i*(grad phi_a)]. This equals the
    antisymmetrized r-cross-gradient expression in the original generator.
    j is a code-convention proxy, not a separately normalized physical current.
    """
    tau = np.einsum("pi,pi->p", phi_occ @ electric_weights, phi_virt, optimize=True)
    occupied_weighted = phi_occ @ magnetic_weights
    current = np.empty((len(points), 3))
    for component in range(3):
        current[:, component] = (
            np.einsum("pi,pi->p", grad_occ[component] @ magnetic_weights, phi_virt,
                      optimize=True)
            - np.einsum("pi,pi->p", occupied_weighted, grad_virt[component], optimize=True)
        )
    displaced = points - origin
    return displaced * tau[:, None], np.cross(displaced, current), tau, current


def nto_decomposition(x: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    """SVD of complete input TDA X, independently of the field threshold."""
    u, singular, vh = np.linalg.svd(x, full_matrices=False)
    recovered = (u * singular) @ vh
    norm = float(np.linalg.norm(x))
    error = float(np.linalg.norm(recovered - x))
    squared = singular ** 2
    total = float(np.sum(squared))
    fraction = squared / total if total else squared
    summary = {
        "matrix_shape": list(x.shape), "singular_values": singular.tolist(),
        "squared_singular_values": squared.tolist(), "sum_x2": total,
        "normalized_weights": fraction.tolist(),
        "cumulative_normalized_weights": np.cumsum(fraction).tolist(),
        "pairs_for_95_percent_weight": (
            int(np.searchsorted(np.cumsum(fraction), .95)) + 1 if total else None),
        "full_reconstruction_frobenius_absolute_error": error,
        "full_reconstruction_frobenius_relative_error": error / norm if norm else None,
        "meaning": "X=U diag(s) V^T; columns U mix occupied MOs, columns V mix virtual MOs",
        "cautions": [
            "NTO weight coverage is not transition-moment coverage.",
            "Near-degenerate NTOs should be interpreted as subspaces.",
            "Mixing matrices reconstruct the un-phase-aligned source amplitudes.",
        ],
    }
    return summary, u, vh.T


def error_metrics(integrated: np.ndarray, reference: np.ndarray) -> dict:
    residual = integrated - reference
    rn, vn = float(np.linalg.norm(reference)), float(np.linalg.norm(integrated))
    return {
        "reference_vector": reference.tolist(), "integrated_vector": integrated.tolist(),
        "reference_norm": rn, "integrated_norm": vn, "residual_vector": residual.tolist(),
        "absolute_vector_error": float(np.linalg.norm(residual)),
        "relative_vector_error": float(np.linalg.norm(residual)) / rn if rn else None,
        "relative_magnitude_error": abs(vn - rn) / rn if rn else None,
    }


def _load_wavefunction(args):
    from chiroptmap.wfx import read_wfx, require_restricted_canonical_mos
    from chiroptmap.gaussian import (
        diagnose_coefficients, parse_edtm_vector, parse_excited_state_amplitudes,
        parse_excited_state_block, parse_mdtm_vector,
    )
    from chiroptmap.wfx_cubes import _geometry_check, _weight_matrices
    text = args.output.read_text(encoding="utf-8", errors="ignore")
    threshold = args.min_abs_coeff
    if args.engine == "orca":
        if args.json is None:
            raise ValueError("--json is required for ORCA")
        if args.response_convention != "tda":
            raise ValueError("This ORCA bridge supports restricted singlet TDA only")
        from chiroptmap.orca import (
            parse_absorption_spectrum_all_states, parse_cd_spectrum_all_states,
        )
        from chiroptmap.orca_json import read_orca_json_wavefunction
        bridge = read_orca_json_wavefunction(args.json, args.wfx)
        root = next((r for r in bridge.roots if r.state == args.state), None)
        if root is None:
            raise ValueError("Selected ORCA TDA root was not found")
        x_full = root.x_amplitudes.copy()
        occupied_full, virtual_full = root.occupied_mos, root.virtual_mos
        mask = np.abs(x_full) >= threshold
        oi, vi = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
        selected = x_full[np.ix_(oi, vi)].copy()
        selected[~mask[np.ix_(oi, vi)]] = 0
        occ = tuple(occupied_full[i] for i in oi)
        virt = tuple(virtual_full[i] for i in vi)
        ew, mw = -math.sqrt(2) * selected, -selected / math.sqrt(2)
        e = np.asarray(parse_absorption_spectrum_all_states(text)[args.state][3:6], float)
        m_printed = np.asarray(parse_cd_spectrum_all_states(text)[args.state][1:4], float)
        m = 2 * m_printed
        wfx = bridge.wfx
        meta = {
            "bridge": asdict(bridge.diagnostics), "input_amplitudes": "complete JSON TDA Xia",
            "retained_nonzero_pairs": int(np.count_nonzero(selected)),
            "selected_sum_x2": float(np.sum(selected ** 2)),
            "full_sum_x2": float(np.sum(x_full ** 2)),
            "electric_factor_per_X": "-sqrt(2)", "magnetic_factor_per_X": "-1/sqrt(2)",
            "printed_magnetic_vector": m_printed.tolist(),
            "printed_magnetic_reference_factor": 2,
            "unit_convention_status": (
                "Matches existing ChiroptMap ORCA/Gaussian-compatible convention. "
                "The factor-of-two reference mapping is not independently established by this harness."
            ),
        }
    else:
        wfx = read_wfx(args.wfx)
        if not args.skip_geometry_check:
            _geometry_check(wfx, text)
        forward, reverse = parse_excited_state_amplitudes(text, args.state)
        if args.response_convention == "tda" and any(abs(v) > 1e-14 for v in reverse.values()):
            raise ValueError("Nonzero de-excitation amplitudes found: do not label this input TDA")
        candidates = set(forward) | (set(reverse) if args.response_convention != "tda" else set())
        pairs = sorted(pair for pair in candidates
                       if max(abs(forward.get(pair, 0)), abs(reverse.get(pair, 0))) >= threshold)
        if not pairs:
            raise ValueError("No Gaussian configurations remain at this threshold")
        occ, virt = sorted({p[0] for p in pairs}), sorted({p[1] for p in pairs})
        ew, mw = _weight_matrices(
            forward, reverse, pairs, occ, virt, np.float64, "validation",
            "forward-only" if args.response_convention == "tda" else "xplus_yminus", -2, -1)
        occupied_full = sorted({p[0] for p in forward})
        virtual_full = sorted({p[1] for p in forward})
        x_full = np.zeros((len(occupied_full), len(virtual_full)))
        oi, vi = {n: i for i, n in enumerate(occupied_full)}, {n: i for i, n in enumerate(virtual_full)}
        for pair, c in forward.items():
            x_full[oi[pair[0]], vi[pair[1]]] = math.sqrt(2) * c
        diag = diagnose_coefficients(text, args.state, parse_excited_state_block(text, args.state))
        e, m = np.asarray(parse_edtm_vector(text, args.state)), np.asarray(parse_mdtm_vector(text, args.state))
        meta = {
            "input_amplitudes": "printed Gaussian coefficients only; threshold=0 cannot restore omissions",
            "printed_coefficient_diagnostics": asdict(diag),
            "retained_nonzero_forward_pairs": sum(forward.get(p, 0) != 0 for p in pairs),
            "retained_nonzero_reverse_pairs": sum(reverse.get(p, 0) != 0 for p in pairs),
            "electric_factor_per_printed_coefficient": -2,
            "magnetic_factor_per_printed_coefficient": -1,
            "response_convention": args.response_convention,
            "full_sum_x2_of_printed_forward_only": float(np.sum(x_full ** 2)),
        }
    if not occ or not virt or not np.any(ew):
        raise ValueError("Selected transition has no nonzero electric amplitudes")
    all_mos = sorted(set(occ) | set(virt))
    require_restricted_canonical_mos(wfx, all_mos)
    if not np.isfinite(ew).all() or not np.isfinite(mw).all():
        raise ValueError("Non-finite excitation weights")
    if np.linalg.norm(e) == 0 or np.linalg.norm(m) == 0:
        raise ValueError("Reference EDTM and MDTM must both be nonzero for signed projections")
    return wfx, occ, virt, ew, mw, e, m, x_full, occupied_full, virtual_full, meta


def integrate(wfx, occupied, virtual, ew, mw, reference_e, reference_m, grid: Grid,
              origin: np.ndarray, labels: list[str], atom_to_region: np.ndarray,
              max_points=4096, primitive_chunk=64, phase_align=True, progress=True,
              partition="voronoi", evaluation="contracted", pilot_points=None) -> dict:
    from chiroptmap.wfx import evaluate_mos
    all_mos = sorted(set(occupied) | set(virtual))
    index = {n: i for i, n in enumerate(all_mos)}
    oi, vi = [index[n] for n in occupied], [index[n] for n in virtual]
    eval_wfx, reuse = (contracted_wavefunction(wfx, occupied, virtual, ew, mw)
                       if evaluation == "contracted" else (wfx, False))
    eval_mos = eval_wfx.mo_numbers if evaluation == "contracted" else all_mos
    units = np.vstack((reference_e / np.linalg.norm(reference_e),
                       reference_m / np.linalg.norm(reference_m)))
    region_vectors = np.zeros((2, len(labels), 3))
    positive, negative = np.zeros((2, len(labels))), np.zeros((2, len(labels)))
    totals = np.zeros((2, 3))
    zero_totals = np.zeros((2, 3))
    charge_sum, current_sum = 0., np.zeros(3)
    point_counts = np.zeros(len(labels), dtype=np.int64)
    effective_points = np.zeros(len(labels))
    maximum_partition_residual = 0.
    limit = min(grid.size, pilot_points) if pilot_points is not None else grid.size
    if limit < 1:
        raise ValueError("pilot_points must be positive")
    start, completed, last_log = time.perf_counter(), 0, -1.
    for points in grid.blocks(max_points):
        points = points[:min(len(points), limit - completed)]
        phi, gradients = evaluate_mos(eval_wfx, eval_mos, points, dtype=np.float64,
                                      primitive_chunk=primitive_chunk)
        if evaluation == "contracted":
            electric, magnetic, tau, current = fields_from_contracted(
                points, phi, gradients, len(occupied), reuse, origin)
        else:
            electric, magnetic, tau, current = fields_from_orbitals(
                points, phi[:, oi], phi[:, vi], tuple(g[:, oi] for g in gradients),
                tuple(g[:, vi] for g in gradients), ew, mw, origin)
        if partition == "voronoi":
            membership = voronoi_regions(points, wfx.coordinates_bohr, atom_to_region)
            point_counts += np.bincount(membership, minlength=len(labels))
            weights = None
        elif partition == "becke":
            atomic = becke_atomic_weights(points, wfx.coordinates_bohr)
            weights = np.column_stack([atomic[:, atom_to_region == i].sum(axis=1)
                                       for i in range(len(labels))])
            maximum_partition_residual = max(
                maximum_partition_residual, float(np.max(np.abs(weights.sum(axis=1) - 1))))
            effective_points += weights.sum(axis=0)
        else:
            raise ValueError(f"Unknown partition {partition}")
        for q, field in enumerate((electric, magnetic)):
            totals[q] += field.sum(axis=0, dtype=np.float64)
            projection = field @ units[q]
            if weights is None:
                for c in range(3):
                    region_vectors[q, :, c] += np.bincount(
                        membership, weights=field[:, c], minlength=len(labels))
                positive[q] += np.bincount(membership, weights=np.maximum(projection, 0),
                                          minlength=len(labels))
                negative[q] += np.bincount(membership, weights=np.maximum(-projection, 0),
                                          minlength=len(labels))
            else:
                region_vectors[q] += weights.T @ field
                positive[q] += weights.T @ np.maximum(projection, 0)
                negative[q] += weights.T @ np.maximum(-projection, 0)
        zero_totals[0] += (points * tau[:, None]).sum(axis=0, dtype=np.float64)
        zero_totals[1] += np.cross(points, current).sum(axis=0, dtype=np.float64)
        charge_sum += float(tau.sum(dtype=np.float64))
        current_sum += current.sum(axis=0, dtype=np.float64)
        completed += len(points)
        elapsed = time.perf_counter() - start
        if progress and (elapsed - last_log > 30 or completed == limit):
            print(f"[stream] {completed}/{grid.size} points ({elapsed:.1f} s)", file=sys.stderr, flush=True)
            last_log = elapsed
        if completed == limit:
            break
    phase = -1 if phase_align and np.dot(zero_totals[0], reference_e) < 0 else 1
    dvol = grid.volume
    totals *= phase * dvol
    zero_totals *= phase * dvol
    region_vectors *= phase * dvol
    charge_sum *= phase * dvol
    current_sum *= phase * dvol
    positive, negative = positive * dvol, negative * dvol
    if phase == -1:
        positive, negative = negative, positive
    predicted = np.vstack((zero_totals[0] - origin * charge_sum,
                           zero_totals[1] - np.cross(origin, current_sum)))
    output = {
        "is_complete_grid": completed == grid.size,
        "processed_points": completed, "grid_points": grid.size,
        "pilot_warning": (
            "INCOMPLETE prefix-grid pilot; integral/error/cancellation outputs are not molecular results"
            if completed < grid.size else None),
        "rough_full_grid_seconds_from_prefix": (
            (time.perf_counter() - start) * grid.size / completed if completed < grid.size else None),
        "timing_caution": "Prefix timing is only a rough resource estimate; not a benchmark or guaranteed ETA.",
        "peak_process_rss_bytes": peak_rss_bytes(),
        "orbital_evaluation": evaluation,
        "evaluated_orbital_combinations": len(eval_mos),
        "exact_contraction_uses_No_NTO_rank_cutoff": evaluation == "contracted",
        "electric_is_twice_magnetic_weight_reuse": reuse,
        "partition": partition, "maximum_partition_unity_residual": maximum_partition_residual,
        "elapsed_integration_seconds": time.perf_counter() - start,
        "global_phase_factor": phase,
        "phase_basis": "zero-origin integrated EDTM vs printed reference; same factor for both fields",
        "origin_bohr": origin.tolist(),
        "projection_reference_origin_bohr": [0, 0, 0],
        "projection_note": (
            "All signed projections use the fixed original-output reference directions. "
            "At a shifted origin these are a diagnostic projection, not transformed parent moments."
        ),
        "reference_validation_at_zero_origin": {
            "EDTM": error_metrics(zero_totals[0], reference_e),
            "MDTM": error_metrics(zero_totals[1], reference_m),
        },
        "origin_shift_check": {
            "integrated_charge_weighted_tau": charge_sum,
            "integrated_code_magnetic_current_proxy": current_sum.tolist(),
            "identities": ["E(O)=E(0)-O*integral(tau)", "M(O)=M(0)-O cross integral(j)"],
            "predicted_shifted_vectors": predicted.tolist(),
            "maximum_absolute_residual": float(np.max(np.abs(predicted - totals))),
            "independent_parent_reference_at_shifted_origin": None,
        },
        "regions": {},
        "totals": {},
    }
    for q, name in enumerate(("EDTM", "MDTM")):
        sum_region = region_vectors[q].sum(axis=0)
        norm_sum = float(np.linalg.norm(region_vectors[q], axis=1).sum())
        total_norm = float(np.linalg.norm(totals[q]))
        absolute_residual = float(np.linalg.norm(sum_region - totals[q]))
        output["totals"][name] = {
            "vector": totals[q].tolist(), "norm": total_norm,
            "reference_direction": units[q].tolist(),
            "signed_projection": float(totals[q] @ units[q]),
            "positive_projection_integral": float(positive[q].sum()),
            "negative_projection_absolute_integral": float(negative[q].sum()),
            "region_vector_cancellation_index": 1 - total_norm / norm_sum if norm_sum > 1e-14 else None,
            "region_vector_norm_sum": norm_sum,
            "region_additivity_absolute_error": absolute_residual,
            "region_additivity_relative_error": absolute_residual / total_norm if total_norm > 1e-14 else None,
        }
        for a, label in enumerate(labels):
            output["regions"].setdefault(label, {
                "assigned_grid_points": int(point_counts[a]) if partition == "voronoi" else None,
                "sum_grid_partition_weights": (float(point_counts[a]) if partition == "voronoi"
                                               else float(effective_points[a])),
            })
            output["regions"][label][name] = {
                "vector": region_vectors[q, a].tolist(),
                "norm": float(np.linalg.norm(region_vectors[q, a])),
                "signed_projection": float(region_vectors[q, a] @ units[q]),
                "positive_projection_integral": float(positive[q, a]),
                "negative_projection_absolute_integral": float(negative[q, a]),
                "projection_balance_residual": float(
                    positive[q, a] - negative[q, a] - region_vectors[q, a] @ units[q]),
            }
    return output


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--engine", choices=("orca", "gaussian"), required=True)
    p.add_argument("--json", type=Path, help="ORCA canonical-MO and TDA JSON")
    p.add_argument("--output", type=Path, required=True, help="ORCA output or Gaussian log")
    p.add_argument("--wfx", type=Path, required=True)
    p.add_argument("--state", type=int, default=1)
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--nto-json", type=Path, help="Optional full NTO MO-mixing matrices")
    p.add_argument("--response-convention", choices=("tda", "xplus_yminus"), default="tda")
    p.add_argument("--min-abs-coeff", type=float, default=0.)
    p.add_argument("--spacing", type=float, default=.2)
    p.add_argument("--padding", type=float, default=8.)
    p.add_argument("--snap", type=float, default=.4)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--bounds", type=float, nargs=6, metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1"))
    group.add_argument("--reference-cube", type=Path)
    p.add_argument("--origin", type=float, nargs=3, default=(0., 0., 0.))
    p.add_argument("--regions", type=Path)
    p.add_argument("--partition", choices=("voronoi", "becke"), default="voronoi")
    p.add_argument("--orbital-evaluation", choices=("canonical", "contracted"), default="contracted")
    p.add_argument("--pilot-points", type=int, help="Integrate only this many prefix points; outputs marked incomplete")
    p.add_argument("--max-points", type=int, default=4096)
    p.add_argument("--primitive-chunk", type=int, default=64)
    p.add_argument("--no-phase-align", action="store_true")
    p.add_argument("--skip-geometry-check", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Validate inputs and report grid/NTO, without integration")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.state < 1:
        raise ValueError("State must be a positive one-based root")
    for target in (args.result, args.nto_json):
        if target is not None and target.exists():
            raise FileExistsError(target)
    if args.min_abs_coeff < 0 or not math.isfinite(args.min_abs_coeff):
        raise ValueError("min-abs-coeff must be finite and nonnegative")
    if args.max_points < 1 or args.primitive_chunk < 1 or (args.pilot_points is not None and args.pilot_points < 1):
        raise ValueError("Block and primitive chunk sizes must be positive")
    if not np.isfinite(args.origin).all():
        raise ValueError("Origin must be finite")
    if not (args.repo / "chiroptmap").is_dir():
        raise FileNotFoundError(f"Cannot find ChiroptMap source in {args.repo}")
    (wfx, occ, virt, ew, mw, e, m, x, occ_full, virt_full, amplitude_meta) = _load_wavefunction(args)
    grid = (cube_header_grid(args.reference_cube, wfx) if args.reference_cube else
            snapped_grid(wfx.coordinates_bohr, args.spacing, args.padding, args.snap, args.bounds))
    labels, membership, region_meta = load_regions(args.regions, wfx.n_nuclei)
    paths = [args.output, args.wfx] + ([args.json] if args.json else [])
    if args.reference_cube:
        paths.append(args.reference_cube)
    source_paths = [Path(__file__)] + [
        args.repo / "chiroptmap" / name for name in (
            "wfx.py", "wfx_cubes.py", "orca_json.py", "orca.py", "gaussian.py", "densities.py")]
    report = {
        "schema": "chiroptmap-stream-analysis-1",
        "status": "dry-run" if args.dry_run else (
            "INCOMPLETE-PILOT" if args.pilot_points is not None and args.pilot_points < grid.size else "complete"),
        "engine": args.engine, "state": args.state, "response_convention": args.response_convention,
        "min_abs_coefficient": args.min_abs_coeff, "arguments": vars(args),
        "inputs": [{"path": str(p.resolve()), "sha256": sha256(p)} for p in paths],
        "implementation": [{"path": str(p.resolve()), "sha256": sha256(p)} for p in source_paths],
        "grid": grid.metadata(), "amplitudes": amplitude_meta,
        "atomic_numbers": wfx.atomic_numbers.tolist(),
        "coordinates_bohr": wfx.coordinates_bohr.tolist(),
        "partition": {
            "type": args.partition,
            "definition_details": (
                "nearest nucleus; exact ties to lowest atom index" if args.partition == "voronoi"
                else "equal-radius Becke polynomial switching, 3 iterations, no heteroatomic radii adjustment"),
            "atom_to_region_one_based": (membership + 1).tolist(), "labels": labels, **region_meta,
        },
        "limitations": [
            "No independent physical-unit calibration is performed; factors match the existing code.",
            "Local moments and cancellation depend on operator origin, field representation and partition.",
            "Equal-radius Becke weights omit heteroatomic radius adjustment.",
            "No transition-current physical normalization or local g are implemented.",
            "No parent reference at shifted origin is inferred from the original printed vector.",
            "Only input TDA X is decomposed as NTOs; Gaussian printed data may omit small amplitudes.",
            "No component cubes are emitted; use the existing generator for selected final fields.",
        ],
        "memory_model": {
            "full_grid_arrays": 0, "max_points_per_block": args.max_points,
            "selected_mos": len(set(occ) | set(virt)), "primitive_chunk": args.primitive_chunk,
            "evaluation": args.orbital_evaluation,
            "evaluated_combinations": (
                len(occ) * (2 if np.allclose(ew, 2 * mw, rtol=5e-15, atol=0) else 3)
                if args.orbital_evaluation == "contracted" else len(set(occ) | set(virt))),
            "note": "Base estimate excludes primitive evaluation temporaries, BLAS buffers, MO coefficients and JSON parsing.",
        },
    }
    report["memory_model"]["mo_value_and_gradient_base_bytes_per_block"] = (
        args.max_points * report["memory_model"]["evaluated_combinations"] * 4 * 8)
    if args.response_convention == "tda":
        summary, holes, particles = nto_decomposition(x)
        report["nto"] = summary
        if args.nto_json:
            nto = {
                "schema": "chiroptmap-tda-nto-1", **summary,
                "occupied_mo_numbers": list(occ_full), "virtual_mo_numbers": list(virt_full),
                "hole_mixing_columns": holes.tolist(), "particle_mixing_columns": particles.tolist(),
            }
            args.nto_json.parent.mkdir(parents=True, exist_ok=True)
            args.nto_json.write_text(json.dumps(nto, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    else:
        report["nto"] = {"status": "not performed; NTO analysis here is restricted to TDA"}
    if not args.dry_run:
        report["analysis"] = integrate(
            wfx, occ, virt, ew, mw, e, m, grid, np.asarray(args.origin), labels, membership,
            args.max_points, args.primitive_chunk, not args.no_phase_align,
            partition=args.partition, evaluation=args.orbital_evaluation, pilot_points=args.pilot_points)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False,
                   default=lambda obj: str(obj) if isinstance(obj, Path) else obj),
        encoding="utf-8")
    print(f"[result] {args.result.resolve()}", file=sys.stderr)
    return report


if __name__ == "__main__":
    main()
