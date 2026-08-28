"""Direct TEDM/TMDM cube generation from Gaussian output and canonical WFX."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .cubeio import (
    coordinate_1d_aligned_bohr,
    cube_step_sizes_bohr,
    cube_voxel_volume_bohr3,
    read_cube,
    write_cube,
)
from .densities import excitation_weights
from .gaussian import (
    build_cpl_rows,
    diagnose_coefficients,
    parse_edtm_vector,
    parse_excited_state_block,
    parse_excited_state_amplitudes,
    parse_mdtm_vector,
    write_cpl_csv,
)
from .utils import eprint
from .validation import compare_component_integrals, write_validation_csv
from .wfx import (
    WFXWavefunction,
    build_cube_grid,
    evaluate_mos,
    parse_last_standard_orientation,
    read_wfx,
    require_restricted_canonical_mos,
)


def _weight_matrices(
    forward: Dict[Tuple[int, int], float],
    reverse: Dict[Tuple[int, int], float],
    pairs: Sequence[Tuple[int, int]],
    occupied_mos: Sequence[int],
    virtual_mos: Sequence[int],
    dtype: np.dtype,
    density_mode: str,
    response_convention: str,
    validation_tedm_factor: float,
    validation_tmdm_factor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    occupied_index = {mo: i for i, mo in enumerate(occupied_mos)}
    virtual_index = {mo: i for i, mo in enumerate(virtual_mos)}
    tedm = np.zeros((len(occupied_mos), len(virtual_mos)), dtype=dtype)
    tmdm = np.zeros_like(tedm)
    for pair in pairs:
        x_amplitude = forward.get(pair, 0.0)
        y_amplitude = reverse.get(pair, 0.0)
        if response_convention == "forward-only":
            electric_amplitude = x_amplitude
            magnetic_amplitude = x_amplitude
        elif response_convention == "xplus_yminus":
            electric_amplitude = x_amplitude + y_amplitude
            magnetic_amplitude = x_amplitude - y_amplitude
        else:
            raise ValueError(f"Unknown response convention: {response_convention}")
        electric_weight, _ = excitation_weights(
            electric_amplitude,
            density_mode,
            validation_edtm_factor=validation_tedm_factor,
            validation_mdtm_factor=validation_tmdm_factor,
        )
        _, magnetic_weight = excitation_weights(
            magnetic_amplitude,
            density_mode,
            validation_edtm_factor=validation_tedm_factor,
            validation_mdtm_factor=validation_tmdm_factor,
        )
        tedm[occupied_index[pair[0]], virtual_index[pair[1]]] += electric_weight
        tmdm[occupied_index[pair[0]], virtual_index[pair[1]]] += magnetic_weight
    return tedm, tmdm


def _geometry_check(wfx: WFXWavefunction, log_text: str) -> None:
    orientation = parse_last_standard_orientation(log_text)
    if orientation is None:
        eprint("[Geometry warning] Gaussian Standard orientation was not found; coordinate matching was skipped.")
        return
    atomic_numbers, coordinates_bohr = orientation
    if atomic_numbers.shape != wfx.atomic_numbers.shape or not np.array_equal(atomic_numbers, wfx.atomic_numbers):
        raise ValueError("The Gaussian log and canonical WFX have different atom lists.")
    max_difference = float(np.max(np.abs(coordinates_bohr - wfx.coordinates_bohr)))
    if max_difference > 2.0e-5:
        raise ValueError(
            "The Gaussian log and canonical WFX are not in the same Standard orientation "
            f"(maximum coordinate difference {max_difference:.3e} bohr)."
        )
    eprint(f"[Geometry] Log/WFX coordinate match: max |delta|={max_difference:.3e} bohr")


def _reference_cube_geometry_check(wfx: WFXWavefunction, cube) -> None:
    if cube.natoms != wfx.n_nuclei:
        raise ValueError("The reference cube and WFX contain different atom counts.")
    atomic_numbers = []
    coordinates = []
    for line in cube.atoms_block:
        tokens = line.split()
        if len(tokens) < 5:
            raise ValueError("Malformed atom line in reference cube.")
        atomic_numbers.append(int(float(tokens[0])))
        coordinates.append([float(tokens[2]), float(tokens[3]), float(tokens[4])])
    if not np.array_equal(np.asarray(atomic_numbers, dtype=np.int64), wfx.atomic_numbers):
        raise ValueError("The reference cube and WFX contain different atom lists.")
    maximum = float(np.max(np.abs(np.asarray(coordinates) - wfx.coordinates_bohr)))
    if maximum > 2.0e-5:
        raise ValueError(
            "The reference cube and WFX coordinates differ "
            f"(maximum coordinate difference {maximum:.3e} bohr)."
        )
    eprint(f"[Geometry] Reference-cube/WFX coordinate match: max |delta|={maximum:.3e} bohr")


def generate_cubes(args: argparse.Namespace) -> Tuple[Path, Path]:
    log_text = args.log.read_text(encoding="utf-8", errors="ignore")
    wfx = read_wfx(args.wfx)
    all_excitations = parse_excited_state_block(log_text, args.state)
    all_forward, all_reverse = parse_excited_state_amplitudes(log_text, args.state)
    diagnostics = diagnose_coefficients(log_text, args.state, all_excitations)
    if args.strict_ci and (
        not diagnostics.has_recommended_iop or diagnostics.two_sum_c_squared < 0.98
    ):
        raise ValueError(
            "The printed excitation coefficients failed the strict completeness check. "
            "Repeat Gaussian with IOp(9/40=4). In the validated molecule1 scan, "
            "IOp(9/40=5) printed exactly the same amplitudes as n=4."
        )

    if args.min_abs_coeff < 0.0:
        raise ValueError("--min-abs-coeff must be non-negative.")
    if args.response_convention != "forward-only" and args.density_mode != "validation":
        raise ValueError("X/Y response conventions are supported only with --density-mode validation.")
    candidate_pairs = set(all_forward)
    if args.response_convention != "forward-only":
        candidate_pairs.update(all_reverse)
    pairs = sorted(
        pair
        for pair in candidate_pairs
        if max(abs(all_forward.get(pair, 0.0)), abs(all_reverse.get(pair, 0.0)))
        >= args.min_abs_coeff
    )
    if not pairs:
        raise ValueError(
            f"No configurations remain after applying --min-abs-coeff={args.min_abs_coeff:g}."
        )
    forward = {pair: all_forward[pair] for pair in pairs if pair in all_forward}
    reverse = {pair: all_reverse[pair] for pair in pairs if pair in all_reverse}
    retained_two_sum_c_squared = 2.0 * sum(value * value for value in forward.values())

    needed_mos = sorted({mo for pair in pairs for mo in pair})
    require_restricted_canonical_mos(wfx, needed_mos)
    if not args.skip_geometry_check:
        _geometry_check(wfx, log_text)

    dtype = np.dtype(args.dtype)
    if args.reference_cube is not None:
        cube = read_cube(args.reference_cube, dtype=dtype, cube_units="bohr")
        _reference_cube_geometry_check(wfx, cube)
        grid_source = str(args.reference_cube.resolve())
    else:
        cube = build_cube_grid(
            wfx,
            padding_bohr=args.padding,
            spacing_bohr=args.spacing,
            dtype=dtype,
        )
        grid_source = "direct_wfx_bounding_box"
    x, y, z = coordinate_1d_aligned_bohr(cube)
    grid_shape = (cube.nx, cube.ny, cube.nz)
    dx, dy, dz = cube_step_sizes_bohr(cube)

    occupied_mos = sorted({pair[0] for pair in pairs})
    virtual_mos = sorted({pair[1] for pair in pairs})
    all_mos = sorted(set(occupied_mos) | set(virtual_mos))
    all_index = {mo: i for i, mo in enumerate(all_mos)}
    occupied_columns = np.asarray([all_index[mo] for mo in occupied_mos], dtype=np.int64)
    virtual_columns = np.asarray([all_index[mo] for mo in virtual_mos], dtype=np.int64)
    tedm_weights, tmdm_weights = _weight_matrices(
        forward,
        reverse,
        pairs,
        occupied_mos,
        virtual_mos,
        dtype,
        args.density_mode,
        args.response_convention,
        args.validation_tedm_factor,
        args.validation_tmdm_factor,
    )

    tedm_components = tuple(np.zeros(grid_shape, dtype=dtype) for _ in range(3))
    tmdm_components = tuple(np.zeros(grid_shape, dtype=dtype) for _ in range(3))

    eprint(
        f"[Grid] shape={grid_shape} steps=({dx:g},{dy:g},{dz:g}) bohr source={grid_source}"
    )
    eprint(
        f"[MOs] occupied={len(occupied_mos)} virtual={len(virtual_mos)} "
        f"total={len(all_mos)} primitives={wfx.n_primitives}"
    )
    eprint(
        f"[CI] state={args.state} min_abs_coeff={args.min_abs_coeff:g} "
        f"pairs={len(pairs)} forward={len(forward)}/{len(all_forward)} "
        f"reverse={len(reverse)}/{len(all_reverse)} response={args.response_convention} "
        f"2*sum(c^2)={retained_two_sum_c_squared:.8f}/{diagnostics.two_sum_c_squared:.8f} "
        f"IOp(9/40)={diagnostics.iop_940_level}"
    )

    for block_number, x0 in enumerate(range(0, cube.nx, args.x_block), 1):
        x1 = min(cube.nx, x0 + args.x_block)
        xx, yy, zz = np.meshgrid(x[x0:x1], y, z, indexing="ij")
        points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel())).astype(dtype, copy=False)
        phi, gradients = evaluate_mos(
            wfx,
            all_mos,
            points,
            dtype=dtype,
            primitive_chunk=args.primitive_chunk,
        )
        gx, gy, gz = gradients
        phi_occ = phi[:, occupied_columns]
        phi_virt = phi[:, virtual_columns]
        grad_occ = (gx[:, occupied_columns], gy[:, occupied_columns], gz[:, occupied_columns])
        grad_virt = (gx[:, virtual_columns], gy[:, virtual_columns], gz[:, virtual_columns])

        tau = np.einsum(
            "ij,ij->i",
            phi_occ @ tedm_weights,
            phi_virt,
            optimize=True,
        )
        block_shape = (x1 - x0, cube.ny, cube.nz)
        for component, coordinate in enumerate((points[:, 0], points[:, 1], points[:, 2])):
            tedm_components[component][x0:x1] = (coordinate * tau).reshape(block_shape)

        coordinate_columns = (points[:, 0, None], points[:, 1, None], points[:, 2, None])
        a_occ = (
            coordinate_columns[1] * grad_occ[2] - coordinate_columns[2] * grad_occ[1],
            coordinate_columns[2] * grad_occ[0] - coordinate_columns[0] * grad_occ[2],
            coordinate_columns[0] * grad_occ[1] - coordinate_columns[1] * grad_occ[0],
        )
        a_virt = (
            coordinate_columns[1] * grad_virt[2] - coordinate_columns[2] * grad_virt[1],
            coordinate_columns[2] * grad_virt[0] - coordinate_columns[0] * grad_virt[2],
            coordinate_columns[0] * grad_virt[1] - coordinate_columns[1] * grad_virt[0],
        )
        phi_occ_weighted = phi_occ @ tmdm_weights
        for component in range(3):
            term1 = np.einsum(
                "ij,ij->i", a_occ[component] @ tmdm_weights, phi_virt, optimize=True
            )
            term2 = np.einsum(
                "ij,ij->i", phi_occ_weighted, a_virt[component], optimize=True
            )
            tmdm_components[component][x0:x1] = (term1 - term2).reshape(block_shape)

        eprint(f"[Grid] completed x slices {x1}/{cube.nx}")

    gaussian_tedm = parse_edtm_vector(log_text, args.state)
    gaussian_tmdm = parse_mdtm_vector(log_text, args.state)
    tedm_norm = math.sqrt(sum(value * value for value in gaussian_tedm))
    tmdm_norm = math.sqrt(sum(value * value for value in gaussian_tmdm))
    if tedm_norm == 0.0 or tmdm_norm == 0.0:
        raise ValueError("The selected state has a zero Gaussian TEDM or TMDM vector norm.")

    if args.phase_align == "tedm":
        volume = cube_voxel_volume_bohr3(cube)
        integrated = np.asarray(
            [float(np.sum(grid, dtype=np.float64) * volume) for grid in tedm_components]
        )
        if float(np.dot(integrated, np.asarray(gaussian_tedm))) < 0.0:
            for grid in tedm_components + tmdm_components:
                grid *= -1.0
            eprint("[Phase] Multiplied all component densities by -1 to align TEDM with Gaussian.")

    prefix = args.outprefix or f"{args.log.stem}_S{args.state}_{args.density_mode.replace('-', '_')}"
    args.outdir.mkdir(parents=True, exist_ok=True)

    if not args.no_cpl_csv:
        cpl_path = args.cpl_csv_path or (args.outdir / f"{args.log.stem}-CPL.csv")
        write_cpl_csv(cpl_path, args.log, build_cpl_rows(log_text))
        eprint(f"[CPL] {cpl_path}")

    if not args.skip_validation_report:
        rows = compare_component_integrals(cube, "TEDM", tedm_components, gaussian_tedm)
        rows += compare_component_integrals(cube, "TMDM", tmdm_components, gaussian_tmdm)
        validation_path = args.outdir / f"{prefix}_moment_validation.csv"
        write_validation_csv(validation_path, rows)
        eprint(f"[Validation] {validation_path}")

    tedm_unit = np.asarray(gaussian_tedm, dtype=float) / tedm_norm
    tmdm_unit = np.asarray(gaussian_tmdm, dtype=float) / tmdm_norm
    tedm_total = sum(tedm_unit[index] * tedm_components[index] for index in range(3))
    tmdm_total = sum(tmdm_unit[index] * tmdm_components[index] for index in range(3))

    tedm_path = args.outdir / f"{prefix}_TEDM_total.cube"
    tmdm_path = args.outdir / f"{prefix}_TMDM_total.cube"
    write_cube(
        tedm_path,
        cube,
        tedm_total,
        f"TEDM density from {args.log.name}; state={args.state}; mode={args.density_mode}",
        f"Direct canonical-WFX primitive evaluation; steps={dx:g},{dy:g},{dz:g} bohr",
    )
    write_cube(
        tmdm_path,
        cube,
        tmdm_total,
        f"TMDM density from {args.log.name}; state={args.state}; mode={args.density_mode}",
        f"Direct canonical-WFX primitive evaluation; steps={dx:g},{dy:g},{dz:g} bohr",
    )

    if args.keep_components:
        for label, grids in (("TEDM", tedm_components), ("TMDM", tmdm_components)):
            for component, grid in zip(("x", "y", "z"), grids):
                write_cube(
                    args.outdir / f"{prefix}_{label}_{component}.cube",
                    cube,
                    grid,
                    f"{label} component density from {args.log.name}; state={args.state}",
                    f"component={component}; mode={args.density_mode}",
                )

    settings_path = args.outdir / f"{prefix}_settings.txt"
    settings_path.write_text(
        "\n".join(
            [
                f"log={args.log.resolve()}",
                f"canonical_wfx={args.wfx.resolve()}",
                f"state={args.state}",
                f"density_mode={args.density_mode}",
                f"response_convention={args.response_convention}",
                f"spacing_bohr={args.spacing}",
                f"padding_bohr={args.padding}",
                f"grid={cube.nx},{cube.ny},{cube.nz}",
                f"grid_steps_bohr={dx},{dy},{dz}",
                f"grid_source={grid_source}",
                f"min_abs_coeff={args.min_abs_coeff}",
                f"retained_pairs={len(pairs)}",
                f"retained_forward={len(forward)}",
                f"all_printed_forward={len(all_forward)}",
                f"retained_reverse={len(reverse)}",
                f"all_printed_reverse={len(all_reverse)}",
                f"retained_2sumc2={retained_two_sum_c_squared}",
                f"all_printed_2sumc2={diagnostics.two_sum_c_squared}",
                f"validation_tedm_factor={args.validation_tedm_factor}",
                f"validation_tmdm_factor={args.validation_tmdm_factor}",
                "mo_source=restricted_canonical_scf_wfx_with_virtuals",
                "checkpoint_dependency=none",
                "mo_cube_dependency=none",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    eprint(f"[Done] TEDM: {tedm_path}")
    eprint(f"[Done] TMDM: {tmdm_path}")
    return tedm_path, tmdm_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate ChiroptMap-compatible TEDM/TMDM cubes directly from a Gaussian TD/TDA log "
            "and a matching canonical ground-state SCF WFX containing virtual orbitals."
        )
    )
    parser.add_argument("--log", required=True, type=Path, help="Gaussian TD-DFT/TDA output")
    parser.add_argument("--wfx", required=True, type=Path, help="Matching canonical SCF WFX")
    parser.add_argument("--state", required=True, type=int, help="Excited-state number")
    parser.add_argument("--outdir", type=Path, default=Path("wfx_cubes"))
    parser.add_argument("--outprefix", default=None)
    parser.add_argument(
        "--reference-cube",
        type=Path,
        default=None,
        help="Use this Gaussian cube's exact grid for a controlled fchk/WFX comparison.",
    )
    parser.add_argument(
        "--density-mode",
        choices=["contribution-map", "validation"],
        default="contribution-map",
    )
    parser.add_argument(
        "--response-convention",
        choices=["forward-only", "xplus_yminus"],
        default="forward-only",
        help="Use Gaussian X only, or validation convention TEDM=X+Y and TMDM=X-Y.",
    )
    parser.add_argument("--spacing", type=float, default=0.20, help="Grid spacing in Bohr")
    parser.add_argument("--padding", type=float, default=6.0, help="Molecular padding in Bohr")
    parser.add_argument("--x-block", type=int, default=2, help="Number of x slices per block")
    parser.add_argument("--primitive-chunk", type=int, default=64)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument(
        "--min-abs-coeff",
        type=float,
        default=1.0e-4,
        help=(
            "Discard configurations with |coefficient| below this value. "
            "Default: 1e-4, matching the intended Gaussian IOp(9/40=4) resolution."
        ),
    )
    parser.add_argument(
        "--cpl-csv-path",
        type=Path,
        default=None,
        help="CPL summary CSV path. Default: <outdir>/<log-stem>-CPL.csv.",
    )
    parser.add_argument(
        "--no-cpl-csv",
        action="store_true",
        help="Do not write the state-resolved CPL summary and g_CPL CSV.",
    )
    parser.add_argument("--keep-components", action="store_true")
    parser.add_argument("--phase-align", choices=["none", "tedm"], default="none")
    parser.add_argument("--strict-ci", action="store_true")
    parser.add_argument("--skip-validation-report", action="store_true")
    parser.add_argument("--skip-geometry-check", action="store_true")
    parser.add_argument("--validation-tedm-factor", type=float, default=-2.0)
    parser.add_argument("--validation-tmdm-factor", type=float, default=-1.0)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.log.is_file():
        raise FileNotFoundError(args.log)
    if not args.wfx.is_file():
        raise FileNotFoundError(args.wfx)
    if args.reference_cube is not None and not args.reference_cube.is_file():
        raise FileNotFoundError(args.reference_cube)
    if args.x_block < 1:
        raise ValueError("--x-block must be at least 1.")
    generate_cubes(args)


if __name__ == "__main__":
    main()
