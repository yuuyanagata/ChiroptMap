"""Direct EDTM/MDTM cube generation from ORCA JSON plus occupied WFX."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .cubeio import (
    coordinate_1d_aligned_bohr,
    cube_step_sizes_bohr,
    cube_voxel_volume_bohr3,
    read_cube,
    write_cube,
)
from .gaussian import write_cpl_csv
from .orca import (
    build_cpl_rows,
    parse_absorption_spectrum_all_states,
    parse_cd_spectrum_all_states,
)
from .orca_json import OrcaTdaRoot, read_orca_json_wavefunction
from .utils import eprint
from .validation import compare_component_integrals, write_validation_csv
from .wfx import build_cube_grid, evaluate_mos, require_restricted_canonical_mos


def _selected_weight_matrix(
    root: OrcaTdaRoot,
    threshold: float,
    dtype: np.dtype,
) -> Tuple[Tuple[int, ...], Tuple[int, ...], np.ndarray, int, float]:
    mask = np.abs(root.x_amplitudes) >= threshold
    occupied_rows = np.flatnonzero(np.any(mask, axis=1))
    virtual_columns = np.flatnonzero(np.any(mask, axis=0))
    if occupied_rows.size == 0 or virtual_columns.size == 0:
        raise ValueError(f"No ORCA configurations remain at threshold {threshold:g}.")
    selected = root.x_amplitudes[np.ix_(occupied_rows, virtual_columns)].astype(dtype, copy=True)
    selected_mask = mask[np.ix_(occupied_rows, virtual_columns)]
    selected[~selected_mask] = 0.0
    occupied_mos = tuple(root.occupied_mos[index] for index in occupied_rows)
    virtual_mos = tuple(root.virtual_mos[index] for index in virtual_columns)
    pair_count = int(np.count_nonzero(selected_mask))
    retained_norm = float(np.sum(selected * selected, dtype=np.float64))
    return occupied_mos, virtual_mos, selected, pair_count, retained_norm


def _reference_cube_geometry_check(wfx, cube) -> None:
    if cube.natoms != wfx.n_nuclei:
        raise ValueError("The reference cube and ORCA wavefunction have different atom counts.")
    atomic_numbers = []
    coordinates = []
    for line in cube.atoms_block:
        tokens = line.split()
        atomic_numbers.append(int(float(tokens[0])))
        coordinates.append([float(tokens[2]), float(tokens[3]), float(tokens[4])])
    if not np.array_equal(np.asarray(atomic_numbers, dtype=np.int64), wfx.atomic_numbers):
        raise ValueError("The reference cube and ORCA wavefunction have different atom lists.")
    maximum = float(np.max(np.abs(np.asarray(coordinates) - wfx.coordinates_bohr)))
    if maximum > 2.0e-5:
        raise ValueError(
            f"The reference cube and ORCA wavefunction coordinates differ by {maximum:.3e} bohr."
        )
    eprint(f"[Geometry] Reference-cube/ORCA coordinate match: max |delta|={maximum:.3e} bohr")


def generate_orca_json_cubes(args: argparse.Namespace) -> Tuple[Path, Path]:
    bridge = read_orca_json_wavefunction(args.json, args.wfx)
    roots = {root.state: root for root in bridge.roots}
    if args.state not in roots:
        raise ValueError(f"ORCA JSON does not contain singlet root {args.state}.")
    root = roots[args.state]
    dtype = np.dtype(args.dtype)
    occupied_mos, virtual_mos, weights, pair_count, retained_norm = _selected_weight_matrix(
        root, args.min_abs_coeff, dtype
    )
    all_mos = sorted(set(occupied_mos) | set(virtual_mos))
    require_restricted_canonical_mos(bridge.wfx, all_mos)

    if args.reference_cube is not None:
        cube = read_cube(args.reference_cube, dtype=dtype, cube_units="bohr")
        _reference_cube_geometry_check(bridge.wfx, cube)
        grid_source = str(args.reference_cube.resolve())
    else:
        cube = build_cube_grid(
            bridge.wfx,
            padding_bohr=args.padding,
            spacing_bohr=args.spacing,
            dtype=dtype,
        )
        grid_source = "direct_orca_json_bounding_box"

    output_text = args.output.read_text(encoding="utf-8", errors="ignore")
    electric_rows = parse_absorption_spectrum_all_states(output_text)
    magnetic_rows = parse_cd_spectrum_all_states(output_text)
    if args.state not in electric_rows or args.state not in magnetic_rows:
        raise ValueError(f"ORCA output does not contain transition moments for state {args.state}.")
    _energy_ev, _wavelength, _oscillator, ex, ey, ez = electric_rows[args.state]
    _rotatory_strength, mx, my, mz = magnetic_rows[args.state]
    orca_edtm = (ex, ey, ez)
    # ORCA's length-gauge CD table uses R = mu dot (2*m) for the tabulated
    # magnetic components.  ChiroptMap's Gaussian-compatible MDTM convention is
    # therefore compared against twice the printed ORCA vector.
    orca_mdtm = (2.0 * mx, 2.0 * my, 2.0 * mz)

    all_index = {mo: index for index, mo in enumerate(all_mos)}
    occupied_columns = np.asarray([all_index[mo] for mo in occupied_mos], dtype=np.int64)
    virtual_columns = np.asarray([all_index[mo] for mo in virtual_mos], dtype=np.int64)
    electric_weights = np.asarray(-math.sqrt(2.0) * weights, dtype=dtype, order="C")
    magnetic_weights = np.asarray(-weights / math.sqrt(2.0), dtype=dtype, order="C")

    x, y, z = coordinate_1d_aligned_bohr(cube)
    grid_shape = (cube.nx, cube.ny, cube.nz)
    dx, dy, dz = cube_step_sizes_bohr(cube)
    edtm_components = tuple(np.zeros(grid_shape, dtype=dtype) for _ in range(3))
    mdtm_components = tuple(np.zeros(grid_shape, dtype=dtype) for _ in range(3))

    eprint(
        f"[ORCA bridge] AOs/MOs={bridge.diagnostics.n_mos} "
        f"occupied={bridge.diagnostics.n_occupied} primitives={bridge.diagnostics.n_primitives} "
        f"occupied reconstruction max error="
        f"{bridge.diagnostics.maximum_occupied_reconstruction_error:.3e}"
    )
    eprint(
        f"[CI] state={args.state} threshold={args.min_abs_coeff:g} pairs={pair_count} "
        f"occupied={len(occupied_mos)} virtual={len(virtual_mos)} "
        f"sum_x2={retained_norm:.12f}/{float(np.sum(root.x_amplitudes ** 2)):.12f}"
    )
    eprint(f"[Grid] shape={grid_shape} steps=({dx:g},{dy:g},{dz:g}) bohr source={grid_source}")

    for x0 in range(0, cube.nx, args.x_block):
        x1 = min(cube.nx, x0 + args.x_block)
        xx, yy, zz = np.meshgrid(x[x0:x1], y, z, indexing="ij")
        points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel())).astype(dtype, copy=False)
        phi, gradients = evaluate_mos(
            bridge.wfx,
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

        tau = np.einsum("ij,ij->i", phi_occ @ electric_weights, phi_virt, optimize=True)
        block_shape = (x1 - x0, cube.ny, cube.nz)
        for component, coordinate in enumerate((points[:, 0], points[:, 1], points[:, 2])):
            edtm_components[component][x0:x1] = (coordinate * tau).reshape(block_shape)

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
        phi_occ_weighted = phi_occ @ magnetic_weights
        for component in range(3):
            term1 = np.einsum(
                "ij,ij->i", a_occ[component] @ magnetic_weights, phi_virt, optimize=True
            )
            term2 = np.einsum(
                "ij,ij->i", phi_occ_weighted, a_virt[component], optimize=True
            )
            mdtm_components[component][x0:x1] = (term1 - term2).reshape(block_shape)
        eprint(f"[Grid] completed x slices {x1}/{cube.nx}")

    volume = cube_voxel_volume_bohr3(cube)
    integrated_edtm = np.asarray(
        [float(np.sum(grid, dtype=np.float64) * volume) for grid in edtm_components]
    )
    if float(np.dot(integrated_edtm, np.asarray(orca_edtm))) < 0.0:
        for grid in edtm_components + mdtm_components:
            grid *= -1.0
        eprint("[Phase] Multiplied all densities by -1 to align with the ORCA EDTM.")

    args.outdir.mkdir(parents=True, exist_ok=True)
    prefix = args.outprefix or f"{args.json.stem}_S{args.state}"
    if not args.no_cpl_csv:
        cpl_path = args.cpl_csv_path or (args.outdir / f"{args.output.stem}-CPL.csv")
        write_cpl_csv(cpl_path, args.output, build_cpl_rows(output_text))
        eprint(f"[CPL] {cpl_path}")
    validation_rows = compare_component_integrals(cube, "EDTM", edtm_components, orca_edtm)
    validation_rows += compare_component_integrals(cube, "MDTM", mdtm_components, orca_mdtm)
    validation_path = args.outdir / f"{prefix}_moment_validation.csv"
    write_validation_csv(validation_path, validation_rows)

    edtm_norm = float(np.linalg.norm(np.asarray(orca_edtm)))
    mdtm_norm = float(np.linalg.norm(np.asarray(orca_mdtm)))
    if edtm_norm == 0.0 or mdtm_norm == 0.0:
        raise ValueError("The selected ORCA state has a zero EDTM or MDTM norm.")
    edtm_unit = np.asarray(orca_edtm) / edtm_norm
    mdtm_unit = np.asarray(orca_mdtm) / mdtm_norm
    edtm_total = sum(edtm_unit[index] * edtm_components[index] for index in range(3))
    mdtm_total = sum(mdtm_unit[index] * mdtm_components[index] for index in range(3))

    edtm_path = args.outdir / f"{prefix}_EDTM_total.cube"
    mdtm_path = args.outdir / f"{prefix}_MDTM_total.cube"
    write_cube(
        edtm_path,
        cube,
        edtm_total,
        f"ORCA EDTM density from {args.json.name}; state={args.state}",
        f"Direct JSON/WFX primitive evaluation; threshold={args.min_abs_coeff:g}",
    )
    write_cube(
        mdtm_path,
        cube,
        mdtm_total,
        f"ORCA MDTM density from {args.json.name}; state={args.state}",
        f"Direct JSON/WFX primitive evaluation; threshold={args.min_abs_coeff:g}",
    )
    if args.keep_components:
        for label, grids in (("EDTM", edtm_components), ("MDTM", mdtm_components)):
            for component, grid in zip(("x", "y", "z"), grids):
                write_cube(
                    args.outdir / f"{prefix}_{label}_{component}.cube",
                    cube,
                    grid,
                    f"ORCA {label} component from {args.json.name}; state={args.state}",
                    f"component={component}; threshold={args.min_abs_coeff:g}",
                )

    grid_origin = ",".join(f"{float(value):.16g}" for value in cube.origin_bohr)
    settings_path = args.outdir / f"{prefix}_settings.txt"
    settings_path.write_text(
        "\n".join(
            [
                f"json={args.json.resolve()}",
                f"occupied_wfx={args.wfx.resolve()}",
                f"orca_output={args.output.resolve()}",
                f"state={args.state}",
                f"min_abs_coeff={args.min_abs_coeff}",
                f"retained_pairs={pair_count}",
                f"retained_sum_x2={retained_norm}",
                f"grid={cube.nx},{cube.ny},{cube.nz}",
                f"grid_origin_bohr={grid_origin}",
                f"grid_steps_bohr={dx},{dy},{dz}",
                f"grid_source={grid_source}",
                "magnetic_operator_origin_bohr=0,0,0",
                "coordinate_frame=wavefunction_cartesian_coordinates",
                f"occupied_reconstruction_max_error={bridge.diagnostics.maximum_occupied_reconstruction_error}",
                "electric_factor=-sqrt(2)",
                "magnetic_factor=-1/sqrt(2)",
                "orca_printed_magnetic_reference_factor=2",
                "checkpoint_dependency=none",
                "mo_cube_dependency=none",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    eprint(f"[Validation] {validation_path}")
    eprint(f"[Done] EDTM: {edtm_path}")
    eprint(f"[Done] MDTM: {mdtm_path}")
    return edtm_path, mdtm_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate EDTM/MDTM cubes directly from ORCA JSON and occupied WFX."
    )
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--wfx", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--state", type=int, default=1)
    parser.add_argument("--reference-cube", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=Path("orca_json_cubes"))
    parser.add_argument("--outprefix", default=None)
    parser.add_argument("--min-abs-coeff", type=float, default=1.0e-4)
    parser.add_argument(
        "--cpl-csv-path",
        type=Path,
        default=None,
        help="CPL summary CSV path. Default: <outdir>/<output-stem>-CPL.csv.",
    )
    parser.add_argument(
        "--no-cpl-csv",
        action="store_true",
        help="Do not write the state-resolved CPL summary and g_CPL CSV.",
    )
    parser.add_argument("--spacing", type=float, default=0.20)
    parser.add_argument("--padding", type=float, default=6.0)
    parser.add_argument("--x-block", type=int, default=2)
    parser.add_argument("--primitive-chunk", type=int, default=256)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--keep-components", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.min_abs_coeff < 0.0:
        raise ValueError("--min-abs-coeff must be non-negative.")
    for path in (args.json, args.wfx, args.output):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.reference_cube is not None and not args.reference_cube.is_file():
        raise FileNotFoundError(args.reference_cube)
    generate_orca_json_cubes(args)
