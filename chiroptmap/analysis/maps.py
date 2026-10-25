#!/usr/bin/env python3
"""Stream selected linear EDTM/MDTM and separate TDA NTO orbital cubes.

The moment maps use ALL amplitudes surviving the explicitly reported source
threshold, never the selected subset of NTO pairs. NTO orbitals are auxiliary
orbital pictures and must not be labelled transition-moment-density maps.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from . import stream


class CubeStream:
    """One scalar, positive-count, Bohr cube written in x/y/z-fastest order."""

    def __init__(self, path: Path, grid: stream.Grid, wfx, title: str, subtitle: str):
        self.final_path = path
        self.partial_path = path.with_name(path.name + ".partial")
        self.expected = grid.size
        self.written = 0
        self.pending = np.empty(0)
        self.handle = self.partial_path.open("w", encoding="ascii", newline="\n")
        self.handle.write(title.replace("\n", " ") + "\n" + subtitle.replace("\n", " ") + "\n")
        self.handle.write(f"{wfx.n_nuclei:5d} " + " ".join(f"{v: .12f}" for v in grid.lower) + "\n")
        for axis in range(3):
            vector = np.zeros(3)
            vector[axis] = grid.steps[axis]
            self.handle.write(f"{grid.counts[axis]:5d} "
                              + " ".join(f"{v: .12f}" for v in vector) + "\n")
        for z, charge, coordinate in zip(wfx.atomic_numbers, wfx.nuclear_charges, wfx.coordinates_bohr):
            self.handle.write(f"{int(z):5d} {float(charge): .8f} "
                              + " ".join(f"{v: .12f}" for v in coordinate) + "\n")

    def write(self, values):
        values = np.asarray(values, float).ravel()
        if not np.isfinite(values).all() or self.written + len(values) > self.expected:
            raise ValueError("Non-finite cube values or too many grid points")
        self.written += len(values)
        values = np.concatenate((self.pending, values))
        boundary = len(values) // 6 * 6
        self.handle.write("".join(
            " ".join(f"{v: .10E}" for v in values[i:i + 6]) + "\n"
            for i in range(0, boundary, 6)))
        self.pending = values[boundary:].copy()

    def close(self, success=True):
        if self.handle.closed:
            return
        if success and self.written != self.expected:
            self.handle.close()
            raise ValueError(f"Incomplete cube: {self.written}/{self.expected}: {self.final_path}")
        if success and len(self.pending):
            self.handle.write(" ".join(f"{v: .10E}" for v in self.pending) + "\n")
        self.handle.close()
        if success:
            self.partial_path.replace(self.final_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close(success=exc_type is None)


def selected_ntos(wfx, occupied_numbers, virtual_numbers, x, indices):
    """Normalized MO mixing without singular-value scaling of orbital amplitudes."""
    summary, holes, particles = stream.nto_decomposition(x)
    selected = np.asarray(indices, dtype=int) - 1
    if np.any(selected < 0) or np.any(selected >= holes.shape[1]) or len(set(indices)) != len(indices):
        raise ValueError("NTO pair indices must be distinct, positive and within the SVD rank dimension")
    mo_index = {number: i for i, number in enumerate(wfx.mo_numbers)}
    c_occ = wfx.mo_coefficients[[mo_index[n] for n in occupied_numbers]]
    c_virtual = wfx.mo_coefficients[[mo_index[n] for n in virtual_numbers]]
    selected_holes, selected_particles = holes[:, selected], particles[:, selected]
    coeff = np.vstack((selected_holes.T @ c_occ, selected_particles.T @ c_virtual))
    count = len(coeff)
    nto_wfx = replace(
        wfx, title=wfx.title + " [selected TDA NTO orbitals]",
        mo_numbers=tuple(range(1, count + 1)), mo_coefficients=coeff,
        occupations=np.zeros(count), energies=np.zeros(count),
        spin_types=tuple("Alpha and Beta" for _ in range(count)))
    records = [{
        "pair": i, "singular_value": summary["singular_values"][i - 1],
        "normalized_weight": summary["normalized_weights"][i - 1],
        "hole_mo_mixing": holes[:, i - 1].tolist(),
        "particle_mo_mixing": particles[:, i - 1].tolist(),
    } for i in indices]
    metadata = {
        "full_X_reconstruction_relative_error": summary["full_reconstruction_frobenius_relative_error"],
        "full_X_sum_squared": summary["sum_x2"],
        "selected_pairs": records,
        "selected_normalized_weight_sum": sum(r["normalized_weight"] for r in records),
        "occupied_mo_numbers": list(occupied_numbers), "virtual_mo_numbers": list(virtual_numbers),
        "hole_MO_mixing_orthonormality_max_error": float(
            np.max(np.abs(selected_holes.T @ selected_holes - np.eye(len(indices))))),
        "particle_MO_mixing_orthonormality_max_error": float(
            np.max(np.abs(selected_particles.T @ selected_particles - np.eye(len(indices))))),
        "orbital_amplitude_convention": (
            "Unit-norm MO mixing; singular values are reported separately, not multiplied into cubes."),
        "phases": (
            "Raw SVD paired signs; each orbital sign is arbitrary. NTOs reconstruct source X "
            "before common field-phase alignment. Do not interpret orbital red/blue as moment sign."),
        "interpretation": (
            "Selected NTO orbital cubes only; weight coverage does NOT establish moment coverage."),
    }
    return nto_wfx, metadata


def verify_analysis_result(path, args, grid):
    """Require complete matching provenance before reusing its global field phase."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "complete" or not data.get("analysis", {}).get("is_complete_grid"):
        raise ValueError("Phase source is not a complete-grid analysis result")
    if data.get("engine") != args.engine or data.get("state") != args.state:
        raise ValueError("Phase source engine/state differs")
    if data.get("response_convention") != args.response_convention:
        raise ValueError("Phase source response convention differs")
    if data.get("min_abs_coefficient") != args.min_abs_coeff:
        raise ValueError("Phase source coefficient threshold differs")
    for key in ("lower_bohr", "upper_bohr", "steps_bohr"):
        if not np.allclose(data["grid"][key], grid.metadata()[key], atol=1e-11, rtol=0):
            raise ValueError(f"Phase source grid differs: {key}")
    if data["grid"]["counts"] != list(grid.counts):
        raise ValueError("Phase source grid counts differ")
    if not np.allclose(data["analysis"]["origin_bohr"], args.origin, atol=1e-12, rtol=0):
        raise ValueError("Phase source operator origin differs")
    hashes = {record["sha256"] for record in data["inputs"]}
    for path_arg in (args.output, args.wfx, args.json):
        if path_arg is not None and stream.sha256(path_arg) not in hashes:
            raise ValueError(f"Phase source input hashes do not match {path_arg}")
    harness_hashes = {
        item["sha256"] for item in data.get("implementation", [])
        if Path(item["path"].replace("\\", "/")).name in ("stream_analysis.py", "stream.py")
    }
    if stream.sha256(Path(stream.__file__)) not in harness_hashes:
        raise ValueError("Phase source used a different streaming harness snapshot")
    for record in data.get("implementation", []):
        name = Path(record["path"].replace("\\", "/")).name
        if name not in ("stream_analysis.py", "stream.py"):
            source = args.repo / "chiroptmap" / name
            if not source.is_file() or stream.sha256(source) != record["sha256"]:
                raise ValueError(f"Phase source used different ChiroptMap code: {name}")
    factor = data["analysis"]["global_phase_factor"]
    if factor not in (-1, 1):
        raise ValueError("Invalid global phase in analysis result")
    return factor, data


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--engine", choices=("orca", "gaussian"), required=True)
    p.add_argument("--json", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--wfx", type=Path, required=True)
    p.add_argument("--state", type=int, default=1)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--outprefix", default="selected_state")
    p.add_argument("--analysis-result", type=Path,
                   help="Complete matching stream_analysis JSON; avoids preliminary phase integration")
    p.add_argument("--response-convention", choices=("tda", "xplus_yminus"), default="tda")
    p.add_argument("--min-abs-coeff", type=float, default=0.)
    p.add_argument("--spacing", type=float, default=.2)
    p.add_argument("--padding", type=float, default=8.)
    p.add_argument("--snap", type=float, default=.4)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--bounds", type=float, nargs=6)
    group.add_argument("--reference-cube", type=Path)
    p.add_argument("--origin", type=float, nargs=3, default=(0., 0., 0.))
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--nto-pairs", type=int, default=2, help="Leading pairs (0 disables NTO cubes)")
    selection.add_argument("--nto-select", nargs="+", type=int, help="Specific one-based NTO pair indices")
    p.add_argument("--no-moment-maps", action="store_true", help="Only selected NTO orbitals")
    p.add_argument("--max-points", type=int, default=4096)
    p.add_argument("--primitive-chunk", type=int, default=64)
    p.add_argument("--skip-geometry-check", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    start = time.perf_counter()
    if args.nto_pairs < 0 or args.min_abs_coeff < 0 or not math.isfinite(args.min_abs_coeff):
        raise ValueError("NTO count and coefficient threshold must be nonnegative")
    if args.max_points < 1 or args.primitive_chunk < 1 or not np.isfinite(args.origin).all():
        raise ValueError("Invalid block size, primitive chunk or origin")
    if Path(args.outprefix).name != args.outprefix or args.outprefix in ("", ".", ".."):
        raise ValueError("outprefix must be a simple file-name stem")
    from chiroptmap.wfx import evaluate_mos
    (wfx, occupied, virtual, ew, mw, reference_e, reference_m, x, occ_full, virt_full,
     amplitude_meta) = stream._load_wavefunction(args)
    grid = (stream.cube_header_grid(args.reference_cube, wfx) if args.reference_cube else
            stream.snapped_grid(wfx.coordinates_bohr, args.spacing, args.padding, args.snap, args.bounds))
    pairs = args.nto_select if args.nto_select is not None else list(range(1, min(args.nto_pairs, min(x.shape)) + 1))
    if args.response_convention != "tda" and pairs:
        raise ValueError("NTO orbital export is restricted to TDA; use --nto-pairs 0 for full TDDFT")
    if args.no_moment_maps and not pairs:
        raise ValueError("No outputs requested")
    nto_wfx, nto_meta = selected_ntos(wfx, occ_full, virt_full, x, pairs) if pairs else (None, None)
    target_names = []
    if not args.no_moment_maps:
        target_names.extend((f"{args.outprefix}_EDTM_linear_projection.cube",
                             f"{args.outprefix}_MDTM_linear_projection.cube"))
    for pair in pairs:
        target_names.extend((f"{args.outprefix}_NTO{pair:02d}_hole_orbital.cube",
                             f"{args.outprefix}_NTO{pair:02d}_particle_orbital.cube"))
    manifest_path = args.outdir / f"{args.outprefix}_map_manifest.json"
    targets = [args.outdir / name for name in target_names] + [manifest_path]
    if not args.overwrite and any(path.exists() or path.with_name(path.name + ".partial").exists()
                                  for path in targets):
        raise FileExistsError("Output/partial files already exist; use a new directory or explicit --overwrite")
    args.outdir.mkdir(parents=True, exist_ok=True)
    phase = None
    phase_source = "not needed for NTO-only output"
    source_analysis = None
    if not args.no_moment_maps:
        if args.analysis_result:
            phase, source_analysis = verify_analysis_result(args.analysis_result, args, grid)
            phase_source = {"path": str(args.analysis_result.resolve()), "sha256": stream.sha256(args.analysis_result)}
        else:
            print("[phase] No complete analysis supplied; performing preliminary integration.", file=sys.stderr)
            phase_analysis = stream.integrate(
                wfx, occupied, virtual, ew, mw, reference_e, reference_m, grid, np.asarray(args.origin),
                ["all"], np.zeros(wfx.n_nuclei, int), args.max_points, args.primitive_chunk,
                evaluation="contracted")
            phase = phase_analysis["global_phase_factor"]
            phase_source = "preliminary complete-grid integration, common EDTM-aligned phase"
    eval_wfx, reuse = stream.contracted_wavefunction(wfx, occupied, virtual, ew, mw)
    ref_units = np.vstack((reference_e / np.linalg.norm(reference_e), reference_m / np.linalg.norm(reference_m)))
    field_sums = np.zeros(2)
    overlaps = np.zeros((2 * len(pairs), 2 * len(pairs)))
    completed, last_print = 0, time.perf_counter()
    with ExitStack() as stack:
        moment_writers = []
        if not args.no_moment_maps:
            for field in ("EDTM", "MDTM"):
                moment_writers.append(stack.enter_context(CubeStream(
                    args.outdir / f"{args.outprefix}_{field}_linear_projection.cube", grid, wfx,
                    f"{field} linear transition-moment projection; state {args.state}",
                    f"All retained source amplitudes; threshold={args.min_abs_coeff:g}; NOT NTO-truncated")))
        nto_writers = []
        # Evaluation rows are all holes then all particles, not alternating.
        for kind in ("hole", "particle"):
            for pair in pairs:
                weight = nto_meta["selected_pairs"][pairs.index(pair)]["normalized_weight"]
                nto_writers.append(stack.enter_context(CubeStream(
                    args.outdir / f"{args.outprefix}_NTO{pair:02d}_{kind}_orbital.cube", grid, wfx,
                    f"TDA NTO {pair} {kind} ORBITAL amplitude; state {args.state}",
                    f"Normalized MO mixing; pair weight={weight:.10g}; NOT a moment-density map")))
        for points in grid.blocks(args.max_points):
            if moment_writers:
                phi, gradients = evaluate_mos(eval_wfx, eval_wfx.mo_numbers, points, primitive_chunk=args.primitive_chunk)
                electric, magnetic, _, _ = stream.fields_from_contracted(
                    points, phi, gradients, len(occupied), reuse, np.asarray(args.origin))
                for q, field in enumerate((electric, magnetic)):
                    projected = phase * (field @ ref_units[q])
                    moment_writers[q].write(projected)
                    field_sums[q] += projected.sum()
            if nto_writers:
                values, _ = evaluate_mos(nto_wfx, nto_wfx.mo_numbers, points, primitive_chunk=args.primitive_chunk)
                overlaps += values.T @ values
                for column, writer in enumerate(nto_writers):
                    writer.write(values[:, column])
            completed += len(points)
            now = time.perf_counter()
            if now - last_print > 30 or completed == grid.size:
                print(f"[export] {completed}/{grid.size} points", file=sys.stderr, flush=True)
                last_print = now
    if nto_meta:
        nto_meta["finite_grid_orbital_overlap_matrix_holes_then_particles"] = (overlaps * grid.volume).tolist()
        nto_meta["grid_overlap_caution"] = (
            "A finite-grid quadrature diagnostic, distinct from exact MO-mixing orthonormality.")
    metadata = {
        "schema": "chiroptmap-selected-map-export-1", "status": "complete",
        "engine": args.engine, "state": args.state, "response_convention": args.response_convention,
        "arguments": vars(args), "grid": grid.metadata(),
        "inputs": [{"path": str(path.resolve()), "sha256": stream.sha256(path)}
                   for path in (args.output, args.wfx, args.json) if path is not None],
        "implementation": [
            {"path": str(path.resolve()), "sha256": stream.sha256(path)}
            for path in [Path(__file__), Path(stream.__file__)] + [
                args.repo / "chiroptmap" / name for name in
                ("wfx.py", "wfx_cubes.py", "orca_json.py", "orca.py", "gaussian.py", "densities.py")]],
        "amplitudes": amplitude_meta, "global_field_phase": phase, "phase_source": phase_source,
        "operator_origin_bohr": args.origin,
        "projection_directions_EDTM_MDTM": ref_units.tolist(),
        "projection_reference_origin_bohr": [0., 0., 0.],
        "projection_note": "Fixed original-output directions, including when the operator origin is shifted.",
        "linear_map_projected_integrals": (
            dict(zip(("EDTM", "MDTM"), (field_sums * grid.volume).tolist()))
            if not args.no_moment_maps else None),
        "nto": nto_meta,
        "separation_of_meanings": (
            "EDTM/MDTM maps use all retained SOURCE amplitudes independently of NTO selection. "
            "NTO cubes contain orbital amplitudes, not moment-density contributions or local g."),
        "files": [{"name": name, "sha256": stream.sha256(args.outdir / name),
                   "bytes": (args.outdir / name).stat().st_size} for name in target_names],
        "elapsed_seconds": time.perf_counter() - start, "peak_process_rss_bytes": stream.peak_rss_bytes(),
    }
    if source_analysis and moment_writers:
        metadata["projection_integral_differences_from_matching_analysis"] = {
            field: float(field_sums[q] * grid.volume - source_analysis["analysis"]["totals"][field]["signed_projection"])
            for q, field in enumerate(("EDTM", "MDTM"))}
    partial = manifest_path.with_name(manifest_path.name + ".partial")
    partial.write_text(json.dumps(
        metadata, indent=2, ensure_ascii=False, allow_nan=False,
        default=lambda obj: str(obj) if isinstance(obj, Path) else obj), encoding="utf-8")
    partial.replace(manifest_path)
    print(f"[manifest] {manifest_path.resolve()}", file=sys.stderr)
    return metadata


def cli():
    main()
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
