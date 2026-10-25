#!/usr/bin/env python3
"""Full-amplitude ORCA/TDA cancellation, partition and origin analysis.

Every exported X amplitude enters the physical fields. Exact virtual-orbital
contraction bounds grid memory without an NTO cutoff. Zero, ElOri and an
optional shifted operator origin share one traversal of a fixed Cartesian
box. No regions file is needed for whole-molecule analysis. This command
performs post-processing only and never submits an electronic-structure job.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


def load_backend():
    from . import stream, analytic, origins, maps
    return stream, analytic, origins, maps


def cancellation(positive, negative):
    absolute = float(positive + negative)
    return 1 - abs(float(positive - negative)) / absolute if absolute > 0 else None


def translated_moments(electric, magnetic, tau, current, origin):
    origin = np.asarray(origin, float)
    return (np.asarray(electric) - origin * float(tau),
            np.asarray(magnetic) - np.cross(origin, np.asarray(current)))


def shifted_fields(electric_zero, magnetic_zero, tau, current, origin):
    origin = np.asarray(origin, float)
    return electric_zero - tau[:, None] * origin, magnetic_zero - np.cross(origin, current)


def reference_record(electric, magnetic, tau, current, origin):
    e, b = translated_moments(electric, magnetic, tau, current, origin)
    en, bn = np.linalg.norm(e), np.linalg.norm(b)
    if en == 0 or bn == 0 or not np.isfinite(np.r_[e, b]).all():
        raise ValueError("A nonzero finite analytic electric and magnetic moment is required")
    return {"origin_bohr": np.asarray(origin, float), "electric": e, "magnetic": b,
            "electric_unit": e / en, "magnetic_unit": b / bn}


def common_partition_definition(wfx, labels, atom_to_region, centres_one_based):
    """Use a declared subset of nuclei as centres, retaining the full field."""
    indices = list(centres_one_based)
    if not indices or any(type(i) is not int or not 1 <= i <= wfx.n_nuclei for i in indices):
        raise ValueError("Common centres must be nonempty one-based integer atom indices")
    if len(indices) != len(set(indices)):
        raise ValueError("Common-centre indices must be unique")
    keep = np.asarray(indices) - 1
    present = sorted(set(np.asarray(atom_to_region)[keep]))
    local_labels = [labels[i] for i in present]
    remap = {old: new for new, old in enumerate(present)}
    mapping = np.asarray([remap[int(i)] for i in np.asarray(atom_to_region)[keep]])
    return {"labels": local_labels, "mapping": mapping,
            "centres": np.asarray(wfx.coordinates_bohr)[keep],
            "centres_one_based": indices,
            "definition": "Equal-radius Becke weights of the explicitly selected nuclei; the entire molecular field is integrated, including density outside the selected centres.",
            "composition_invariance_condition": "Weights are identical for identical shared-centre coordinates and sampling points; geometric displacements of shared centres can change the weights."}


def region_weights(atomic, mapping, nregions):
    return np.column_stack([atomic[:, np.asarray(mapping) == i].sum(axis=1)
                            for i in range(nregions)])


def make_partition_definitions(wfx, region_path, global_only, stream, common_centres=None):
    if global_only:
        return ({"whole_molecule": {
            "mode": "whole_molecule", "labels": ["whole_molecule"],
            "definition": "One region with weight identically one over the entire integration box; no atomic partition is evaluated."}},
            {"definition": "Whole molecule; no atom assignment required",
             "region_file_used": False})
    if region_path is None:
        raise ValueError("--regions is required unless --global-only is selected")
    labels, mapping, region_meta = stream.load_regions(region_path, wfx.n_nuclei)
    partitions = {"equal_becke_all_nuclei": {
                "labels": labels, "mapping": mapping, "centres": wfx.coordinates_bohr,
                "centres_one_based": list(range(1, wfx.n_nuclei + 1)),
                "definition": "Equal-radius Becke partition of every physical nucleus, three polynomial iterations."}}
    if common_centres is not None:
        data = json.loads(common_centres.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("centres_one_based"), list):
            raise ValueError("Common-centre JSON needs centres_one_based: [atom indices]")
        partitions["common_centres_becke"] = common_partition_definition(
            wfx, labels, mapping, data["centres_one_based"])
    return partitions, region_meta


class Accumulator:
    """Linear vectors and positive/negative scalar projection measures."""
    def __init__(self, labels):
        self.labels = labels
        self.vectors = np.zeros((2, len(labels), 3))
        self.positive = np.zeros((3, len(labels)))
        self.negative = np.zeros((3, len(labels)))
        self.partition_residual = 0.

    def update(self, electric, magnetic, reference, weights):
        self.partition_residual = max(self.partition_residual,
                                     float(np.max(np.abs(weights.sum(axis=1) - 1))))
        self.vectors[0] += weights.T @ electric
        self.vectors[1] += weights.T @ magnetic
        projections = (electric @ reference["electric_unit"],
                       magnetic @ reference["magnetic_unit"],
                       -magnetic @ reference["electric_unit"])
        for i, scalar in enumerate(projections):
            self.positive[i] += weights.T @ np.maximum(scalar, 0)
            self.negative[i] += weights.T @ np.maximum(-scalar, 0)

    def report(self, dvol, reference, total_e, total_b):
        vectors = self.vectors * dvol
        positive, negative = self.positive * dvol, self.negative * dvol
        projections = {name: {"positive_integral": float(positive[i].sum()),
                              "negative_absolute_integral": float(negative[i].sum()),
                              "signed_integral": float(positive[i].sum() - negative[i].sum()),
                              "absolute_integral": float(positive[i].sum() + negative[i].sum()),
                              "cancellation_fraction": cancellation(positive[i].sum(), negative[i].sum())}
                       for i, name in enumerate(("s_mu", "s_b", "s_R"))}
        regions = {}
        for i, label in enumerate(self.labels):
            regions[label] = {"electric_vector": vectors[0, i].tolist(),
                              "magnetic_vector": vectors[1, i].tolist(),
                              "q_region": float(-reference["electric"] @ vectors[1, i]),
                              "s_R_signed_integral": float(positive[2, i] - negative[2, i]),
                              "s_R_positive_integral": float(positive[2, i]),
                              "s_R_negative_absolute_integral": float(negative[2, i])}
        q_grid = float(-reference["electric"] @ total_b)
        qsum = sum(row["q_region"] for row in regions.values())
        residual = vectors.sum(axis=1) - np.asarray([total_e, total_b])
        return {"C_b": projections["s_b"]["cancellation_fraction"],
                "C_R": projections["s_R"]["cancellation_fraction"],
                "projections": projections, "regions": regions,
                "q_grid": q_grid, "q_region_sum": qsum,
                "q_additivity_absolute_residual": abs(qsum - q_grid),
                "vector_additivity_maximum_absolute_residual": float(np.max(np.abs(residual))),
                "maximum_partition_unity_residual": self.partition_residual,
                "regional_vector_cancellation_index_b": (
                    1 - float(np.linalg.norm(total_b)) / float(np.linalg.norm(vectors[1], axis=1).sum())
                    if np.linalg.norm(vectors[1], axis=1).sum() > 0 else None)}


def map_subgrid(grid, stride, stream):
    return stream.Grid(grid.lower.copy(), grid.steps * stride,
                       tuple((n - 1) // stride + 1 for n in grid.counts),
                       f"Every {stride}th point on each axis of the integration grid")


def sample_mask(start, n, counts, stride):
    linear = np.arange(start, start + n, dtype=np.int64)
    ny, nz = counts[1:]
    return ((linear // (ny * nz) % stride == 0)
            & (linear // nz % ny % stride == 0) & (linear % nz % stride == 0))


def read_analytic_reference(path, state, input_records):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != "chiroptmap-analytic-primitive-audit-1" or data.get("state") != state:
        raise ValueError("Analytic reference schema/state mismatch")
    hashes = {item["sha256"] for item in data["inputs"]}
    if hashes != {item["sha256"] for item in input_records}:
        raise ValueError("Analytic reference does not match the three current input hashes")
    selected = [r for r in data["results"] if r["threshold"] == 0]
    if len(selected) != 1:
        raise ValueError("Analytic reference needs exactly one all-X threshold=0 result")
    row = selected[0]
    phase = row["global_phase"]
    if phase not in (-1, 1):
        raise ValueError("Invalid analytic field phase")
    return data, (np.asarray(row["EDTM"]["integrated_vector"]),
                  np.asarray(row["MDTM"]["integrated_vector"]),
                  float(row["integrated_charge_weighted_tau"]),
                  np.asarray(row["integrated_magnetic_current_proxy"]), phase)


def write_json_new(path, data):
    path = Path(path)
    partial = path.with_name(path.name + ".partial")
    if path.exists() or partial.exists():
        raise FileExistsError(f"Use a new output path; refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text(json.dumps(data, indent=2, allow_nan=False,
                                 default=lambda x: str(x) if isinstance(x, Path) else x) + "\n",
                       encoding="utf-8")
    partial.replace(path)


def compute_analytic(wfx, occ, virt, ew, mw, e_printed, b_printed, state,
                     inputs, primitive_block, backend):
    stream, analytic, _, _ = backend
    rows, diagnostic = analytic.analytic_moments(wfx, occ, virt, [ew], [mw],
                                                block_size=primitive_block)
    row = rows[0]
    phase = -1 if np.dot(row["EDTM"], e_printed) < 0 else 1
    e, b = phase * np.asarray(row["EDTM"]), phase * np.asarray(row["MDTM"])
    tau = phase * row["integrated_charge_weighted_tau"]
    current = phase * np.asarray(row["integrated_magnetic_current_proxy"])
    row = {"threshold": 0., "global_phase": phase,
           "EDTM": {"integrated_vector": e.tolist(), "operator_origin_bohr": [0., 0., 0.]},
           "MDTM": {"integrated_vector": b.tolist(), "operator_origin_bohr": [0., 0., 0.]},
           "integrated_charge_weighted_tau": tau,
           "integrated_magnetic_current_proxy": current.tolist()}
    data = {"schema": "chiroptmap-analytic-primitive-audit-1", "state": state,
            "method": "All-space analytic Cartesian-Gaussian integrals; no spatial grid or finite box",
            "inputs": inputs, "diagnostics": diagnostic, "results": [row],
            "script_sha256": stream.sha256(Path(analytic.__file__)),
            "magnetic_convention": "Inherited reduced magnetic vector; twice the printed ORCA CD table vector"}
    return data, (e, b, tau, current, phase)


def integrate(args, wfx, occ, virt, ew, mw, x, occ_full, virt_full, references,
              partitions, grid, phase, backend):
    stream, _, _, maps = backend
    from chiroptmap.wfx import evaluate_mos
    transformed, reuse = stream.contracted_wavefunction(wfx, occ, virt, ew, mw)
    accumulators = {origin: {name: Accumulator(part["labels"]) for name, part in partitions.items()}
                    for origin in references}
    totals = {name: np.zeros((2, 3)) for name in references}
    tau_sum, current_sum = 0., np.zeros(3)
    start_time, last_log, completed = time.perf_counter(), -1., 0
    limit = min(grid.size, args.pilot_points) if args.pilot_points else grid.size
    map_files, map_sums = [], np.zeros(3)
    nto_meta, map_grid = None, None
    with ExitStack() as stack:
        writers = {}
        if args.cube_dir:
            if limit != grid.size:
                raise ValueError("Cube export requires a complete integration; omit --cube-dir for pilots")
            kinds = ("s_b", "s_R", "q_density", "NTO01_hole", "NTO01_particle")
            for kind in kinds:
                target = args.cube_dir / f"state{args.state}_{kind}.cube"
                if target.exists() or target.with_name(target.name + ".partial").exists():
                    raise FileExistsError(f"Use a new map directory: {target}")
            args.cube_dir.mkdir(parents=True, exist_ok=True)
            map_grid = map_subgrid(grid, args.map_stride, stream)
            for kind in kinds:
                path = args.cube_dir / f"state{args.state}_{kind}.cube"
                writers[kind] = stack.enter_context(maps.CubeStream(
                    path, map_grid, wfx, f"State {args.state}: {kind}; operator origin Cartesian zero",
                    "All-X moment fields; NTO cubes are auxiliary unit-mixing orbital amplitudes"))
                map_files.append(path)
            nto_wfx, nto_meta = maps.selected_ntos(wfx, occ_full, virt_full, x, [1])
        for points in grid.blocks(args.max_points):
            points = points[:min(len(points), limit - completed)]
            phi, grad = evaluate_mos(transformed, transformed.mo_numbers, points,
                                     dtype=np.float64, primitive_chunk=args.primitive_chunk)
            e0, b0, tau, current = stream.fields_from_contracted(
                points, phi, grad, len(occ), reuse, np.zeros(3))
            e0, b0, tau, current = phase * e0, phase * b0, phase * tau, phase * current
            weights = {name: (np.ones((len(points), 1)) if part.get("mode") == "whole_molecule" else
                             region_weights(stream.becke_atomic_weights(points, part["centres"]),
                                            part["mapping"], len(part["labels"])))
                       for name, part in partitions.items()}
            for origin, reference in references.items():
                e, b = shifted_fields(e0, b0, tau, current, reference["origin_bohr"])
                totals[origin] += np.asarray([e.sum(axis=0), b.sum(axis=0)])
                for name, weight in weights.items():
                    accumulators[origin][name].update(e, b, reference, weight)
            tau_sum += float(tau.sum())
            current_sum += current.sum(axis=0)
            if writers:
                choose = sample_mask(completed, len(points), grid.counts, args.map_stride)
                if choose.any():
                    zero = references["common_zero"]
                    sb = b0[choose] @ zero["magnetic_unit"]
                    sr = -b0[choose] @ zero["electric_unit"]
                    qd = -b0[choose] @ zero["electric"]
                    for i, (name, values) in enumerate(zip(("s_b", "s_R", "q_density"), (sb, sr, qd))):
                        writers[name].write(values)
                        map_sums[i] += values.sum()
                    nto_values, _ = evaluate_mos(nto_wfx, nto_wfx.mo_numbers, points[choose],
                                                 dtype=np.float64, primitive_chunk=args.primitive_chunk)
                    writers["NTO01_hole"].write(nto_values[:, 0])
                    writers["NTO01_particle"].write(nto_values[:, 1])
            completed += len(points)
            elapsed = time.perf_counter() - start_time
            if elapsed - last_log >= 30 or completed == limit:
                print(f"[state {args.state}] {completed}/{grid.size} points; {elapsed:.1f}s", file=sys.stderr, flush=True)
                last_log = elapsed
            if completed == limit:
                break
    dvol = grid.volume
    tau_sum, current_sum = tau_sum * dvol, current_sum * dvol
    totals = {key: value * dvol for key, value in totals.items()}
    origins = {}
    for name, ref in references.items():
        e, b = totals[name]
        expected = translated_moments(*totals["common_zero"], tau_sum, current_sum, ref["origin_bohr"])
        origins[name] = {
            "origin_bohr": ref["origin_bohr"].tolist(), "electric_vector": e.tolist(), "magnetic_vector": b.tolist(),
            "electric_norm": float(np.linalg.norm(e)), "magnetic_norm": float(np.linalg.norm(b)),
            "analytic_electric_vector": ref["electric"].tolist(), "analytic_magnetic_vector": ref["magnetic"].tolist(),
            "cos_mu_b_analytic": float(ref["electric_unit"] @ ref["magnetic_unit"]),
            "q_analytic": float(-ref["electric"] @ ref["magnetic"]),
            "analytic_reference_errors": {"electric": stream.error_metrics(e, ref["electric"]),
                                          "magnetic": stream.error_metrics(b, ref["magnetic"])},
            "origin_translation_maximum_absolute_residual": float(np.max(np.abs(np.asarray(expected) - totals[name]))),
            "partitions": {part: accumulator.report(dvol, ref, e, b)
                           for part, accumulator in accumulators[name].items()}}
    changes = {}
    for origin in references:
        if origin == "common_zero":
            continue
        changes[origin] = {}
        for part in partitions:
            zero = origins["common_zero"]["partitions"][part]
            target = origins[origin]["partitions"][part]
            changes[origin][part] = {"C_b_difference": (target["C_b"] - zero["C_b"]
                                                        if target["C_b"] is not None and zero["C_b"] is not None else None),
                                    "C_R_difference": (target["C_R"] - zero["C_R"]
                                                        if target["C_R"] is not None and zero["C_R"] is not None else None),
                                    "regional_q_differences": {label: target["regions"][label]["q_region"] - row["q_region"]
                                                               for label, row in zero["regions"].items()}}
    return {"is_complete_grid": completed == grid.size, "processed_points": completed, "grid_points": grid.size,
            "scientific_results_eligible": completed == grid.size,
            "pilot_warning": None if completed == grid.size else "INCOMPLETE prefix pilot: all integral/cancellation/region outputs are resource diagnostics and must not be used as molecular results.",
            "origins": origins, "changes_from_common_zero": changes,
            "integrated_tau": tau_sum, "integrated_code_current_proxy": current_sum.tolist(),
            "elapsed_seconds": time.perf_counter() - start_time,
            "rough_full_grid_seconds_from_prefix": ((time.perf_counter() - start_time) * grid.size / completed
                                                       if completed < grid.size else None),
            "exact_weighted_virtual_contraction": True, "NTO_rank_cutoff_in_physical_fields": None,
            "evaluated_orbital_combinations": len(transformed.mo_numbers),
            "map_export": {"grid": map_grid.metadata() if map_grid else None,
                           "files": [{"path": str(p.resolve()), "sha256": stream.sha256(p), "bytes": p.stat().st_size}
                                     for p in map_files],
                           "scalar_grid_integrals": dict(zip(("s_b", "s_R", "q_density"),
                                                               (map_sums * map_grid.volume).tolist())) if map_grid else None,
                           "NTO": nto_meta,
                           "note": "Maps sample the integration grid at the specified stride; only the full integration-grid accumulators define the cancellation results."}}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", type=Path, required=True)
    p.add_argument("--wfx", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="ORCA TDA output")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--regions", type=Path, help="One-based exhaustive atom-to-region JSON for equal-radius Becke weights")
    group.add_argument("--global-only", action="store_true", help="One whole-molecule region (also the default without --regions)")
    p.add_argument("--common-centres", type=Path, help="Optional JSON subset of partition centres; requires --regions")
    p.add_argument("--state", type=int, required=True, help="Positive one-based singlet TDA root")
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--analytic-reference", type=Path, help="Existing matching all-X analytic JSON; otherwise compute and checkpoint it")
    p.add_argument("--spacing", type=float, default=.2)
    p.add_argument("--padding", type=float, default=8.)
    p.add_argument("--snap", type=float, default=.4)
    p.add_argument("--bounds", nargs=6, type=float, required=True, help="Identical physical X0 Y0 Z0 X1 Y1 Z1 for all cases/states")
    p.add_argument("--max-points", type=int, default=4096)
    p.add_argument("--primitive-chunk", type=int, default=64)
    p.add_argument("--primitive-block", type=int, default=64)
    p.add_argument("--pilot-points", type=int)
    p.add_argument("--cube-dir", type=Path)
    p.add_argument("--map-stride", type=int, default=2)
    origins = p.add_mutually_exclusive_group()
    origins.add_argument("--shift-origin", type=float, nargs=3, help="Additional operator origin X Y Z in bohr")
    origins.add_argument("--axis-shift", type=float, help="Additional operator origin (dx,0,0) in bohr")
    p.add_argument("--dry-run", action="store_true", help="Validate inputs/grid/NTO/partitions without analytic/grid integration")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    args.global_only = args.global_only or args.regions is None
    if args.state < 1:
        raise ValueError("State must be a positive one-based root")
    if args.common_centres is not None and args.regions is None:
        raise ValueError("--common-centres requires --regions")
    if min(args.max_points, args.primitive_chunk, args.primitive_block, args.map_stride) < 1:
        raise ValueError("Block sizes and map stride must be positive")
    if args.pilot_points is not None and args.pilot_points < 1:
        raise ValueError("Pilot points must be positive")
    if not np.isfinite(args.bounds).all() or (args.axis_shift is not None and not math.isfinite(args.axis_shift)):
        raise ValueError("Bounds and origins must be finite")
    if args.shift_origin is not None and not np.isfinite(args.shift_origin).all():
        raise ValueError("Shifted origin must be finite")
    for target in (args.result, args.result.with_name(args.result.name + ".partial")):
        if target.exists():
            raise FileExistsError(f"Use a new output path: {target}")
    backend = load_backend()
    stream, analytic, origins, _ = backend
    # The inherited parser's options are explicit here: threshold zero and TDA.
    loader_options = argparse.Namespace(engine="orca", json=args.json, wfx=args.wfx, output=args.output,
                                        state=args.state, min_abs_coeff=0., response_convention="tda",
                                        skip_geometry_check=False)
    wfx, occ, virt, ew, mw, ep, bp, x, occ_full, virt_full, amplitude = stream._load_wavefunction(loader_options)
    partitions, region_meta = make_partition_definitions(wfx, args.regions, args.global_only, stream, args.common_centres)
    parent = origins.parse_orca_transition_origin(args.output.read_text(errors="replace"), wfx.coordinates_bohr)
    grid = stream.snapped_grid(wfx.coordinates_bohr, args.spacing, args.padding, args.snap, args.bounds)
    summary, _, _ = stream.nto_decomposition(x)
    inputs = [{"path": str(p.resolve()), "sha256": stream.sha256(p)} for p in (args.json, args.wfx, args.output)]
    sources = [Path(__file__), Path(stream.__file__), Path(analytic.__file__), Path(origins.__file__), Path(backend[3].__file__)]
    package = Path(__file__).resolve().parent.parent
    sources += [package / name for name in ("wfx.py", "wfx_cubes.py", "orca_json.py", "orca.py", "gaussian.py", "densities.py")]
    report = {"schema": "chiroptmap-state-fields-1", "state": args.state, "engine": "orca", "response_convention": "tda",
              "status": "dry-run" if args.dry_run else "pending", "arguments": vars(args),
              "min_abs_coefficient": 0., "input_amplitudes": "Complete JSON X; no amplitude threshold; no NTO cutoff in physical fields",
              "inputs": inputs, "implementation": [{"path": str(p.resolve()), "sha256": stream.sha256(p)} for p in sources],
              "grid": grid.metadata(), "amplitudes": amplitude, "NTO": summary,
              "atomic_numbers": wfx.atomic_numbers.tolist(), "coordinates_bohr": wfx.coordinates_bohr.tolist(),
              "region_source": region_meta, "global_only": args.global_only,
              "partition_inputs": [{"sha256": stream.sha256(p), "path": str(p.resolve())}
                                   for p in (args.regions, args.common_centres) if p is not None],
              "partitions": {name: {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                     for k, v in part.items() if k != "centres"}
                             for name, part in partitions.items()}, "parent_origin": parent,
              "definitions": {"rho_b": "(r-O) cross j_proxy, inherited real reduced-magnetic convention",
                              "s_b": "b_hat_analytic(O) dot rho_b(O)",
                              "s_R": "-mu_hat_analytic(O) dot rho_b(O)",
                              "q_density": "-mu_analytic(O) dot rho_b(O) = |mu_analytic(O)| s_R(O)",
                              "q_region": "Integral of partition_weight * q_density; same units as inherited q_A; mu_B omitted",
                              "C_b_and_C_R": "1-|P-N|/(P+N), using positive and absolute-negative scalar integrals",
                              "origins": "Zero and ElOri use identical points/weights and are evaluated in one pass from tau and j_proxy. Analytic reference directions are translated to the same operator origin.",
                              "phase": "One analytic electric-aligned phase is used for E, b, tau and j at all origins and partitions.",
                              "symmetry": "No A/B state or C2 label is assigned by this postprocessor."},
              "limitations": ["Inherited magnetic normalization is preserved; this adapter does not independently calibrate its physical units.",
                              "Grid completion is distinct from grid-spacing and box-size convergence.",
                              "Common-centre weights retain geometric dependence and do not isolate a substituent observable.",
                              "NTO orbital maps are auxiliary orbital amplitudes; NTO weight coverage does not measure moment coverage."]}
    if not args.dry_run:
        if args.analytic_reference:
            audit, reference = read_analytic_reference(args.analytic_reference, args.state, inputs)
            analytic_path = args.analytic_reference
        else:
            audit, reference = compute_analytic(wfx, occ, virt, ew, mw, ep, bp, args.state, inputs,
                                                args.primitive_block, backend)
            analytic_path = args.result.with_name(args.result.stem + ".analytic.json")
            write_json_new(analytic_path, audit)
        e, b, tau, current, phase = reference
        refs = {"common_zero": reference_record(e, b, tau, current, [0, 0, 0]),
                "ElOri": reference_record(e, b, tau, current, parent["transition_property_origin_bohr"])}
        if args.axis_shift is not None:
            refs["x_axis_shift"] = reference_record(e, b, tau, current, [args.axis_shift, 0, 0])
        if args.shift_origin is not None:
            refs["shifted"] = reference_record(e, b, tau, current, args.shift_origin)
        report["analytic_reference"] = {"path": str(analytic_path.resolve()), "sha256": stream.sha256(analytic_path),
                                        "global_phase": phase, "moments_zero": {"electric": e.tolist(), "magnetic": b.tolist()},
                                        "analytic_probe_orthonormality_maximum_error": audit["diagnostics"].get("probe_orthonormality_maximum_error")}
        report["analysis"] = integrate(args, wfx, occ, virt, ew, mw, x, occ_full, virt_full,
                                        refs, partitions, grid, phase, backend)
        report["status"] = "complete" if report["analysis"]["is_complete_grid"] else "INCOMPLETE-PILOT"
        parent_result = report["analysis"]["origins"]["ElOri"]
        report["parent_validation_at_ElOri"] = {"electric": stream.error_metrics(np.asarray(parent_result["electric_vector"]), ep),
                                               "magnetic": stream.error_metrics(np.asarray(parent_result["magnetic_vector"]), bp)}
    write_json_new(args.result, report)
    print(f"[{report['status']}] {args.result}", file=sys.stderr, flush=True)
    return report


def cli():
    main()
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
