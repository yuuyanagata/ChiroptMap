#!/usr/bin/env python3
"""Independent all-space Cartesian-Gaussian moment integrals: no spatial grid.

WFX coefficients multiply unnormalized primitives; no extra normalization is
introduced. This diagnostic reuses the input bridge, not the grid evaluator.
It does not independently establish the physical magnetic unit convention.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from . import stream


def axis_integrals(alpha, beta, centers_a, centers_b, angular_a, angular_b):
    """Return S, position R, and antisymmetrized derivative D for one axis."""
    alpha, beta = np.asarray(alpha)[:, None], np.asarray(beta)[None, :]
    ca, cb = np.asarray(centers_a)[:, None], np.asarray(centers_b)[None, :]
    p = alpha + beta
    product_center = (alpha * ca + beta * cb) / p
    pa, pb = product_center - ca, product_center - cb
    ilim, jlim = int(np.max(angular_a)) + 1, int(np.max(angular_b)) + 1
    table = np.empty((ilim + 1, jlim + 1, len(angular_a), len(angular_b)))
    table[0, 0] = np.sqrt(math.pi / p) * np.exp(-alpha * beta / p * (ca - cb) ** 2)
    for j in range(1, jlim + 1):
        table[0, j] = pb * table[0, j - 1]
        if j > 1:
            table[0, j] += (j - 1) / (2 * p) * table[0, j - 2]
    for i in range(1, ilim + 1):
        for j in range(jlim + 1):
            table[i, j] = pa * table[i - 1, j]
            if i > 1:
                table[i, j] += (i - 1) / (2 * p) * table[i - 2, j]
            if j:
                table[i, j] += j / (2 * p) * table[i - 1, j - 1]
    la, lb = np.asarray(angular_a)[:, None], np.asarray(angular_b)[None, :]
    ia, ib = np.arange(len(angular_a))[:, None], np.arange(len(angular_b))[None, :]
    overlap = table[la, lb, ia, ib]
    a_up, b_up = table[la + 1, lb, ia, ib], table[la, lb + 1, ia, ib]
    position = a_up + ca * overlap
    derivative = (
        la * table[np.maximum(la - 1, 0), lb, ia, ib] - 2 * alpha * a_up
        - lb * table[la, np.maximum(lb - 1, 0), ia, ib] + 2 * beta * b_up)
    return overlap, position, derivative


def primitive_operator_block(exponents, centers, angular, start, stop):
    axes = [
        axis_integrals(exponents[start:stop], exponents, centers[start:stop, axis], centers[:, axis],
                       angular[start:stop, axis], angular[:, axis])
        for axis in range(3)]
    (sx, rx, dx), (sy, ry, dy), (sz, rz, dz) = axes
    return {
        "overlap": sx * sy * sz,
        "EDTM_x": rx * sy * sz, "EDTM_y": sx * ry * sz, "EDTM_z": sx * sy * rz,
        "MDTM_x": sx * (ry * dz - rz * dy),
        "MDTM_y": sy * (rz * dx - rx * dz),
        "MDTM_z": sz * (rx * dy - ry * dx),
        "current_x": dx * sy * sz, "current_y": sx * dy * sz, "current_z": sx * sy * dz,
    }


def analytic_moments(wfx, occupied, virtual, electric_weights, magnetic_weights,
                     block_size=64, progress=True):
    """Integrate several weight matrices against identical primitives in blocks."""
    if block_size < 1 or not electric_weights or len(electric_weights) != len(magnetic_weights):
        raise ValueError("Require a positive block size and matching nonempty weight lists")
    from chiroptmap.wfx import primitive_angular_powers
    indices = {n: i for i, n in enumerate(wfx.mo_numbers)}
    c_occ = wfx.mo_coefficients[[indices[n] for n in occupied]]
    c_virt = wfx.mo_coefficients[[indices[n] for n in virtual]]
    ntargets = len(electric_weights)
    weighted_e = np.vstack([weights @ c_virt for weights in electric_weights])
    weighted_m = np.vstack([weights @ c_virt for weights in magnetic_weights])
    centers = wfx.coordinates_bohr[wfx.primitive_centers]
    angular = np.array([primitive_angular_powers(int(t)) for t in wfx.primitive_types])
    totals = {key: np.zeros(ntargets) for key in
              ("overlap", "EDTM_x", "EDTM_y", "EDTM_z", "MDTM_x", "MDTM_y", "MDTM_z",
               "current_x", "current_y", "current_z")}
    probe_indices = sorted(set(
        [0, wfx.n_electrons // 2 - 1, wfx.n_electrons // 2, wfx.n_mos - 1]
        + np.linspace(0, wfx.n_mos - 1, min(12, wfx.n_mos), dtype=int).tolist()))
    probe_coeff = wfx.mo_coefficients[probe_indices]
    probe_gram = np.zeros((len(probe_indices), len(probe_indices)))
    start_time, last = time.perf_counter(), time.perf_counter()
    for start in range(0, wfx.n_primitives, block_size):
        stop = min(start + block_size, wfx.n_primitives)
        operators = primitive_operator_block(wfx.primitive_exponents, centers, angular, start, stop)
        lhs = np.tile(c_occ[:, start:stop].T, (1, ntargets))
        for key, operator in operators.items():
            weighted = weighted_e if key.startswith("EDTM") or key == "overlap" else weighted_m
            contracted = operator @ weighted.T
            totals[key] += np.einsum("ij,ij->j", lhs, contracted).reshape(ntargets, len(occupied)).sum(axis=1)
        probe_gram += probe_coeff[:, start:stop] @ (operators["overlap"] @ probe_coeff.T)
        now = time.perf_counter()
        if progress and (now - last > 30 or stop == wfx.n_primitives):
            print(f"[analytic] primitive rows {stop}/{wfx.n_primitives} ({now-start_time:.1f}s)",
                  file=sys.stderr, flush=True)
            last = now
    results = [{
        "EDTM": [float(totals[f"EDTM_{c}"][i]) for c in "xyz"],
        "MDTM": [float(totals[f"MDTM_{c}"][i]) for c in "xyz"],
        "integrated_charge_weighted_tau": float(totals["overlap"][i]),
        "integrated_magnetic_current_proxy": [float(totals[f"current_{c}"][i]) for c in "xyz"],
    } for i in range(ntargets)]
    return results, {
        "probe_MO_numbers": [wfx.mo_numbers[i] for i in probe_indices],
        "probe_overlap_matrix": probe_gram.tolist(),
        "probe_orthonormality_maximum_error": float(np.max(np.abs(probe_gram - np.eye(len(probe_indices))))),
        "elapsed_seconds": time.perf_counter() - start_time,
        "primitive_row_block": block_size, "n_primitives": wfx.n_primitives,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", type=Path, required=True)
    p.add_argument("--wfx", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--state", type=int, default=1)
    p.add_argument("--thresholds", nargs="+", type=float, default=(0., 1e-4))
    p.add_argument("--primitive-block", type=int, default=64)
    p.add_argument("--result", type=Path, required=True)
    args = p.parse_args(argv)
    if args.state < 1 or args.primitive_block < 1:
        raise ValueError("State and primitive block must be positive")
    if any(not math.isfinite(t) or t < 0 for t in args.thresholds):
        raise ValueError("Thresholds must be finite and nonnegative")
    if args.result.exists() or args.result.with_name(args.result.name + ".partial").exists():
        raise FileExistsError(args.result)
    from chiroptmap.orca_json import read_orca_json_wavefunction
    from chiroptmap.orca import parse_absorption_spectrum_all_states, parse_cd_spectrum_all_states
    bridge = read_orca_json_wavefunction(args.json, args.wfx)
    root = next(root for root in bridge.roots if root.state == args.state)
    text = args.output.read_text(encoding="utf-8", errors="ignore")
    from .origins import parse_orca_transition_origin, translate_vectors
    origin_meta = parse_orca_transition_origin(text, bridge.wfx.coordinates_bohr)
    origin = origin_meta["transition_property_origin_bohr"]
    e_ref = np.array(parse_absorption_spectrum_all_states(text)[args.state][3:])
    m_printed = np.array(parse_cd_spectrum_all_states(text)[args.state][1:])
    m_ref = 2 * m_printed
    selected = []
    for threshold in args.thresholds:
        if threshold < 0:
            raise ValueError("Negative threshold")
        x = root.x_amplitudes.copy()
        x[np.abs(x) < threshold] = 0
        selected.append(x)
    results, diagnostics = analytic_moments(
        bridge.wfx, root.occupied_mos, root.virtual_mos,
        [-math.sqrt(2) * x for x in selected], [-x / math.sqrt(2) for x in selected],
        block_size=args.primitive_block)
    for threshold, x, result in zip(args.thresholds, selected, results):
        phase = -1 if np.dot(result["EDTM"], e_ref) < 0 else 1
        result["threshold"] = threshold
        result["retained_sum_X2"] = float(np.sum(x ** 2))
        result["retained_nonzero_pairs"] = int(np.count_nonzero(x))
        result["global_phase"] = phase
        e_zero, b_zero = phase * np.asarray(result["EDTM"]), phase * np.asarray(result["MDTM"])
        result["EDTM"] = {"integrated_vector": e_zero.tolist(), "operator_origin_bohr": [0., 0., 0.]}
        result["MDTM"] = {"integrated_vector": b_zero.tolist(), "operator_origin_bohr": [0., 0., 0.]}
        result["integrated_charge_weighted_tau"] *= phase
        result["integrated_magnetic_current_proxy"] = (
            phase * np.asarray(result["integrated_magnetic_current_proxy"])).tolist()
        e_parent, b_parent = translate_vectors(e_zero, b_zero, result["integrated_charge_weighted_tau"],
                                               result["integrated_magnetic_current_proxy"], origin)
        result["parent_validation_at_ElOri"] = {
            "electric": stream.error_metrics(e_parent, e_ref),
            "magnetic": stream.error_metrics(b_parent, m_ref)}
    report = {
        "schema": "chiroptmap-analytic-primitive-audit-1",
        "method": "All-space analytic Cartesian-Gaussian integrals; no spatial grid or finite box",
        "state": args.state,
        "parent_origin": origin_meta,
        "inputs": [{"path": str(path.resolve()), "sha256": stream.sha256(path)}
                   for path in (args.json, args.wfx, args.output)],
        "script_sha256": stream.sha256(Path(__file__)),
        "bridge": asdict(bridge.diagnostics), "diagnostics": diagnostics, "results": results,
        "printed_ORCA_magnetic_vector": m_printed.tolist(),
        "printed_magnetic_reference_factor": 2,
        "conventions": ["WFX primitive functions are unnormalized; WFX coefficients already contain normalization.",
                        "EDTM factor -sqrt(2)X; MDTM factor -X/sqrt(2), inherited code convention.",
                        "Operator origin is Cartesian (0,0,0) in the exported wavefunction frame.",
                        "Full-space primitive algebra checks numerical recovery, not independent unit theory."],
        "peak_process_rss_bytes": stream.peak_rss_bytes(),
    }
    from .state_fields import write_json_new
    write_json_new(args.result, report)
    print(json.dumps({"diagnostics": diagnostics, "results": results}, indent=2))


if __name__ == "__main__":
    main()
