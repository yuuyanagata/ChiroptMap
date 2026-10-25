"""Check explicit numerical targets on two matching, complete state-field grids."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

import numpy as np

from .state_fields import write_json_new
from .stream import sha256


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _vector(value, label):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{label} must contain three components")
    return np.asarray([_number(v, label) for v in value])


def _hashes(records, label, count=None):
    hashes = [r.get("sha256") for r in records]
    if (not hashes or (count is not None and len(hashes) != count)
            or any(not isinstance(h, str) or not re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes)):
        raise ValueError(f"Missing or invalid {label} SHA-256 provenance")
    return sorted(hashes)


def _complete(report):
    if (report.get("schema") != "chiroptmap-state-fields-1" or report.get("status") != "complete"
            or report.get("engine") != "orca" or report.get("response_convention") != "tda"
            or type(report.get("state")) is not int or report["state"] < 1
            or report.get("min_abs_coefficient") != 0):
        raise ValueError("Require a complete, all-amplitude ORCA/TDA state-fields result")
    analysis, grid = report["analysis"], report["grid"]
    n = grid["points"]
    if (type(n) is not int or n <= 0 or analysis.get("is_complete_grid") is not True
            or analysis.get("scientific_results_eligible") is not True
            or type(analysis.get("processed_points")) is not int
            or type(analysis.get("grid_points")) is not int
            or analysis["processed_points"] != n or analysis["grid_points"] != n
            or analysis.get("NTO_rank_cutoff_in_physical_fields", "missing") is not None
            or analysis.get("exact_weighted_virtual_contraction") is not True):
        raise ValueError("Incomplete/pilot/truncated grid is not eligible for a precision check")
    counts = grid["counts"]
    if (len(counts) != 3 or any(type(v) is not int or v < 2 for v in counts)
            or math.prod(counts) != n):
        raise ValueError("Grid point count does not match its dimensions")
    lower, upper, step = (_vector(grid[k], k) for k in ("lower_bohr", "upper_bohr", "steps_bohr"))
    if (np.any(step <= 0) or np.any(upper <= lower)
            or not np.allclose(lower + (np.asarray(counts) - 1) * step, upper, rtol=0, atol=1e-10)
            or not math.isclose(_number(grid["voxel_volume_bohr3"], "voxel volume"),
                                float(np.prod(step)), rel_tol=1e-12, abs_tol=0)):
        raise ValueError("Inconsistent grid bounds, steps or voxel volume")
    return lower, upper, step


def compare_grids(coarse, fine, *, electric_rtol, magnetic_rtol, cancellation_atol,
                  origin="ElOri", partition="whole_molecule"):
    """Return a numerical-target verdict; incompatible inputs raise ValueError.

    Errors are recomputed from vectors, never inferred from completion flags.
    The verdict applies only to this box, state, origin, partition and targets.
    """
    targets = {"electric_relative_vector_error": _number(electric_rtol, "electric target"),
               "magnetic_relative_vector_error": _number(magnetic_rtol, "magnetic target"),
               "C_R_absolute_difference": _number(cancellation_atol, "cancellation target")}
    if any(v < 0 for v in targets.values()):
        raise ValueError("Numerical targets must be nonnegative")
    lo_c, hi_c, h_c = _complete(coarse)
    lo_f, hi_f, h_f = _complete(fine)
    for key in ("state", "engine", "response_convention", "partitions", "atomic_numbers", "coordinates_bohr"):
        if key not in coarse or key not in fine or coarse[key] != fine[key]:
            raise ValueError(f"Mismatched or missing {key}")
    for key, count in (("inputs", 3), ("implementation", None)):
        if _hashes(coarse[key], key, count) != _hashes(fine[key], key, count):
            raise ValueError(f"Mismatched {key} hashes")
    # Historical reports have no partition_inputs; their full definitions and
    # geometries above still have to match. New reports additionally hash files.
    pc, pf = coarse.get("partition_inputs", []), fine.get("partition_inputs", [])
    if pc or pf:
        if _hashes(pc, "partition inputs") != _hashes(pf, "partition inputs"):
            raise ValueError("Mismatched partition input hashes")
    if not np.allclose(lo_c, lo_f, rtol=0, atol=1e-10) or not np.allclose(hi_c, hi_f, rtol=0, atol=1e-10):
        raise ValueError("Require identical physical box bounds")
    ratio = h_c / h_f
    if np.any(ratio <= 1) or not np.allclose(ratio, np.round(ratio), rtol=0, atol=1e-10):
        raise ValueError("Fine grid must strictly refine every axis by an integer factor")
    if coarse["grid"]["quadrature"] != fine["grid"]["quadrature"]:
        raise ValueError("Mismatched quadrature conventions")
    for key in ("sha256", "global_phase"):
        if coarse["analytic_reference"][key] != fine["analytic_reference"][key]:
            raise ValueError(f"Mismatched analytic reference {key}")
    _hashes([fine["analytic_reference"]], "analytic reference", 1)
    if fine["analytic_reference"]["global_phase"] not in (-1, 1):
        raise ValueError("Invalid global phase")
    oc, of = (r["analysis"]["origins"][origin] for r in (coarse, fine))
    if not np.array_equal(_vector(oc["origin_bohr"], "origin"), _vector(of["origin_bohr"], "origin")):
        raise ValueError("Mismatched operator origins")
    errors = {"coarse": {}, "fine": {}}
    for kind in ("electric", "magnetic"):
        ref_c = _vector(oc[f"analytic_{kind}_vector"], "analytic vector")
        ref_f = _vector(of[f"analytic_{kind}_vector"], "analytic vector")
        norm = float(np.linalg.norm(ref_f))
        if norm <= 0 or not np.array_equal(ref_c, ref_f):
            raise ValueError("Analytic reference vectors must be identical and nonzero")
        for label, row in (("coarse", oc), ("fine", of)):
            errors[label][f"{kind}_relative_vector_error"] = float(
                np.linalg.norm(_vector(row[f"{kind}_vector"], "grid vector") - ref_f) / norm)
    cr_c, cr_f = (_number(o["partitions"][partition]["C_R"], "C_R") for o in (oc, of))
    if not 0 <= cr_c <= 1 or not 0 <= cr_f <= 1:
        raise ValueError("C_R must be in [0, 1]")
    measured = dict(errors["fine"], C_R_absolute_difference=abs(cr_f - cr_c))
    checks = {key: measured[key] <= target for key, target in targets.items()}
    return {"schema": "chiroptmap-grid-precision-1", "state": fine["state"],
            "origin": origin, "origin_bohr": of["origin_bohr"], "partition": partition,
            "targets": targets, "measured": measured, "checks": checks,
            "coarse_errors": errors["coarse"], "C_R_coarse": cr_c, "C_R_fine": cr_f,
            "coarse_steps_bohr": h_c.tolist(), "fine_steps_bohr": h_f.tolist(),
            "precision_confirmed": all(checks.values()),
            "scope": "Explicit numerical targets for this grid pair only; not experimental uncertainty, box-size convergence, state assignment or a universal accuracy guarantee.",
            "execution_note": "This checks result contents; separately retain successful command return codes and electronic-structure termination evidence."}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coarse", type=Path, required=True)
    p.add_argument("--fine", type=Path, required=True)
    p.add_argument("--electric-rtol", type=float, required=True, help="Fraction, e.g. 0.005 means 0.5 percent")
    p.add_argument("--magnetic-rtol", type=float, required=True)
    p.add_argument("--cancellation-atol", type=float, required=True)
    p.add_argument("--origin", default="ElOri")
    p.add_argument("--partition", default="whole_molecule")
    p.add_argument("--result", type=Path, required=True)
    args = p.parse_args(argv)
    # Refuse overwrites even if validation subsequently fails.
    if args.result.exists() or args.result.with_name(args.result.name + ".partial").exists():
        raise FileExistsError(args.result)
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in (args.coarse, args.fine)]
    result = compare_grids(*reports, electric_rtol=args.electric_rtol, magnetic_rtol=args.magnetic_rtol,
                           cancellation_atol=args.cancellation_atol, origin=args.origin, partition=args.partition)
    result["results"] = [{"path": str(path.resolve()), "sha256": sha256(path)}
                         for path in (args.coarse, args.fine)]
    write_json_new(args.result, result)
    print(json.dumps({k: result[k] for k in ("precision_confirmed", "measured", "checks")}, indent=2))
    return 0 if result["precision_confirmed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
