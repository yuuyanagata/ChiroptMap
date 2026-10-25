"""Explicit ORCA transition-property origin metadata and exact vector translation."""
from __future__ import annotations
import re
import numpy as np

FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"


def parse_orca_transition_origin(text, coordinates_bohr=None):
    """Use Angular Momentum (ElOri), not the generic magnetic GIAO setting."""
    lines = text.splitlines()
    electric_lines = [line.strip() for line in lines if "Position of electric origin" in line]
    magnetic_lines = [line.strip() for line in lines if "Position of magnetic origin" in line]
    angular_lines = [line.strip() for line in lines if re.search(r"Angular Momentum\s*\(ElOri\)", line)]
    choice_e = [line.strip() for line in lines if "Choice of electric origin" in line]
    choice_m = [line.strip() for line in lines if "Choice of magnetic origin" in line]
    if not electric_lines or not angular_lines:
        raise ValueError("Cannot establish ORCA transition angular-momentum origin: missing ElOri evidence")
    def values(line):
        tokens = re.findall(FLOAT, line.split("...")[-1])
        if len(tokens) != 3:
            raise ValueError(f"Unexpected origin position format: {line}")
        return [float(t.replace("D", "E").replace("d", "e")) for t in tokens]
    origin = values(electric_lines[-1])
    result = {
        "transition_property_origin": "ElOri (electric property origin)",
        "transition_property_origin_bohr": origin,
        "electric_origin_source_line": electric_lines[-1],
        "angular_momentum_source_lines": angular_lines,
        "electric_origin_choice_lines": choice_e,
        "generic_magnetic_origin_choice_lines": choice_m,
        "generic_magnetic_origin_position_lines": magnetic_lines,
        "generic_magnetic_origin_is_not_used_for_transition_comparison": True,
        "coordinate_precision_note": "Printed property origin is rounded; retain rounding uncertainty.",
    }
    # Corroborate position units with the explicit A.U. coordinate table and
    # printed isotopic/atomic masses, optionally using higher-precision WFX coordinates.
    matches = list(re.finditer(r"CARTESIAN COORDINATES \(A\.U\.\)", text))
    if matches:
        tail = text[matches[0].end():]
        row_pattern = re.compile(
            rf"^\s*\d+\s+[A-Za-z]+\s+{FLOAT}\s+\d+\s+({FLOAT})\s+({FLOAT})\s+({FLOAT})\s+({FLOAT})\s*$")
        rows, started = [], False
        for line in tail.splitlines():
            match = row_pattern.match(line)
            if match:
                rows.append([float(x.replace("D", "E")) for x in match.groups()])
                started = True
            elif started:
                break
        if rows:
            array = np.asarray(rows)
            coords = array[:, 1:]
            if coordinates_bohr is not None:
                if len(coordinates_bohr) != len(rows) or np.max(np.abs(coords - coordinates_bohr)) > 2e-5:
                    raise ValueError("A.U. output coordinates do not match the WFX atom order/frame")
                coords = np.asarray(coordinates_bohr)
            com = np.average(coords, axis=0, weights=array[:, 0])
            delta = float(np.max(np.abs(com - origin)))
            result["mass_center_from_AU_coordinates_bohr"] = com.tolist()
            result["mass_center_maximum_difference_from_printed_ElOri_bohr"] = delta
            result["ElOri_matches_AU_mass_center_within_2e6_bohr"] = delta < 2e-6
    return result


def translate_vectors(electric_zero, magnetic_zero, integrated_tau, integrated_current,
                      target_origin):
    origin = np.asarray(target_origin, float)
    return (
        np.asarray(electric_zero, float) - origin * float(integrated_tau),
        np.asarray(magnetic_zero, float) - np.cross(origin, np.asarray(integrated_current, float)),
    )


def rounding_metrics(residual, component_half_width):
    residual = np.asarray(residual, float)
    return {
        "assumed_component_rounding_half_width": component_half_width,
        "euclidean_rounding_bound": float(np.sqrt(3) * component_half_width),
        "all_components_inside_rounding_intervals": bool(np.all(np.abs(residual) <= component_half_width)),
        "residual_norm_inside_euclidean_rounding_bound": bool(
            np.linalg.norm(residual) <= np.sqrt(3) * component_half_width),
        "note": "Assumes nearest rounding of the displayed 5-decimal table; not a statistical confidence interval.",
    }
