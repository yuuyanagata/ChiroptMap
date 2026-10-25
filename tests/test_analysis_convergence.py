"""Precision verdicts must fail closed on incompatible or incomplete inputs."""
from copy import deepcopy
import json

import pytest

from chiroptmap.analysis.convergence import compare_grids, main


def result(h, electric_error, cr):
    n = round(1/h)+1
    return {"schema": "chiroptmap-state-fields-1", "status": "complete", "state": 2,
            "engine": "orca", "response_convention": "tda", "min_abs_coefficient": 0.,
            "inputs": [{"sha256": c*64} for c in "abc"], "implementation": [{"sha256": "d"*64}],
            "analytic_reference": {"sha256": "e"*64, "global_phase": 1},
            "partitions": {"whole_molecule": {"labels": ["whole_molecule"], "mode": "whole_molecule"}},
            "atomic_numbers": [2], "coordinates_bohr": [[0, .7, 0]],
            "grid": {"points": n**3, "counts": [n]*3, "steps_bohr": [h]*3,
                     "lower_bohr": [0.,0.,0.], "upper_bohr": [1.,1.,1.],
                     "voxel_volume_bohr3": h**3, "quadrature": "equal-weight"},
            "analysis": {"is_complete_grid": True, "scientific_results_eligible": True,
                         "processed_points": n**3, "grid_points": n**3,
                         "NTO_rank_cutoff_in_physical_fields": None, "exact_weighted_virtual_contraction": True,
                         "origins": {"ElOri": {"origin_bohr": [.1,0.,0.],
                            "electric_vector": [1+electric_error,0.,0.], "magnetic_vector": [0.,0.,1.],
                            "analytic_electric_vector": [1.,0.,0.], "analytic_magnetic_vector": [0.,0.,1.],
                            "partitions": {"whole_molecule": {"C_R": cr}}}}}}


TARGETS = dict(electric_rtol=.005, magnetic_rtol=.0001, cancellation_atol=.001)


def test_verdict_uses_fine_vectors_and_explicit_targets():
    coarse, fine = result(.5, .14, .88414), result(.25, .0025, .88415)
    checked = compare_grids(coarse, fine, **TARGETS)
    assert checked["precision_confirmed"] is True
    assert checked["measured"]["electric_relative_vector_error"] == pytest.approx(.0025)
    assert checked["measured"]["C_R_absolute_difference"] == pytest.approx(.00001)
    fine["analysis"]["origins"]["ElOri"]["electric_vector"] = [1.014,0,0]
    # Falsified cached error metadata is deliberately ignored.
    fine["analysis"]["origins"]["ElOri"]["analytic_reference_errors"] = {"electric": {"relative_vector_error": 0}}
    checked = compare_grids(coarse, fine, **TARGETS)
    assert checked["precision_confirmed"] is False
    assert checked["checks"]["electric_relative_vector_error"] is False


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(state=1),
    lambda r: r.update(status="INCOMPLETE-PILOT"),
    lambda r: r["analysis"].update(scientific_results_eligible=False),
    lambda r: r["analysis"].update(processed_points=1),
    lambda r: r["analysis"].update(NTO_rank_cutoff_in_physical_fields=2),
    lambda r: r["inputs"][0].update(sha256="f"*64),
    lambda r: r["implementation"][0].update(sha256="f"*64),
    lambda r: r["partitions"].update(other={}),
    lambda r: r["analytic_reference"].update(sha256="f"*64),
    lambda r: r["analysis"]["origins"]["ElOri"].update(origin_bohr=[0,0,0]),
    lambda r: r["analysis"]["origins"]["ElOri"].update(analytic_electric_vector=[0,0,0]),
    lambda r: r["analysis"]["origins"]["ElOri"].update(electric_vector=[float("nan"),0,0]),
    lambda r: r["grid"].update(voxel_volume_bohr3=1.),
    lambda r: r["grid"].update(points=True),
    lambda r: r["analysis"]["origins"]["ElOri"]["partitions"]["whole_molecule"].update(C_R=None),
])
def test_incompatible_or_incomplete_results_rejected(mutate):
    coarse, fine = result(.5, .14, .88414), result(.25, .0025, .88415)
    mutate(fine)
    with pytest.raises(ValueError):
        compare_grids(coarse, fine, **TARGETS)


def test_same_box_nested_refinement_and_partition_file_hashes_required():
    coarse, fine = result(.5, .14, .88414), result(.25, .0025, .88415)
    for k in ("lower_bohr", "upper_bohr"):
        fine["grid"][k] = [x+.1 for x in fine["grid"][k]]
    with pytest.raises(ValueError, match="physical box"):
        compare_grids(coarse, fine, **TARGETS)
    with pytest.raises(ValueError, match="strictly refine"):
        compare_grids(coarse, deepcopy(coarse), **TARGETS)
    fine = result(.25, .0025, .88415)
    fine["partition_inputs"] = [{"sha256": "f"*64}]
    with pytest.raises(ValueError):
        compare_grids(coarse, fine, **TARGETS)
    with pytest.raises(ValueError):
        compare_grids(coarse, fine, **dict(TARGETS, electric_rtol=float("inf")))


def test_cli_writes_verdict_returns_failure_code_and_preserves_existing_output(tmp_path):
    coarse, fine = result(.5, .14, .88414), result(.25, .014, .88415)
    paths = [tmp_path/"coarse.json", tmp_path/"fine.json"]
    for path, report in zip(paths, (coarse, fine)):
        path.write_text(json.dumps(report))
    output = tmp_path/"precision.json"
    args = ["--coarse", str(paths[0]), "--fine", str(paths[1]), "--result", str(output),
            "--electric-rtol", ".005", "--magnetic-rtol", ".0001", "--cancellation-atol", ".001"]
    assert main(args) == 1
    assert json.loads(output.read_text())["precision_confirmed"] is False
    with pytest.raises(FileExistsError):
        main(args)
