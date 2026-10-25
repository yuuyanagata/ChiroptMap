"""Synthetic, closed-form checks of the public multi-origin analysis."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from chiroptmap.analysis import origins, state_fields as sf, stream
from chiroptmap.analysis import analytic
from chiroptmap.analysis.convergence import compare_grids
from test_analysis_stream import synthetic_wfx


def test_translation_is_pointwise_and_uses_cross_product_sign():
    points = np.array([[1., 2., 3.], [-3., 2., 1.]])
    tau = np.array([.2, -.4])
    current = np.array([[.5, -.7, .1], [.2, .8, -.4]])
    origin = np.array([.3, -.1, .6])
    e, b = points * tau[:, None], np.cross(points, current)
    shifted = sf.shifted_fields(e, b, tau, current, origin)
    np.testing.assert_allclose(shifted[0], (points - origin) * tau[:, None])
    np.testing.assert_allclose(shifted[1], np.cross(points - origin, current))
    total = sf.translated_moments(e.sum(0), b.sum(0), tau.sum(), current.sum(0), origin)
    np.testing.assert_allclose(total, np.asarray(shifted).sum(axis=1))
    np.testing.assert_allclose(origins.translate_vectors(e.sum(0), b.sum(0), tau.sum(), current.sum(0), origin), total)


def test_partitions_preserve_vectors_q_and_cancellation():
    rng = np.random.default_rng(184)
    e, b = rng.normal(size=(2, 41, 3))
    weights = rng.random((41, 3))
    weights /= weights.sum(axis=1, keepdims=True)
    ref = sf.reference_record([.2, .3, .4], [.1, -.5, .7], 0, [0, 0, 0], [0, 0, 0])
    region, whole = sf.Accumulator(["a", "b", "c"]), sf.Accumulator(["all"])
    # Different block boundaries must not change the additive quantities.
    for start in range(0, len(e), 7):
        region.update(e[start:start+7], b[start:start+7], ref, weights[start:start+7])
    whole.update(e, b, ref, np.ones((len(e), 1)))
    r, w = (a.report(.01, ref, e.sum(0)*.01, b.sum(0)*.01) for a in (region, whole))
    assert r["maximum_partition_unity_residual"] < 1e-15
    assert r["vector_additivity_maximum_absolute_residual"] < 1e-15
    assert r["q_additivity_absolute_residual"] < 1e-15
    assert r["C_b"] == pytest.approx(w["C_b"], abs=1e-15)
    assert r["C_R"] == pytest.approx(w["C_R"], abs=1e-15)
    assert sf.cancellation(0, 0) is None
    with pytest.raises(ValueError):
        sf.reference_record([0, 0, 0], [1, 0, 0], 0, [0, 0, 0], [0, 0, 0])


def test_common_centres_are_explicit_and_not_molecule_specific(tmp_path):
    wfx = replace(synthetic_wfx(), atomic_numbers=np.array([2]*4),
                  coordinates_bohr=np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]]))
    region_path, common_path = tmp_path/"regions.json", tmp_path/"common.json"
    region_path.write_text(json.dumps({"atom_index_base": 1, "regions": {"one": [1,2], "two": [3,4]}}))
    common_path.write_text(json.dumps({"centres_one_based": [1,3]}))
    definitions, _ = sf.make_partition_definitions(wfx, region_path, False, stream, common_path)
    common = definitions["common_centres_becke"]
    np.testing.assert_array_equal(common["centres"], wfx.coordinates_bohr[[0,2]])
    np.testing.assert_array_equal(common["mapping"], [0,1])
    assert common["labels"] == ["one", "two"]
    assert definitions["equal_becke_all_nuclei"]["centres_one_based"] == [1,2,3,4]
    for bad in ([], [0], [5], [1,1], [True], [1.0]):
        with pytest.raises(ValueError):
            sf.common_partition_definition(wfx, ["one", "two"], np.array([0,0,1,1]), bad)


def test_origin_parser_requires_transition_elori_not_generic_giao():
    with pytest.raises(ValueError):
        origins.parse_orca_transition_origin("Position of magnetic origin ... 0 0 0\nChoice of magnetic origin GIAO")
    metadata = origins.parse_orca_transition_origin(
        "Position of electric origin ... .1 0 .2\nAngular Momentum (ElOri)\nPosition of magnetic origin ... 5 6 7")
    assert metadata["transition_property_origin_bohr"] == [.1, 0, .2]


def test_analytic_cli_keeps_zero_vectors_and_compares_parent_at_elori(tmp_path, monkeypatch):
    from test_orca_json_bridge import ORCA_TRANSITION_TABLES
    import chiroptmap.orca_json as bridge_module
    wfx = synthetic_wfx()
    bridge = SimpleNamespace(wfx=wfx, diagnostics=SimpleNamespace(), roots=[
        SimpleNamespace(state=1, x_amplitudes=np.array([[.7]]), occupied_mos=[1], virtual_mos=[2])])
    monkeypatch.setattr(bridge_module, "read_orca_json_wavefunction", lambda *args: bridge)
    monkeypatch.setattr(analytic, "asdict", lambda diagnostics: {"synthetic": True})
    inp, wf, out = tmp_path/"input.json", tmp_path/"orbitals.wfx", tmp_path/"parent.out"
    inp.write_text("synthetic input")
    wf.write_text("synthetic orbitals")
    out.write_text(ORCA_TRANSITION_TABLES + "\nPosition of electric origin ... .2 -.1 .3\nAngular Momentum (ElOri)\n")
    result = tmp_path/"analytic.json"
    args = ["--json", str(inp), "--wfx", str(wf), "--output", str(out),
            "--state", "1", "--thresholds", "0", "--result", str(result)]
    assert analytic.main(args) is None
    data = json.loads(result.read_text())
    row = data["results"][0]
    e0, b0 = np.asarray(row["EDTM"]["integrated_vector"]), np.asarray(row["MDTM"]["integrated_vector"])
    origin = np.array([.2, -.1, .3])
    ep = e0-origin*row["integrated_charge_weighted_tau"]
    bp = b0-np.cross(origin, row["integrated_magnetic_current_proxy"])
    np.testing.assert_allclose(row["parent_validation_at_ElOri"]["electric"]["integrated_vector"], ep)
    np.testing.assert_allclose(row["parent_validation_at_ElOri"]["magnetic"]["integrated_vector"], bp)
    _, reference = sf.read_analytic_reference(result, 1, data["inputs"])
    np.testing.assert_allclose(reference[:2], [e0, b0])
    with pytest.raises(FileExistsError):
        analytic.main(args)


def test_full_grid_pilot_dry_run_and_precision_cli_on_synthetic_state(tmp_path, monkeypatch):
    wfx = synthetic_wfx()
    x = np.array([[.7]])
    ew, mw = -2*x, -x
    # Exact unnormalized s/px integrals; nonzero electric x and magnetic z.
    i0 = (np.pi/2)**1.5
    electric, magnetic = np.array([-.7*i0/2, 0, 0]), np.array([0, 0, -.49*i0])
    monkeypatch.setattr(stream, "_load_wavefunction", lambda args: (
        wfx, [1], [2], ew, mw, electric, magnetic, x, [1], [2], {"synthetic": True}))
    inp, wf, out = tmp_path/"input.json", tmp_path/"orbitals.wfx", tmp_path/"parent.out"
    inp.write_text("synthetic input")
    wf.write_text("synthetic orbitals")
    out.write_text("Position of electric origin ... .2 0 0\nAngular Momentum (ElOri)\n")
    common = ["--json", str(inp), "--wfx", str(wf), "--output", str(out), "--state", "3",
              "--bounds", "-5", "-5", "-5", "5", "5", "5", "--max-points", "513",
              "--shift-origin", ".2", "-.1", ".3"]
    coarse = sf.main(common + ["--spacing", ".5", "--result", str(tmp_path/"coarse.json")])
    ref_args = ["--analytic-reference", str(tmp_path/"coarse.analytic.json")]
    fine = sf.main(common + ref_args + ["--spacing", ".25", "--cube-dir", str(tmp_path/"maps"),
                                      "--result", str(tmp_path/"fine.json")])
    assert fine["status"] == "complete"
    assert fine["analysis"]["processed_points"] == 41**3
    assert fine["analysis"]["is_complete_grid"] is True
    assert fine["state"] == 3
    assert fine["analysis"]["NTO_rank_cutoff_in_physical_fields"] is None
    assert len(fine["analysis"]["map_export"]["files"]) == 5
    for origin, row in fine["analysis"]["origins"].items():
        assert row["origin_translation_maximum_absolute_residual"] < 2e-14
        assert row["analytic_reference_errors"]["electric"]["relative_vector_error"] < 1e-12
        assert row["analytic_reference_errors"]["magnetic"]["relative_vector_error"] < 1e-12
    checked = compare_grids(coarse, fine, electric_rtol=1e-10, magnetic_rtol=1e-10, cancellation_atol=.05)
    assert checked["precision_confirmed"] is True
    pilot = sf.main(common + ref_args + ["--spacing", ".5", "--pilot-points", "50", "--result", str(tmp_path/"pilot.json")])
    assert pilot["status"] == "INCOMPLETE-PILOT"
    assert pilot["analysis"]["scientific_results_eligible"] is False
    dry = sf.main(common + ["--dry-run", "--result", str(tmp_path/"dry.json")])
    assert dry["status"] == "dry-run" and "analysis" not in dry
    for incomplete in (pilot, dry):
        with pytest.raises(ValueError):
            compare_grids(incomplete, fine, electric_rtol=1, magnetic_rtol=1, cancellation_atol=1)
    with pytest.raises(FileExistsError):
        sf.main(common + ["--dry-run", "--result", str(tmp_path/"dry.json")])
    monkeypatch.setattr(sf, "main", lambda: {"status": "complete"})
    assert sf.cli() == 0


@pytest.mark.parametrize("spacing", [0., -1., float("nan"), float("inf")])
def test_invalid_spacing_rejected(spacing):
    with pytest.raises(ValueError):
        stream.snapped_grid(np.zeros((1, 3)), spacing, 1, .4)
