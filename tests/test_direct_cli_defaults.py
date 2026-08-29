import math

from chiroptmap.orca_json_cubes import build_parser as build_orca_parser
from chiroptmap.wfx_cubes import build_parser as build_gaussian_wfx_parser


def test_orca_direct_defaults_match_validated_cutoff_and_write_cpl_csv():
    args = build_orca_parser().parse_args(
        ["--output", "job.out", "--json", "job.json", "--wfx", "job.wfx"]
    )
    assert math.isclose(args.min_abs_coeff, 1.0e-4)
    assert args.no_cpl_csv is False
    assert args.cpl_csv_path is None


def test_gaussian_wfx_defaults_match_validated_cutoff_and_write_cpl_csv():
    args = build_gaussian_wfx_parser().parse_args(
        ["--log", "job.log", "--wfx", "job.wfx", "--state", "1"]
    )
    assert math.isclose(args.min_abs_coeff, 1.0e-4)
    assert args.no_cpl_csv is False
    assert args.cpl_csv_path is None
    assert args.phase_align == "none"
    assert math.isclose(args.validation_edtm_factor, -2.0)
    assert math.isclose(args.validation_mdtm_factor, -1.0)


def test_gaussian_wfx_uses_edtm_mdtm_cli_names():
    args = build_gaussian_wfx_parser().parse_args(
        [
            "--log",
            "job.log",
            "--wfx",
            "job.wfx",
            "--state",
            "1",
            "--phase-align",
            "edtm",
            "--validation-edtm-factor",
            "-1.5",
            "--validation-mdtm-factor",
            "-0.75",
        ]
    )
    assert args.phase_align == "edtm"
    assert math.isclose(args.validation_edtm_factor, -1.5)
    assert math.isclose(args.validation_mdtm_factor, -0.75)
