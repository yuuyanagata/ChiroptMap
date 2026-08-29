from pathlib import Path


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _route_section(path: Path) -> str:
    return next(
        line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("#")
    )


def test_matching_gaussian_templates_preserve_cartesian_frame():
    excited_route = _route_section(EXAMPLES / "gaussian_tda_chiroptmap.gjf")
    canonical_route = _route_section(EXAMPLES / "gaussian_canonical_wfx.gjf")
    assert "NoSymm" in excited_route
    assert "NoSymm" in canonical_route
