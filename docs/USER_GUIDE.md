# ChiroptMap user guide

## 1. Overview

ChiroptMap connects an excited-state calculation to two complementary results:

1. a state table containing transition moments, `g_CPL`, excitation energy,
   wavelength, oscillator strength, and a radiative-rate estimate; and
2. real-space TEDM/TMDM cube files showing where the electric and magnetic
   transition-moment contributions are located.

The summary table is derived from the quantum-chemistry program's printed
transition moments. The maps are reconstructed from canonical orbitals and
excitation amplitudes. The two paths are deliberately cross-checked but are
not numerically substituted for one another.

## 2. Installation and environment check

```bash
python --version
python -m pip install .
chiroptmap --help
chiroptmap-gaussian-wfx --help
chiroptmap-orca --help
```

Only NumPy is required by ChiroptMap. Gaussian and ORCA are external licensed
programs and are never installed by this package.

## 3. ORCA workflow

### 3.1 Run ORCA

Use the template in `examples/orca_tda_chiroptmap.inp`. Keep `DoNTO false` so
that the canonical-orbital interpretation is not replaced by an NTO-oriented
output. The direct-map implementation currently requires `TDA true`.

### 3.2 Generate JSON and WFX

```bash
orca_2json job.gbw -json > job.orca_2json.log 2>&1
orca_2aim job > job.orca_2aim.log 2>&1
```

Confirm that `job.out`, `job.json`, and `job.wfx` describe the same calculation.
ChiroptMap validates geometry and basis consistency before computing a grid.

### 3.3 Create an S1 map

```bash
chiroptmap-orca \
  --output job.out \
  --json job.json \
  --wfx job.wfx \
  --state 1 \
  --outdir chiroptmap_S1 \
  --keep-components
```

The default grid spacing and molecular padding are 0.20 and 6.0 Bohr. Use
`--reference-cube` for a controlled comparison, not merely to change the box.

## 4. Gaussian direct-WFX workflow

### 4.1 Excited-state calculation

Print sufficiently small excitation amplitudes. The validated route uses
`IOp(9/40=4)`. For a density validation, request enough excited states and keep
the final output containing the transition electric and magnetic dipole tables.

### 4.2 Canonical all-MO WFX

At the exact same geometry, method, basis, charge, multiplicity, and orientation,
run a separate SCF job with:

```text
Output=WFX IOp(99/18=-1)
```

Write the WFX filename after the molecular specification as required by the
Gaussian input format. ChiroptMap rejects fractional occupations and common
post-SCF-natural-orbital signatures.

### 4.3 TDA maps

```bash
chiroptmap-gaussian-wfx \
  --log job_tda.log \
  --wfx job_canonical_allmo.wfx \
  --state 1 \
  --density-mode validation \
  --phase-align tedm \
  --strict-ci \
  --outdir chiroptmap_S1 \
  --keep-components
```

### 4.4 Full TD-DFT maps

When Gaussian output contains both `i -> a` and `i <- a` amplitudes, use:

```bash
--density-mode validation --response-convention xplus_yminus
```

The electric response then uses `X+Y`; the magnetic response uses `X-Y`.

## 5. Coefficient cutoff and diagnostics

Both direct commands default to `--min-abs-coeff 1e-4`.

- Gaussian filtering is limited by what was printed in the output. If Gaussian
  omitted an amplitude, ChiroptMap cannot recover it from the log.
- ORCA filtering starts from the JSON `Xia` matrix and is independent of table
  printing precision.
- The settings file records retained configuration counts and norms.

For Gaussian, `--strict-ci` requires `IOp(9/40=4)` or another validated
sufficient setting and checks the closed-shell diagnostic
`2*sum(c^2) >= 0.98`. This is a practical warning, not proof of cube
convergence. In the molecule1 scan, n=3 already gave `2*sum(c^2)=0.99760` but
still differed from n=4 by 29% (TEDM) and 41% (TMDM) in relative L2. The n=5
output was exactly the same as n=4 and did not add `1e-5` amplitudes.

## 6. CPL CSV without maps

To export only the summary table from either program:

```bash
chiroptmap --log job.out --program auto --cpl_only --outdir summary
```

To suppress CSV creation during a direct-map job, use `--no-cpl-csv`. To choose
the path explicitly, use `--cpl-csv-path path/to/table.csv`.

## 7. Output interpretation

`*_TEDM_total.cube` is the component-density vector projected onto the unit
electric transition-moment vector. `*_TMDM_total.cube` is projected onto the
unit magnetic transition-moment vector. Positive and negative isosurfaces show
signed contributions along the selected transition-moment direction.

The component files preserve x/y/z information. The validation CSV compares
each component integral with the vector printed by Gaussian or ORCA. Relative
errors for reference components close to zero should not be interpreted alone;
use vector or whole-cube metrics.

The CPL CSV reports `g value` for compatibility with CPLKit output. It is the
dimensionless `g_CPL`, not an electron-spin g tensor.

## 8. Reproducibility checklist

- Record program/version, method, basis, functional dispersion, and geometry.
- Keep the text output, WFX, JSON (ORCA), input, and ChiroptMap settings file.
- Use float64 for validation.
- Use the same grid for pointwise comparisons.
- Record `--min-abs-coeff` and the retained amplitude norm.
- Inspect normal termination before post-processing.
- Do not compare cube signs before global phase alignment.

## 9. Troubleshooting

`No configurations remain`
: Lower `--min-abs-coeff` or confirm that the requested state exists.

`occupied-MO primitive reconstruction error is too large`
: The ORCA JSON and WFX do not match, or the basis/export convention is outside
  the validated implementation.

`fractionally occupied` or `canonical MOs required`
: Supply a restricted canonical SCF WFX, not a post-SCF natural-orbital WFX.

`Could not find ... spectrum`
: Confirm that the final output contains length-gauge absorption and CD tables.

Large legacy TMDM differences
: The older MO-cube route uses finite-difference orbital gradients. Prefer the
  direct analytical-gradient workflow for quantitative validation.
