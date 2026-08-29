# ChiroptMap

[![Tests](https://github.com/yuuyanagata/ChiroptMap/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/yuuyanagata/ChiroptMap/actions/workflows/tests.yml?query=branch%3Amain)

ChiroptMap is an open-source Python toolkit for chiroptical excited-state
analysis. It generates real-space electric-dipole transition-moment (EDTM) and
magnetic-dipole transition-moment (MDTM) cube files from Gaussian or ORCA
calculations and exports state-resolved circularly polarized luminescence (CPL)
summary tables, including the dissymmetry factor `g_CPL`.

The direct workflows do not require Gaussian checkpoint files and do not create
intermediate molecular-orbital cube files.

## Terminology

ChiroptMap consistently uses the following names for its two scalar
transition-moment-density maps:

| Quantity | Label |
|---|---|
| Electric-dipole transition-moment density | EDTM |
| Magnetic-dipole transition-moment density | MDTM |

Direct-workflow releases through v0.1.1 used alternative word-order labels in
their interface and output files. Current commands, output files, runtime
messages, validation tables, and documentation use `EDTM` and `MDTM`
consistently.

## Highlights

- Gaussian TD-DFT/TDA output plus a canonical all-MO WFX file.
- ORCA 6.1 closed-shell TDA output plus JSON and occupied-MO WFX files.
- Direct analytical evaluation of atomic-orbital primitives and gradients.
- EDTM and MDTM total/component cube files.
- Automatic state-resolved CPL CSV export for both Gaussian and ORCA.
- A default excitation-amplitude cutoff of `1e-4`.
- Optional legacy Gaussian checkpoint/MO-cube workflow retained for comparison.
- NumPy-only runtime dependency.

## Supported workflows

| Program | Excited-state model | Required files | EDTM/MDTM cubes | CPL CSV |
|---|---|---|---:|---:|
| Gaussian | closed-shell TDA | TD/TDA output + canonical all-MO WFX | yes | yes |
| Gaussian | closed-shell TD-DFT | output with X/Y amplitudes + canonical all-MO WFX | yes | yes |
| ORCA 6.1 | closed-shell singlet TDA | output + JSON + WFX | yes | yes |
| Gaussian | printed CI + chk/fchk or existing MO cubes | output + orbital data | legacy | yes |

Open-shell ORCA, ORCA full TD-DFT X/Y reconstruction, and triplet maps are not
yet supported by the direct ORCA command.

## Coordinate frame and magnetic origin

MDTM evaluation uses the absolute Cartesian coordinates in the wavefunction
frame, with the magnetic angular-momentum operator referenced to `(0, 0, 0)`
of that frame. The lower corner of a cube is only the sampling-grid origin; it
does not redefine the magnetic-operator origin. Direct-workflow settings files
record both origins explicitly.

For cross-engine comparisons, preserve atom order, coordinates, orientation,
and origin. The supplied Gaussian templates use `NoSymm`, and an ORCA input can
be built from the matching Gaussian WFX coordinates. With `--reference-cube`,
ChiroptMap applies the same grid and rejects atom-list or coordinate mismatches
larger than `2e-5` Bohr.

## Installation

ChiroptMap requires Python 3.9 or later and NumPy.

```bash
git clone https://github.com/yuuyanagata/ChiroptMap.git
cd ChiroptMap
python -m pip install .
```

For development and tests:

```bash
python -m pip install -e ".[test]"
python -m pytest
```

The installed commands are:

```text
chiroptmap                 CPL CSV and legacy Gaussian MO-cube workflow
chiroptmap-gaussian-wfx    Direct Gaussian/WFX EDTM and MDTM cubes
chiroptmap-orca            Direct ORCA JSON/WFX EDTM and MDTM cubes
```

## Quick start: ORCA

Request JSON and WFX-compatible output in the ORCA input:

```text
! B3LYP/G 6-311G(2d,p) D3BJ TightSCF AIM

%pal
  nprocs 16
end

%tddft
  NRoots 6
  TDA true
  TPrint 1e-8
  DoNTO false
end

%output
  JSONGBWFile true
end

* xyzfile 0 1 molecule.xyz
```

After ORCA finishes, create the JSON and WFX files when they were not already
written by the job:

```bash
orca_2json molecule.gbw -json > orca_2json.log 2>&1
orca_2aim molecule > orca_2aim.log 2>&1
```

Generate S1 maps and the CPL CSV:

```bash
chiroptmap-orca \
  --output molecule.out \
  --json molecule.json \
  --wfx molecule.wfx \
  --state 1 \
  --min-abs-coeff 1e-4 \
  --outdir results/orca_S1 \
  --keep-components
```

ORCA's `orca_2aim` WFX export contains occupied orbitals only in the tested
ORCA 6.1.1 workflow. ChiroptMap obtains all canonical orbitals and exact TDA
`Xia` amplitudes from JSON, then uses the shared occupied orbitals to transform
the complete JSON MO set into the WFX Cartesian primitive representation.
Consequently, all three files (`.out`, `.json`, and `.wfx`) are required.

## Quick start: Gaussian without chk/fchk

Run the excited-state calculation with a sufficiently tight coefficient print
threshold. `IOp(9/40=4)` is the validated setting used by this project.

Create a separate matching ground-state WFX at the same geometry, method,
basis, and orientation. It must contain canonical virtual orbitals:

```text
#p B3LYP/6-311G(2d,p) Output=WFX IOp(99/18=-1) NoSymm
```

Use `NoSymm` in the matching excited-state input as well. This preserves the
input Cartesian frame for newly prepared Gaussian/ORCA comparisons; archived
validation calculations instead enforced the common frame by transferring the
Gaussian WFX coordinates directly to ORCA and checking the coordinates before
map generation.

Do not use the WFX written at the end of a TD/TDA job: it can contain post-SCF
natural orbitals rather than the canonical orbitals referenced by printed
`occupied -> virtual` amplitudes.

Generate the maps:

```bash
chiroptmap-gaussian-wfx \
  --log molecule_tda.log \
  --wfx molecule_canonical_allmo.wfx \
  --state 1 \
  --density-mode validation \
  --min-abs-coeff 1e-4 \
  --phase-align edtm \
  --strict-ci \
  --outdir results/gaussian_S1 \
  --keep-components
```

For full TD-DFT response-density validation, print both forward and reverse
amplitudes and add `--response-convention xplus_yminus`.

## The `1e-4` coefficient threshold

`IOp(9/40=4)` and `--min-abs-coeff 1e-4` have the same intended numerical
resolution, but they act at different stages:

- Gaussian `IOp(9/40=4)` controls which amplitudes are printed to the output.
- ChiroptMap filters the amplitudes available in its input after parsing.
- ORCA JSON contains the full TDA matrix; ChiroptMap applies `|Xia| >= 1e-4`
  itself. ORCA `TPrint` only affects human-readable output and is not the map
  cutoff.

Therefore these settings are equivalent for the retained TDA amplitude set,
not identical program instructions. A cutoff-convergence test is recommended
for unusually diffuse or strongly multiconfigurational states.

An explicit molecule1 scan over `IOp(9/40=1,2,3,4,5)` found 2, 16, 654,
9556, and 9556 printed S1 amplitudes, respectively. The n=4 and n=5 amplitude
sets and direct cubes were identical; n=5 did not extend the output to `1e-5`.
See [docs/IOP_9_40_CONVERGENCE.md](docs/IOP_9_40_CONVERGENCE.md).

## CPL CSV and `g_CPL`

Every direct map command writes `<output-stem>-CPL.csv` unless
`--no-cpl-csv` is supplied. The legacy-compatible `g value` column contains

```text
g_CPL = 4 (mu dot m) / (|mu|^2 + |m|^2)
```

Gaussian values are calculated from its printed electric and magnetic
transition-moment vectors after the established scaled-CGS conversion. For
ORCA, ChiroptMap uses the separately printed length-gauge rotatory strength
`R` in the numerator and applies ORCA's factor-of-two magnetic-vector
convention. This avoids avoidable loss of precision from rounded vector
components.

The CSV also contains vector components, vector norms, `cos(theta)`, excitation
energy, wavelength, oscillator strength, and a radiative-rate estimate.

## Principal outputs

```text
*_EDTM_total.cube
*_MDTM_total.cube
*_EDTM_[xyz].cube                  with --keep-components
*_MDTM_[xyz].cube                  with --keep-components
*_moment_validation.csv
*-CPL.csv                          includes g value
*_settings.txt
```

The older Gaussian MO-cube command and both direct commands now use the same
`EDTM`/`MDTM` filename convention.

## Validation

Five molecules were calculated independently with Gaussian and ORCA at
B3LYP/6-311G(2d,p)-D3BJ/TDA using identical geometries and grids. For the S1
maps, ORCA direct evaluation versus Gaussian direct WFX evaluation gave:

- EDTM cosine similarity: 0.999895 to 0.999978.
- EDTM relative L2 error: 0.677% to 1.466% (mean 1.061%).
- MDTM cosine similarity: 0.999753 to 0.999906.
- MDTM relative L2 error: 1.431% to 2.314% (mean 1.774%).
- Maximum S1 energy difference: 0.000277 eV.

See [docs/VALIDATION.md](docs/VALIDATION.md) for the full table and
interpretation.

## Documentation

- [English user guide](docs/USER_GUIDE.md)
- [ORCA conventions and input requirements](docs/ORCA_CPL.md)
- [Validation results](docs/VALIDATION.md)
- [Gaussian IOp(9/40) convergence](docs/IOP_9_40_CONVERGENCE.md)
- [Density modes](docs/VALIDATION_AND_CONTRIBUTION_MAP_MODES.md)
- [Implementation notes](docs/IMPLEMENTATION_NOTES.md)

## License

ChiroptMap is released under the [MIT License](LICENSE). Gaussian, ORCA, their
utilities, and user-generated quantum-chemistry data are not distributed with
or licensed by this repository.
