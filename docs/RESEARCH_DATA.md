# Research data and manuscript reproduction

The manuscript *ChiroptMap: Orbital-Cube-Free Transition-Moment Density Analysis
with State-Resolved Cancellation Diagnostics*, by Yuuya Nagata and Takashi
Hirose, uses two distinct public records:

| Record | Identifier | Contents |
| --- | --- | --- |
| Documentation/citation release v0.3.1 | [10.5281/zenodo.24241780](https://doi.org/10.5281/zenodo.24241780) | Updated dataset/reproduction guidance and citation metadata; numerical algorithms unchanged from v0.3.0; MIT license |
| Installable software v0.3.0 | [10.5281/zenodo.23644576](https://doi.org/10.5281/zenodo.23644576) | ChiroptMap software snapshot; MIT license |
| Research dataset v1.0.0 | [10.5281/zenodo.23644802](https://doi.org/10.5281/zenodo.23644802) | Native quantum files, numerical results, parameters and frozen calculation implementations; CC BY 4.0 data with the original MIT licenses for code |

The manuscript's v0.3.0 software reference corresponds to the immutable
[`v0.3.0` tag](https://github.com/yuuyanagata/ChiroptMap/tree/v0.3.0), commit
`cd8e6eb016cf2d2a8fd1bb959b5d5b7f1487df2d`. Documentation added subsequently
to `main` does not change that release. The dataset preserves the production
implementations, including the inherited v0.2.0 direct-workflow core. Use those
frozen implementations for exact reproduction of the deposited calculations;
do not silently substitute the current package for them.

## Archive and integrity

The published `ChiroptMap_research_data_20261025.zip` contains 911 files and is
1,810,549,238 bytes. Its SHA-256 is:

```text
06770be0272ba44503e7ed7201e226a2f6631fb1ec57a4b95903ec023262f1d3
```

`MANIFEST.csv` records the path, size and SHA-256 of all 910 payload files;
the manifest itself is the remaining file. `DATA_DICTIONARY.txt` specifies
directories, units and interpretation. The archive includes:

- 27 additional quantum-calculation records, including ten exact-C2 scan
  points, four CAM controls, four Gaussian full-response endpoints, four
  unconstrained excited-state candidates, two ground-state geometries and
  three M3-M5 TDA benchmarks;
- four fixed-core composition/geometry controls in `fixed_core/`;
- 22 production field conditions in `fields/` and two additional M5/S2
  precision conditions in `precision/M5_S2/`;
- eight separate S1 reference fields, sixteen post-processing controls, and
  the five-molecule Gaussian/ORCA implementation benchmark;
- unrounded tables in `derived/`, atom-region definitions in `definitions/`,
  and numerical implementations in `code/`.

Quantum records and field conditions have different counts. Separable
one-dimensional quadrature diagnostics are not additional full-grid fields.
Regenerable visualization and intermediate orbital cubes are not duplicated.
Gaussian, ORCA and their licensed utilities are not distributed.

## Reproduction

From the extracted dataset root, with Python 3.9 or later and NumPy:

```bash
python code/recompute.py fields/M1_R1/S2/h010/parameters.json --validate-only
python code/recompute.py fields/M1_R1/S2/h010/parameters.json --output NEW_DIRECTORY
```

The output directory must not exist. `--maps` requests scalar map exports.
Matplotlib is required only for plotting. Full-grid runs may require
substantial computing resources. Parameter paths are relative to the dataset
root; preserve atom ordering, coordinate frames and the associated origins.

## Precision and interpretation

For the eight fixed-core responses at 0.1-bohr spacing, the maximum electric
and magnetic relative vector errors against same-origin all-space integrals
are 0.002285% and 0.041385%, respectively. Halving the spacing changes C_R by
at most 0.00034617. These are quadrature checks with shared electronic inputs.

For M5/S2, the selected complete 0.05-bohr grid at ElOri gives electric and
magnetic relative vector errors of 0.257315% and 0.00070618%, and C_R is
0.8841588934. Its difference from the 0.1-bohr C_R is 0.0000136402. These
satisfy the explicit 0.5%, 0.01% and 0.001 targets. The approximately 14%
electric-vector error belongs to the coarse 0.2-bohr grid and must not be
assigned to C_R or to the selected fine-grid result.

M3/S2 has an unresolved near-zero signed coupling on its single 0.2-bohr
grid: its ElOri grid and analytic couplings have opposite signs. A completed
field calculation or a large cancellation fraction alone does not prove a
converged residual sign. Native C1 state labels are not exact C2 irreducible
representations. Neither these field results nor electronic optimization
convergence establishes a frequency-certified minimum, an emitting-state
population or optical brightness.
