# Five-molecule Gaussian/ORCA validation

## Calculation protocol

Five molecules previously used for CPLKit validation were recalculated with
ORCA 6.1.1 on the IMS RCCS system. Gaussian and ORCA used the same S1-optimized
geometries and the same electronic-structure level:

```text
B3LYP/6-311G(2d,p)-D3BJ
closed-shell singlet TDA
six roots
```

All ORCA calculations used 16 MPI processes and terminated normally. Direct
ORCA maps used JSON amplitudes with `|Xia| >= 1e-4`, float64 arithmetic,
analytical primitive gradients, and the exact Gaussian reference grids.

## Excited-state quantities

| Molecule | ORCA S1 (eV) | Gaussian S1 (eV) | Difference (eV) | TEDM vector cosine | TMDM vector cosine |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.248318 | 2.248200 | +0.000118 | 0.99999995 | 0.99999660 |
| 2 | 2.488751 | 2.488700 | +0.000051 | 1.00000000 | 0.99999909 |
| 3 | 2.529948 | 2.530200 | -0.000252 | 0.99999997 | 0.99999367 |
| 4 | 2.787923 | 2.788200 | -0.000277 | 0.99999987 | 0.99999700 |
| 5 | 2.358859 | 2.359100 | -0.000241 | 0.99999999 | 0.99999657 |

The maximum energy difference was 0.000277 eV. ORCA magnetic vectors were
converted with the factor-of-two convention documented in `ORCA_CPL.md`.

## CPL CSV regression

The released `chiroptmap --cpl_only` path parsed all six roots for each of the
five ORCA outputs. The S1 rows were:

| Molecule | S1 energy (eV) | `g_CPL` | Oscillator strength |
|---:|---:|---:|---:|
| 1 | 2.248318 | +0.01096879793 | 0.075564451 |
| 2 | 2.488751 | -0.0008448459083 | 0.016058384 |
| 3 | 2.529948 | +0.004402885858 | 0.169140185 |
| 4 | 2.787923 | +0.003064702954 | 0.227961158 |
| 5 | 2.358859 | +0.006125259531 | 0.131865741 |

This regression confirms that the CPLKit-compatible `g value` CSV column is
preserved in ChiroptMap for ORCA calculations.

## Direct ORCA versus direct Gaussian/WFX cubes

Global excited-state phase was aligned before pointwise metrics were evaluated.

| Mol. | Map | Cosine | ORCA/Gaussian norm | Relative L2 |
|---:|---|---:|---:|---:|
| 1 | TEDM | 0.9999135603 | 0.99750129 | 0.0133675 |
| 1 | TMDM | 0.9997530851 | 1.00621004 | 0.0231400 |
| 2 | TEDM | 0.9998945953 | 0.99786003 | 0.0146608 |
| 2 | TMDM | 0.9998723429 | 1.00378727 | 0.0164507 |
| 3 | TEDM | 0.9999526786 | 0.99838643 | 0.0098536 |
| 3 | TMDM | 0.9998395046 | 1.00584282 | 0.0188946 |
| 4 | TEDM | 0.9999781687 | 0.99850746 | 0.0067694 |
| 4 | TMDM | 0.9999058241 | 1.00395495 | 0.0143087 |
| 5 | TEDM | 0.9999654609 | 0.99891674 | 0.0083772 |
| 5 | TMDM | 0.9998893096 | 1.00554638 | 0.0159176 |

Mean relative L2 errors were 1.061% for TEDM and 1.774% for TMDM. All cube
cosines exceeded 0.99975.

## Comparison with the legacy fchk/MO-cube workflow

TEDM remained close to the legacy results (0.677-1.466% relative L2). Legacy
TMDM differences were 11.7-32.0% relative L2, with the largest relative value
for molecule 2, whose magnetic moment is small. The same ORCA maps agreed with
Gaussian direct-WFX TMDM maps within 1.43-2.31%. This identifies finite-difference
MO-cube gradients in the legacy path, rather than a failure of the ORCA bridge,
as the principal source of the larger legacy TMDM difference.

## Bridge diagnostics

The maximum occupied-MO reconstruction error from the JSON spherical-AO basis
to the WFX Cartesian primitive representation was between `4.42e-9` and
`5.53e-9` for the five molecules. The implementation rejects errors above
`1e-7`.

## Validated conclusion

The results support an initial production scope of ORCA 6.1, restricted
closed-shell singlet TDA, using `.out + .json + .wfx`. They do not establish
accuracy for open-shell references, triplets, full TD-DFT X/Y response, or
unvalidated ORCA JSON versions.
