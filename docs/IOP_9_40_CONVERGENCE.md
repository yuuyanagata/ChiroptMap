# Gaussian IOp(9/40) convergence for molecule1

## Protocol

The same molecule1 S1-optimized geometry and initial checkpoint were used for
five Gaussian 16 TDA calculations. The only changed input was
`IOp(9/40=n)`, with `n=1,2,3,4,5`.

```text
TDA=(NStates=6) B3LYP/6-311G(2d,p) EmpiricalDispersion=GD3BJ
Geom=AllCheck Guess=Read
```

All jobs used 16 CPU cores and terminated normally. S1 TEDM/TMDM cubes were
then evaluated directly from the same canonical all-MO WFX on the same cube
grid. The primary comparison used every amplitude printed at each IOp level.

## Printed S1 amplitudes

| n | Forward amplitudes | Minimum `|c|` | `2*sum(c^2)` | Norm fraction versus n=5 |
|---:|---:|---:|---:|---:|
| 1 | 2 | 0.10156 | 0.9823469344 | 0.9824083157 |
| 2 | 16 | 0.01001 | 0.9923237622 | 0.9923857669 |
| 3 | 654 | 0.00100 | 0.9976001180 | 0.9976624523 |
| 4 | 9556 | 0.00010 | 0.9999375196 | 1.0000000000 |
| 5 | 9556 | 0.00010 | 0.9999375196 | 1.0000000000 |

The n=4 and n=5 MO-pair sets and coefficient values were identical. In this
Gaussian 16 calculation, requesting n=5 did not print amplitudes down to
`1e-5`; it produced the same `1e-4` result as n=4.

## Direct-WFX cube convergence against n=5

| n | Map | Cosine | Norm ratio | Relative L2 | Shape-only L2 |
|---:|---|---:|---:|---:|---:|
| 1 | TEDM | 0.9286949464 | 1.38417312 | 0.58735515 | 0.37763753 |
| 1 | TMDM | 0.8110118992 | 1.10311577 | 0.65389936 | 0.61479769 |
| 2 | TEDM | 0.9366824638 | 1.22322934 | 0.45247660 | 0.35585822 |
| 2 | TMDM | 0.8276582405 | 1.04038518 | 0.60019546 | 0.58709754 |
| 3 | TEDM | 0.9681370016 | 1.11389846 | 0.28975360 | 0.25244009 |
| 3 | TMDM | 0.9148716123 | 0.95085777 | 0.40534545 | 0.41262183 |
| 4 | TEDM | 1.0000000000 | 1.00000000 | 0 | 0 |
| 4 | TMDM | 1.0000000000 | 1.00000000 | 0 | 0 |
| 5 | TEDM | 1.0000000000 | 1.00000000 | 0 | 0 |
| 5 | TMDM | 1.0000000000 | 1.00000000 | 0 | 0 |

The n=4 and n=5 numerical cube arrays were element-wise identical. Applying a
common ChiroptMap cutoff of `|c| >= 1e-4` also gave zero difference.

## Printed observables

All five calculations printed the same S1 energy (2.2482 eV), transition
electric and magnetic dipole vectors, and `g_CPL` (`0.011019054480426206`).
`IOp(9/40)` changes the amplitude list available for map reconstruction, not
the underlying excited-state solution or the transition-moment table.

## Interpretation

- n=1, n=2, and n=3 are unsuitable for quantitative TEDM/TMDM maps in this
  case.
- A coefficient-norm heuristic alone is insufficient. At n=3,
  `2*sum(c^2)=0.99760`, yet relative cube errors remained 29% for TEDM and 41%
  for TMDM.
- `IOp(9/40=4)` is the recommended Gaussian setting for ChiroptMap.
- n=5 provided no benefit over n=4 in this validated Gaussian 16 case.
- Other methods, basis sets, and molecules should retain n=4 as the default
  and use a direct cube-convergence check when unusually high accuracy is
  required.
