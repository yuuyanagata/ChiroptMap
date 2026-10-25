# Changelog

## Unreleased

## 0.3.0 - 2026-10-25

- Added installable ORCA/TDA state-field analysis with full-amplitude electric
  and magnetic vector integrals, scalar cancellation indices C_b and C_R, and
  additive regional q contributions.
- Added independent all-space Cartesian-Gaussian reference integrals and exact
  operator-origin translations, including ORCA's transition-property ElOri.
- Added equal-radius Becke partitions with user-supplied atom regions and
  optional explicit common-centre subsets. Whole-molecule analysis needs no
  region definition.
- Added bounded-grid evaluation through exact weighted virtual-orbital
  contraction, separate TDA NTO orbital cubes, and streamed scalar map export.
  Physical fields are not truncated to the displayed NTO pairs.
- Added an explicit-target grid precision command that checks provenance,
  matching boxes/origins/partitions, complete processing, relative vector errors
  and the absolute C_R change. Completion alone does not certify precision.
- Added synthetic analytic, origin, additivity, NTO, map and precision-gate
  regression tests, plus installed-package command checks.
- Retained the v0.2.0 direct Gaussian/ORCA cube algorithms and CLI defaults.

## 0.2.0 - 2026-08-29

- Standardized direct Gaussian/ORCA output filenames, cube headers, validation
  labels, CLI help, runtime messages, tests, and documentation on EDTM/MDTM.
- Removed the former alternative word-order Gaussian direct-WFX CLI spellings.
- Recorded the cube-grid origin, magnetic-operator origin, and Cartesian frame
  in direct-workflow settings files.
- Added `NoSymm` to the Gaussian example inputs and documented the common-frame
  requirement for cross-engine MDTM comparisons.
- Added a GitHub Actions status badge to the README.

## 0.1.1 - 2026-08-29

- Added the molecule1 `IOp(9/40=1-5)` direct-WFX convergence study.
- Clarified that n=5 was identical to n=4 in the validated Gaussian 16 case.
- Added an end-to-end synthetic test of the ORCA JSON/WFX shell-local
  AO-to-Cartesian-primitive transformation.
- Added GitHub Actions testing on Python 3.9, 3.10, 3.11, and 3.12.
- Documented the equivalence of the earlier alternative word-order labels.
- Replaced contributor placeholders in release metadata with the manuscript
  author names.

## 0.1.0 - 2026-08-29

- Initial ChiroptMap release under the MIT License.
- Added checkpoint-free Gaussian canonical-WFX EDTM/MDTM maps.
- Added ORCA 6.1 closed-shell TDA JSON/WFX EDTM/MDTM maps.
- Preserved Gaussian and ORCA CPL summary CSV and `g_CPL` calculation.
- Set the direct-workflow amplitude cutoff default to `1e-4`.
- Added five-molecule Gaussian/ORCA validation and English documentation.
