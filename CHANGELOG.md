# Changelog

## Unreleased

## 0.1.1 - 2026-08-29

- Added the molecule1 `IOp(9/40=1-5)` direct-WFX convergence study.
- Clarified that n=5 was identical to n=4 in the validated Gaussian 16 case.
- Added an end-to-end synthetic test of the ORCA JSON/WFX shell-local
  AO-to-Cartesian-primitive transformation.
- Added GitHub Actions testing on Python 3.9, 3.10, 3.11, and 3.12.
- Documented the EDTM/TEDM and MDTM/TMDM terminology equivalence.
- Replaced contributor placeholders in release metadata with the manuscript
  author names.

## 0.1.0 - 2026-08-29

- Initial ChiroptMap release under the MIT License.
- Added checkpoint-free Gaussian canonical-WFX TEDM/TMDM maps.
- Added ORCA 6.1 closed-shell TDA JSON/WFX TEDM/TMDM maps.
- Preserved Gaussian and ORCA CPL summary CSV and `g_CPL` calculation.
- Set the direct-workflow amplitude cutoff default to `1e-4`.
- Added five-molecule Gaussian/ORCA validation and English documentation.
