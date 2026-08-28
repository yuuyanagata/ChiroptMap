# ChiroptMap.py implementation notes

## Scope

ChiroptMap is a CPL-oriented Gaussian/ORCA TD-DFT/TDA post-processing package. It exports CPL summary CSV files from Gaussian or ORCA transition-moment tables. Its preferred workflows evaluate canonical orbitals and analytical gradients directly from Gaussian WFX data or from the ORCA JSON/WFX bridge, without checkpoint or intermediate MO cube files. A legacy Gaussian `cubegen` workflow is retained for comparison.

## Code provenance

The present software was written de novo in Python from published transition-density equations and Gaussian/ORCA output conventions. It is not a MATLAB port, wrapper, or line-by-line translation. No MATLAB source code is required, invoked, or distributed by this package.

## What is new in the present package

The novelty of this software is the integrated validated workflow rather than the known formula for gCPL. The package adds:

1. modular Gaussian and ORCA parsers for transition-moment tables, with Gaussian excited-state configuration parsing,
2. direct Gaussian canonical-WFX orbital and analytical-gradient evaluation,
3. an ORCA 6.1 JSON/WFX bridge that reconstructs all canonical orbitals from the occupied orbitals shared by both formats,
4. native Gaussian cubegen orchestration retained as a legacy comparison path,
5. explicit Bohr unit handling for Gaussian cube files,
6. coefficient-printing and amplitude-cutoff diagnostics,
7. separate coefficient-linear validation and signed square contribution-map modes,
8. full Gaussian TD-DFT response conventions based on X+Y and X-Y combinations,
9. component-level validation against Gaussian or ORCA transition-moment tables,
10. state-resolved CPL CSV output, including g_CPL, for both programs.

## Relation to general parsers and wavefunction-analysis programs

ChiroptMap is not intended to replace general tools such as cclib or Multiwfn. Those programs address broad parsing or wavefunction-analysis tasks. ChiroptMap addresses a narrower CPL-specific workflow: Gaussian/ORCA CPL summary generation, response-density reconstruction, and transition-moment validation handled consistently.

## Density-mode rationale

Contribution-map mode uses signed square weights, `2*c*abs(c)`, because the magnitude follows the closed-shell configuration contribution `2*c**2` while the sign follows the printed CI coefficient. This convention is useful for displaying signed spatial contributions from dominant orbital transitions. It is not a coefficient-linear transition-density amplitude.

Validation mode uses coefficient-linear amplitudes because Gaussian EDTM and MDTM vector components are coherent transition matrix elements. Real-space integration is expected to reproduce the Gaussian transition-moment tables only in validation mode.
