# Examples

- `orca_tda_chiroptmap.inp`: ORCA 6.1 closed-shell TDA template that requests
  JSON and AIM/WFX-compatible output.
- `gaussian_tda_chiroptmap.gjf`: Gaussian TDA output template with
  `IOp(9/40=4)` and `NoSymm`.
- `gaussian_canonical_wfx.gjf`: matching Gaussian SCF all-canonical-MO WFX
  template with `NoSymm`. Replace the placeholder geometry and WFX filename.
- `run_orca_direct.sh`: direct ORCA JSON/WFX map command.
- `run_gaussian_wfx_direct.sh`: checkpoint-free Gaussian WFX map command.
- `run_validation_example.bat` and `run_contribution_map_example.bat`: legacy
  Gaussian MO-cube examples retained for comparison.

The quantum-chemistry input templates contain placeholder coordinates. Review
charge, multiplicity, method, basis, resources, and file names before use.
Keep `NoSymm` in both matching Gaussian jobs so their Cartesian frame is not
reoriented by symmetry handling.
