#!/usr/bin/env bash
set -eu

chiroptmap-gaussian-wfx \
  --log molecule_tda.log \
  --wfx canonical_allmo.wfx \
  --state 1 \
  --density-mode validation \
  --min-abs-coeff 1e-4 \
  --phase-align tedm \
  --strict-ci \
  --outdir results/gaussian_S1 \
  --keep-components
