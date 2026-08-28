#!/usr/bin/env bash
set -eu

chiroptmap-orca \
  --output molecule.out \
  --json molecule.json \
  --wfx molecule.wfx \
  --state 1 \
  --min-abs-coeff 1e-4 \
  --outdir results/orca_S1 \
  --keep-components
