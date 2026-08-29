# ORCA input, CPL conventions, and direct cube generation

## Scope

The direct ORCA implementation is validated for ORCA 6.1.1, restricted
closed-shell singlet TDA calculations. It exports both state-resolved CPL
summary CSV files and S1/Sn EDTM/MDTM cube files.

## Recommended ORCA input

```text
! B3LYP/G 6-311G(2d,p) D3BJ TightSCF AIM

%pal
  nprocs 16
end

%maxcore 4000

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

The method and basis are examples. The keywords that matter to the current
interface are `AIM`, `TDA true`, `DoNTO false`, and `JSONGBWFile true`.

Generate exports, if necessary:

```bash
orca_2json molecule.gbw -json > molecule.orca_2json.log 2>&1
orca_2aim molecule > molecule.orca_2aim.log 2>&1
```

The expected file set is:

```text
molecule.out     ORCA text output with length-gauge absorption and CD tables
molecule.json    full canonical MO coefficients and singlet TDA Xia arrays
molecule.wfx     occupied canonical MOs in WFX primitive form
```

## Why JSON and WFX are both required

In the validated ORCA 6.1.1 calculation, `orca_2aim` exported only occupied
MOs to WFX. That file cannot represent an excitation by itself because virtual
orbitals are absent. ORCA JSON contains all canonical MO coefficients, but in
ORCA's spherical AO representation.

ChiroptMap matches the occupied orbitals shared by JSON and WFX, determines a
full-rank shell-local AO-to-Cartesian-primitive transform, and applies it to
all JSON orbitals. The code rejects inconsistent atom lists, geometries,
occupations, shell dimensions, primitive exponents, or reconstruction errors
larger than `1e-7`.

## Direct-map command

```bash
chiroptmap-orca \
  --output molecule.out \
  --json molecule.json \
  --wfx molecule.wfx \
  --state 1 \
  --min-abs-coeff 1e-4 \
  --spacing 0.20 \
  --padding 6.0 \
  --dtype float64 \
  --outdir results \
  --keep-components
```

Use `--reference-cube reference.cube` to reproduce an existing cube's origin,
axis vectors, and voxel counts exactly. This is intended for point-by-point
validation.

## TDA density convention

For a restricted closed-shell TDA root with ORCA amplitudes `Xia`, ChiroptMap
uses

```text
EDTM weight = -sqrt(2) * Xia
MDTM weight = -Xia / sqrt(2)
```

and evaluates both quantities directly from AO primitives. MDTM uses analytical
orbital gradients; no finite-difference MO cubes are involved.

## Amplitude cutoff

The default is `|Xia| >= 1e-4`. JSON amplitudes below the threshold are set to
zero before the weight matrices are constructed. The settings file records the
retained pair count and retained `sum(Xia^2)` so that cutoff convergence can be
checked. `TPrint` is not used as the numerical cutoff because JSON supplies the
unrounded amplitude matrix.

## CPL CSV convention

The output must contain the last instances of the following length-gauge tables:

```text
ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS
CD SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS
```

The electric vector is converted as

```text
mu_scaled = 254.17 * mu_ORCA(a.u.)
```

The tested ORCA output convention satisfies

```text
R(1e-40 cgs) = mu_scaled dot [2 * (-0.927401) * m_ORCA(a.u.)]
```

ChiroptMap therefore sets

```text
m_scaled = 2 * (-0.927401) * m_ORCA(a.u.)
g_CPL = 4 * R / (|mu_scaled|^2 + |m_scaled|^2)
```

The independently printed `R` is used in `g_CPL`; reconstructing it from the
rounded vector components would reduce precision. Velocity-gauge tables are
not mixed with length-gauge data.

## Current limitations

- Restricted closed-shell singlet TDA only for direct ORCA maps.
- The ORCA 6.1 JSON organization is expected; schema changes are detected as
  explicit errors but are not automatically translated.
- Full TD-DFT X/Y, unrestricted references, triplets, ECP-heavy cases, and
  specialized general-contraction cases require additional validation.
- A state with zero norm for the printed electric or magnetic transition-moment
  vector cannot define a projected total map, although component densities
  remain mathematically meaningful.
