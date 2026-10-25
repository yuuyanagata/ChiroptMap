# State-field analysis and grid precision

The v0.3.0 commands post-process existing electronic-structure outputs. They do
not launch calculations, assign electronic-state symmetry or establish that an
optimized geometry is a minimum. The original direct cube commands remain
available with their existing defaults.

## Inputs and scope

`chiroptmap-state-fields` and `chiroptmap-analytic` support restricted singlet
ORCA TDA: the output, complete JSON X amplitudes and the matching occupied-MO
WFX export used by `chiroptmap-orca`. Preserve the atom order, coordinates and
Cartesian frame of those files. See [the user guide](USER_GUIDE.md)
for export preparation. Select a positive, one-based root with `--state`.
Neither command reconstructs ORCA full TD-DFT X/Y states, open-shell states or
triplets. Confirm calculation termination and state identity independently.

The public package contains the analysis methods. Calculation-specific input,
output and result files belong in the associated research dataset; installing
the package does not download them.

## Whole-molecule calculation

Use explicit physical bounds in bohr, in the order X0 Y0 Z0 X1 Y1 Z1. The
following numbers are an example box, which must be checked for the actual
molecule. Every extent must be an integer multiple of the requested spacing.
Use the same box for a spacing comparison.

```bash
chiroptmap-state-fields --json tda_sp.json --wfx tda_sp.wfx \
  --output tda_sp.out --state 2 --bounds -20 -19 -16 19 19 16 \
  --spacing 0.10 --result h010/result.json

chiroptmap-state-fields --json tda_sp.json --wfx tda_sp.wfx \
  --output tda_sp.out --state 2 --bounds -20 -19 -16 19 19 16 \
  --spacing 0.05 --analytic-reference h010/result.analytic.json \
  --result h005/result.json
```

The first command calculates and checkpoints all-space analytic reference
moments. The second reuses that reference only if its state and all three input
hashes match. Every exported X amplitude enters the physical fields; there is
no amplitude threshold or NTO-rank cutoff. Exact weighted virtual-orbital
contraction reduces orbital work without discarding contributions. Grid
blocks are bounded by `--max-points` (default 4096); the amplitude matrices,
primitive coefficients and NTO decomposition still occupy molecular-size
memory. Smaller spacing increases the number of evaluated points.

Without `--regions`, a single `whole_molecule` region has weight one. Adding
`--dry-run` validates inputs, the grid and NTO decomposition without integrating
fields. `--pilot-points N` integrates only a prefix of the grid for resource
diagnostics. An incomplete pilot is marked `INCOMPLETE-PILOT` and is excluded
from scientific result eligibility. Neither option is a convergence result.
Results and analytic checkpoints refuse existing output paths.

## Definitions, origins and units

The two vector fields are the inherited electric density rho_mu and real
reduced magnetic density rho_b. At operator origin O, the references are the
all-space integrals mu(O) and b(O). The scalar projections are

```text
s_b(r; O) = b_hat(O) dot rho_b(r; O)
s_R(r; O) = -mu_hat(O) dot rho_b(r; O)
q_density(r; O) = -mu(O) dot rho_b(r; O)
```

For each scalar field, let P be its positive integral and N its absolute
negative integral. The cancellation fraction is `1 - abs(P - N)/(P + N)`.
`C_b` uses s_b and `C_R` uses s_R. Zero absolute integral gives an undefined
value (`null`), not a zero-cancellation claim. A nonzero finite analytic
electric and magnetic reference is required to define the unit directions.

For a region with weight w_A, `q_region = integral(w_A * q_density)`. These
contributions sum to `q_grid` within floating-point accumulation error. They
depend on the partition, geometry and operator origin. They are not local
dissymmetry factors or separately measurable substituent observables.

Zero and ORCA ElOri are integrated on identical points in one traversal. ElOri
is read from the electric-property origin and **Angular Momentum (ElOri)**
evidence, not the generic magnetic/GIAO setting. The printed origin is rounded;
its rounding uncertainty remains. When an A.U. mass/coordinate table is present,
the parser checks its atom order/frame against WFX and records a mass-centre
corroboration. `--shift-origin X Y Z` adds any explicitly chosen operator origin
in bohr; `--axis-shift dx` is the x-only alternative.

```text
mu(O) = mu(0) - O * integral(tau)
b(O)  = b(0) - O cross integral(j_proxy)
```

The cube corner is the sampling origin, which does not change these operator
origins. Analytic projection axes are translated to the same origin as each
field. A common electric-aligned phase is used for both fields. Reference
validation against the printed parent is reported at ElOri.

The inherited ORCA weights are `-sqrt(2) X` for electric and `-X/sqrt(2)` for
reduced magnetic fields. The reduced magnetic comparison vector is twice the
printed ORCA CD-table vector. Magnetic physical constants, including mu_B, are
not inserted into regional q; this analysis does not independently calibrate
the magnetic unit convention. In particular C_R is not a dissymmetry factor,
and a physical g calculation must retain a material magnetic contribution to
its denominator. A large cancellation fraction or g alone does not establish
a bright or emitting state.

## Atom regions and common centres

An optional regions JSON must assign every physical atom exactly once, with
one-based integer indices and no empty regions. For a hypothetical four-atom
input:

```json
{"atom_index_base": 1, "regions": {"left": [1, 2], "right": [3, 4]}}
```

Passing `--regions regions.json` replaces the whole-molecule partition with
`equal_becke_all_nuclei`, the sum of equal-radius atomic Becke weights within
each region. Three polynomial iterations are used; there is no heteroatomic
radius adjustment. Add `--common-centres common.json` to also evaluate a
partition using a declared subset of nuclear centres:

```json
{"centres_one_based": [1, 3]}
```

The entire molecular density is retained, including density on atoms outside
that subset. Region labels without a selected centre are absent in the common
partition. Weights agree across molecules only when the selected coordinates
and sampling points agree. Displaced common centres can change the weights.
Both partitions report unity residuals and vector/q additivity residuals.
Choose indices for the actual atom order; there are no built-in molecule or
substituent assignments.

## Numerical precision check

```bash
chiroptmap-check-convergence --coarse h010/result.json --fine h005/result.json \
  --origin ElOri --partition whole_molecule \
  --electric-rtol 0.005 --magnetic-rtol 0.0001 --cancellation-atol 0.001 \
  --result precision.json
```

Targets must be supplied explicitly. Relative targets are fractions: `0.005`
means 0.5 percent. The checker recomputes electric and magnetic relative vector
errors against same-origin all-space analytic references on the **fine** grid,
and the absolute difference of the two C_R values. It returns exit code 0 when
all targets pass, 1 when a target fails; invalid/incompatible inputs fail without
writing a verdict. An existing result is never overwritten.

It requires matching state, three quantum-input hashes, implementation hashes,
analytic-reference hash/phase, partitions, geometries, physical box bounds,
operator origin and quadrature. The fine spacing must strictly refine every
axis by an integer factor. Both results must have status `complete`, full-grid
and eligibility flags, and matching positive point counts. Pilots, dry runs,
nonzero amplitude cutoffs and NTO-truncated fields are rejected. Paths may differ
after copying a dataset; hashes and numerical definitions establish identity.
For a regional comparison, provide the actual partition key.

`precision_confirmed` applies only to those targets and that grid pair. It is
not experimental uncertainty, state assignment or a universal accuracy claim.
Box convergence needs an additional box-size check. Keep successful external
command return codes and parent-calculation termination evidence alongside
results; this command cannot infer them from result JSON alone.

Weak transition moments can amplify a small quadrature residual through nearly
complete cancellation. Refining spacing can remedy undersampling of narrow
Gaussian terms even when the box is already large enough. An electric-vector
error percentage must not be transferred to C_R; compare C_R directly.

## Independent analytic diagnostic

```bash
chiroptmap-analytic --json tda_sp.json --wfx tda_sp.wfx --output tda_sp.out \
  --state 2 --thresholds 0 0.0001 --result analytic.json
```

This uses Cartesian-Gaussian primitive-pair algebra over all space, without the
spatial evaluator or finite box. Primitive normalization is already included in
the WFX coefficients. Zero-origin integrated vectors, tau/current integrals,
probe-MO overlap diagnostics and parent validation at ElOri are recorded. A
threshold-zero row can serve as the state-field analytic checkpoint; additional
threshold rows diagnose amplitude omission. Shared input parsers and conventions
mean this is an independent integration check, not an independent quantum method.

## Maps and NTOs

To save sampled s_b, s_R, q-density and leading NTO orbital cubes during
state-field integration, add `--cube-dir maps --map-stride 2`. These maps use
the common-zero operator origin. Map stride affects display samples, not the
full integration or cancellation values. Cubes are finalized only after their
declared number of values is written; unfinished files keep a `.partial` suffix.

For selected moment-projection and TDA NTO orbital cubes on a separate grid:

```bash
chiroptmap-maps --engine orca --json tda_sp.json --wfx tda_sp.wfx \
  --output tda_sp.out --state 2 --bounds -20 -19 -16 19 19 16 \
  --spacing 0.20 --nto-pairs 2 --outdir selected_maps
```

This exporter uses all retained source amplitudes (default threshold zero),
independently of the selected NTO pairs. NTO cubes have unit-norm MO mixing;
singular values/weights are reported separately. NTO coverage is not moment
coverage, and individual orbital signs are arbitrary.

The selected-map command also accepts Gaussian canonical all-MO WFX and printed
TDA amplitudes. With `--response-convention xplus_yminus --nto-pairs 0`, it can
export Gaussian full TD-DFT moment maps using the inherited X+Y/X-Y convention.
Gaussian threshold zero cannot restore configurations omitted from printed
output. This selected-map command uses fixed parent-output projection axes,
including when `--origin` shifts its operator origin; it is distinct from the
translated analytic axes of `chiroptmap-state-fields`. Its optional
`--analysis-result` expects a matching `chiroptmap.analysis.stream` result,
not the multi-origin state-field schema.
