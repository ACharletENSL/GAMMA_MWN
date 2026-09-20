# Numerical-convergence tests

The measurements behind `appendix_convergence.tex` (the numerical-convergence appendix of
*"Synchrotron emission from cooling internal shocks"*, Charlet, Granot & Beniamini). The
appendix quotes error on the **final observables** — light-curve peak, width, asymmetry,
spectral breaks and slopes, radiative efficiency — rather than on internal numerical
variables, so each test here ends at a number the paper actually uses.

The `.tex` itself lives with the article, not in this repository.

## Running them

Every script resolves the repository root from its own location, so no environment is
needed:

```bash
cd bin/Tools/project_v2/convergence_tests
python3 test1_resolution.py
```

`GAMMA_DIR` in the environment overrides that, for a second checkout. Results land in
`results/` beside the scripts.

**`test1b_*` refuse to run outside the cluster.** They compare the fiducial against the
hi-res caches, and the local `_fc2` point caches predate the leading-edge onset fix while
the HPC ones do not — reading both sides locally silently compares two code vintages.
Run them on `openuHPC`, or set `CONV_ALLOW_LOCAL=1` if you have checked the caches
yourself.

## The runs

The production run is **`cooling_g100_hires`** (10⁴ cells/shell); **`cooling_g100`**
(500 cells/shell, ε_B tuned separately) is the low-resolution twin. Both sit at
log₁₀𝒞 = −3.000 at α = 1, so they compare point by point.

Note `environment.FIDUCIAL_KEY = 'cooling_g100'` is commented *"the run the article is
built on"* — **that comment is stale**. It only selects the unsuffixed cache paths.

## The tests

| script | test | what it establishes |
|---|---|---|
| `test1_resolution.py` | 1 | 500 vs 10⁴ cells on the observables. Reads the CSV tables both runs left in `figures/{fiducial,hires}/`; no compute. |
| `test1b_eps_rad.py` | 1 | ε_rad across resolutions, from the cached flux-sweep points (`E_rad`/`E_inj` is stored in every one, so nothing is recomputed). **Cluster.** |
| `test1b_verify_link.py` | 1 | that the flux-sweep ε_rad *is* the dedicated fine grid's quantity, which is what licenses the line above. **Cluster.** |
| `test2_closure.py` | 2 | global energy closure (patches `_accum_energy` in-process; serial). |
| `test3_distrib.py` | 3 | the cooled electron distribution: stiff ODE on characteristics vs the analytic shape. |
| `test4_settling.py`, `test4b.py`, `test4c.py` | 4 | rarefaction settling tables, their propagation, and the production path. |
| `variants.py` | 5 + 6 | observer-time step cap and observer grid: full-shell runs against cached baselines. `VNPROC` sets workers. |
| `analyse_variants.py` | 5 + 6 | compares every `results/V_*.npz` against its cached production point. |
| `test7_filter.py` | 7 | the density filter: re-extracts cells with it off, and at window 25. |
| `test8_slopes.py` | 8 | free (non-circular) spectral slopes, per regime. **Reads the pooled table — see below.** |
| `test8b_mid_epoch.py` | 8 | the same free slopes split by **epoch**, which is the reading that holds. |
| `control.py` | — | cache reproducibility, after the test runs rebuilt some per-cell caches. |

## Results worth knowing

- **Resolution.** T_pk 0.3–1.4%, FWHM 0.4–1.1%, rise/fall 0.6–2.9% (medians). Regime
  classification **identical** at all nine points in both epochs; slopes ≤0.003.
  ν_pk 1–6%, up to 13% at log𝒞 = −4, −3 where the spectrum is flat at its maximum.
  ε_rad ≤0.44% (≤0.88% with the modelled cut-off).
- **Energy closure.** 4×10⁻⁴ per step, ≤0.12% per shell, in every regime.
- **Density filter.** No effect at all: the gate first fires at R/R_inj ≈ 208, against
  R_rar/R_inj ≈ 2.5 where emission ends.
- **The free mid-slope must be read epoch-split.** `free_slopes_by_regime.csv` pools
  rise/crossing/HLE and makes fast cooling look like a +0.016…+0.050 departure from 1/2.
  Split (`test8b`), the FC excursion tracks **break separation, not epoch** — +0.007,
  +0.014, +0.025, +0.031 as log𝒞 goes −5 → −2, and −0.0002 in VFC at log𝒞 = −5, where the
  breaks are widest and the plateau is genuine. That is a fixed-standoff window tilting on
  a shortening plateau: the prescription has no calibration of its own (at the production
  D = 10 it still leaves a_lo 0.005 short of an exact 4/3), so it **bounds** the FC
  departure rather than measuring one. Slow cooling is the real effect, softening
  monotonically through the pulse (rise −0.005…−0.036, crossing −0.010…−0.062,
  HLE −0.013…−0.073).

### Two derived facts

- The energy-closure identity the pipeline implies, verified on single steps. There is
  **no** factor ½ — an early derivation had a spurious one:

      4*pi*d_L^2/(1+z) * int int dF_nu dnu dT  =  delta * dE'_rad * (1 - tau_max^-2)

- **T_pk by bare `argmax` is not a convergence diagnostic.** Doubling the observer grid
  moves the maximum *sample* by up to 2.6% while moving the peak *flux* by 10⁻⁴. Use the
  production blended estimator, `lightcurve_shape.TOP_FRAC = 0.99`.

## Traps that cost real time

- A **spread-out `klist` breaks the sub-cell path** — it interpolates across neighbour
  gaps, giving NaNs and a 459-emitter blow-up from 60 cells. Use a contiguous block, or
  `subcell_dlogT=None`.
- **`norm=True` flux is not in physical units.** Use `norm=False` for any absolute energy
  work, or you are off by ν₀F₀/ν₀.
- One `get_shell_nuFnu_fromData` call carries **~100 s of fixed overhead** before it
  touches a single cell. A short test is not as short as it looks.
- `pkill -f <pattern>` matches the wrapper shell's own command line and will kill your own
  session. Use explicit PIDs.

## What is not here

The five `V_*.npz` variant outputs (~62 MB) are gitignored like every other `.npz` in this
tree; they are regenerated by `variants.py`. Run logs are not kept either. Everything in
`results/` is the small text/JSON/CSV evidence the appendix quotes directly.
