# Remaining Differences Between `get_pksnunFnu` and `get_peaks_from_data`

Even after fixing the normalization bug, there are **FIVE major differences** that cause different results:

## 1. **Different Frequency Variables**

### project_v1 (`thinshell_analysis.py`, line 2080):
```python
var_sh = ['Tej', 'Tth', 'r', 'nu_m', 'Lth']
```
- Uses `nu_m` which calls `derive_nu_from_gma` (variables.py line 71)
- Requires input: `['rho', 'vx', 'p', 'gmin']`
- **Uses actual `gmin` from the simulation**

### project_v2 (`radiation.py`, line 114):
```python
nub = nuobs/get_variable(cell, "nu_m2", env)
```
- Uses `nu_m2` which calls `derive_nu_m` (variables.py line 77)
- Requires input: `['rho', 'vx', 'p']` and microphysics parameters `['rhoscale', 'psyn', 'eps_B', 'eps_e', 'xi_e', 'z']`
- **Calculates `gmin` from microphysics assumptions**

**Impact:** If the simulation's `gmin` differs from the theoretical calculation (due to cooling, acceleration, or other effects), the frequencies will differ by the ratio `gmin_simulation / gmin_theory`.

---

## 2. **Different Frequency Sampling**

### project_v1 (`run_analysis.py`, lines 346-347):
```python
nuobs = np.concatenate((np.logspace(lognumin, -1, nnu_low), 
                         np.logspace(-1, 1.5, 300)))*env.nu0
```
- For `forpks=True`: `lognumin=-4`, `nnu_low=300`
- Total points: ~600 frequencies
- Higher density at low frequencies

### project_v2 (`radiation.py`, line 50):
```python
def get_radiation_vFC(key, front='RS', norm=True,
  Tmax=5, NT=450, lognu_min=-3, lognu_max=2, Nnu=500):
```
- Default: 500 evenly spaced (in log) frequencies
- Range: `lognu_min=-3` to `lognu_max=2`

**Impact:** Different frequency grids mean peaks might fall between sample points, leading to slightly different peak values.

---

## 3. **Different Flux Normalizations**

### project_v1 (`thinshell_analysis.py`, line 2013):
```python
nuFnus[i] += nuobs * df_get_Fnu_thinshell(row, sh, nuobs, Tobs, env, func, hybrid)
```
where `df_get_Fnu_thinshell` returns (line 2091):
```python
Fnu[tT>=1.] = env.zdl * derive_Lnu(nub, tT[tT>=1.], L, S, b1, b2)
```
- Final result: **`nuFnu` in CGS units** (erg/s/cm²)
- `env.zdl = (1+z)/(4π dL²)`

### project_v2 (`radiation.py`, lines 96-97):
```python
F = get_Fnu_vFC(cell, nuobs, Tobs, env, norm=True)
nF_step = (nuobs/nu0) * F
```
where `get_Fnu_vFC` with `norm=True` returns (line 117):
```python
F /= 3 * (env.L0 if (cell.trac < 1.5) else env.L0FS)
```
- Final result: **`nF` normalized by `nu0 * F0`** where `F0 = env.zdl * env.L0 / 3`
- Dimensionless quantity

**Impact:** The two methods return values in **completely different units**!
- project_v1: Physical flux in erg/s/cm²
- project_v2: Normalized flux (dimensionless)

---

## 4. **Different Data Sources and Preprocessing**

### project_v1:
- Reads from `thinshell_RS.csv` and `thinshell_FS.csv` 
- Created by `extract_contribs()` which:
  - May apply `extrapolate_early()` to fill in early times
  - May apply `smooth_contribs()` to reduce noise
  - Includes ALL contributions

### project_v2 (`radiation.py`, line 93):
```python
data = data.drop_duplicates(subset='i', keep='first')
```
- Reads from `run_data_{z}.csv`
- Uses only **first occurrence** of each cell
- No extrapolation or smoothing

**Impact:** Different preprocessing affects:
- Number of contributing cells
- Temporal resolution
- Smoothness of light curves

---

## 5. **Hybrid Correction (Geometry-Dependent)**

### project_v1 (`thinshell_analysis.py`, lines 2085-2087):
```python
if hybrid and env.geometry == 'cartesian':
    nu_m *= (r*c_/env.R0)**d  # d = -1
    L *= (r*c_/env.R0)**a      # a = 1
```
- Applies radius-dependent corrections for cartesian geometry
- `ν_m ∝ r^(-1)`, `L ∝ r^(1)`

### project_v2:
- **No hybrid correction visible in the flux calculation**

**Impact:** For cartesian geometry runs, project_v1 applies additional scaling factors that project_v2 doesn't.

---

## Summary Table

| Aspect | project_v1 | project_v2 |
|--------|-----------|------------|
| Frequency variable | `nu_m` (from simulation `gmin`) | `nu_m2` (calculated from theory) |
| Frequency sampling | ~600 points, variable density | 500 points, uniform in log |
| Flux units | CGS (erg/s/cm²) | Normalized (dimensionless) |
| Data source | `thinshell_RS.csv` (preprocessed) | `run_data_z.csv` (raw) |
| Cell selection | All contributions | First occurrence only |
| Preprocessing | Extrapolation + smoothing | None |
| Hybrid correction | Yes (for cartesian) | No |

---

## Why Different Peak Values?

The peak values `nF_pk` differ because:

1. **Different peak frequencies**: Using `nu_m` vs `nu_m2` shifts the spectrum
2. **Different sampling**: Peaks may fall between grid points
3. **Different normalizations**: Comparing apples (CGS) to oranges (normalized)
4. **Different data**: More/fewer contributions affect peak heights
5. **Hybrid corrections**: Additional scaling in project_v1 for certain geometries

---

## Recommendations

To reconcile the two methods:

1. **Choose one frequency variable**: Decide whether to use simulation `gmin` or theoretical `gmin`
2. **Use same frequency grid**: Standardize `Nnu`, `lognu_min`, `lognu_max`
3. **Standardize normalization**: Either both in CGS or both normalized
4. **Standardize preprocessing**: Decide on extrapolation/smoothing strategy
5. **Document hybrid corrections**: Make explicit when/why they're applied

The most critical issue is #3 - you're comparing values in different units! To properly compare:

```python
# Convert project_v2 to CGS:
nF_pk_CGS = nF_pk * (env.nu0 * env.F0)

# Or convert project_v1 to normalized:
nuFnu_pk_normalized = nuFnu_pk / (env.nu0 * env.F0)
```

where `env.F0 = env.zdl * env.L0 / 3` for RS, or `env.F0FS` for FS.
