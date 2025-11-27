# Analysis of Peak Frequency and Flux Discrepancies in Hybrid (Cartesian) Case

## Summary of Issue
The peak frequency and flux results differ between:
- `project_v1/thinshell_analysis.py` 
- `project_v2/run_analysis.py`

even after proper normalization in the 'hybrid' (cartesian) geometry case.

## Key Findings

### 1. Hybrid Correction Application

Both implementations apply the SAME geometrical corrections for cartesian geometry:

**project_v1** (`df_get_Fnu_thinshell`, lines 2085-2087):
```python
if hybrid and env.geometry == 'cartesian':
    nu_m *= (r*c_/env.R0)**d  # where d=-1, so: nu_m /= (r/R0)
    L *= (r*c_/env.R0)**a     # where a=1, so: L *= (r/R0)
```

**project_v2** (`get_Fnu_vFC`, lines 68-72):
```python
if env.geometry == 'cartesian':
    rfac = cell.x * c_ / env.R0
    num /= rfac  # nu_m /= (r/R0)
    F   *= rfac  # L *= (r/R0)
```

Both divide frequency by (r/R0) and multiply luminosity by (r/R0). ✓

### 2. Critical Difference: Normalization Factor

The KEY difference is in the luminosity normalization:

**project_v1** (`df_get_Fnu_thinshell`, line 2091):
```python
Fnu[tT>=1.] = env.zdl * derive_Lnu(nub, tT[tT>=1.], L, S, b1, b2)
```
- Multiplies by `env.zdl` directly
- NO division by L0

**project_v2** (`get_Fnu_vFC`, lines 75-76):
```python
if norm:
    F /= (env.L0 if (cell.trac < 1.5) else env.L0FS) / 3
```
- Divides by `(L0 or L0FS) / 3` when `norm=True`
- **This introduces a factor of 3 difference!**

### 3. The Factor of 3 Issue

From `shells_add_radNorm`:
```python
env.F0 = env.Fs/3.          # Line 198/221
env.Fs = env.zdl * env.L0    # Line 197/220
```

So: `F0 = (zdl * L0) / 3`

**In project_v2**, dividing by `(L0/3)` is equivalent to multiplying by `3/L0`:
```python
F /= (L0/3)  ⟹  F *= 3/L0
```

**In project_v1**, the result is later divided by `nu0F0 = nu0 * F0 = nu0 * zdl * L0 / 3` when plotting.

So the final normalized results should be:
- v1: `(nuobs * zdl * L) / (nu0 * zdl * L0 / 3) = (nuobs/nu0) * (3*L/L0)`
- v2: `(nuobs/nu0) * (3*L/L0)`

These appear mathematically equivalent! ✓

### 4. The REAL Discrepancy Source

The actual discrepancy likely stems from HOW `nu_m` and `Lth` are obtained:

**project_v1**:
1. `analysis_hydro_thinshell` calculates `nu_m2` and `Lth` using `get_variable()` at each iteration
2. These are STORED in CSV file `thinshell_hydro.csv`
3. `extract_contribs` READS these stored values and copies them to per-shock CSV files
4. `df_get_Fnu_thinshell` READS the stored values from CSV
5. Hybrid correction applied to STORED values

**project_v2**:
1. `get_Fnu_vFC` CALCULATES `nu_m2` and `Lth` ON-THE-FLY using `get_variable()` from cell data
2. Hybrid correction applied to FRESHLY CALCULATED values

### 5. Potential Issues

#### Issue A: Variable Name Mismatch
In `extract_contribs` (line 2293), the varlist includes `'nu_m'`:
```python
varlist = ['r', 'ish', 'v', 'rho', 'p', 'ShSt', 'Ton', 'Tej', 'Tth', 'lfac', 'nu_m', 'Lth', 'Lp']
```

But in `df_get_all_thinshell_v1` (line 2560), it's calculated as `'nu_m2'`:
```python
numList  = [0. if cell.empty else get_variable(cell, 'nu_m2') for cell in [RS, FS]]
```

**Question**: Is there a difference between `nu_m` and `nu_m2`?

Looking at the variable definitions, both appear to calculate the same thing (observer-frame peak frequency with Doppler factor), but this discrepancy in naming could indicate a problem.

#### Issue B: Stored vs Fresh Calculation
The stored values in v1 might have:
- Rounding errors from CSV I/O
- Been calculated from hydro variables at a slightly different cell position
- Missing some correction that should be applied

#### Issue C: Cell Selection Difference
**project_v1**: Uses `df_get_cellsBehindShock()` to select cells
**project_v2**: Uses `data.drop_duplicates(subset='i', keep='first')` to select contributing cells

These might select slightly different cells or handle the downstream region differently.

#### Issue D: The Hybrid Correction Timing
In v1, the `nu_m` stored in the CSV is ALREADY in observer frame (includes Doppler factor from the velocity at that time). When the hybrid correction is later applied (`nu_m /= r/R0`), this modifies the observer-frame frequency.

In v2, `nu_m2` is calculated fresh from the current cell's hydro variables, then immediately corrected.

If the stored values in v1 were from a slightly different cell position or velocity, this could cause discrepancies.

## Recommendations

1. **Check variable consistency**: Verify that `nu_m` stored in `thinshell_hydro.csv` is indeed the same as what `nu_m2` would calculate from the same cell's hydro variables.

2. **Compare intermediate values**: For a specific test case, print out:
   - The `nu_m` value BEFORE hybrid correction in both versions
   - The `Lth` value BEFORE hybrid correction in both versions  
   - The `r` value used for correction
   - The final peak values

3. **Check cell selection**: Verify both versions are analyzing the SAME cells at the SAME iterations.

4. **Test without hybrid correction**: Try running both analyses on a spherical case (no hybrid correction) to see if they agree. This isolates whether the issue is in the base calculation or the hybrid correction.

5. **Verify CSV accuracy**: Check if reading/writing to CSV in v1 introduces precision loss.

## Code Locations for Investigation

- **project_v1**:
  - Peak extraction: `get_pksnunFnu()` line 199
  - Flux calculation: `df_get_Fnu_thinshell()` line 2062
  - Hybrid correction: lines 2085-2087
  - Data extraction: `analysis_hydro_thinshell()` line 2387
  - Contribution extraction: `extract_contribs()` line 2283

- **project_v2**:
  - Peak extraction: `get_peaks_from_data()` line 154
  - Flux calculation: `get_Fnu_vFC()` line 55
  - Hybrid correction: lines 68-72
  - Variable calculation: Uses `get_variable()` directly

## Conclusion

The normalization schemes, when fully traced through, are mathematically equivalent. The discrepancy most likely arises from:
1. **Different cell selection methods**
2. **Stored vs on-the-fly calculation of nu_m and Lth**
3. **Potential mismatch between `nu_m` and `nu_m2` variables**
4. **Precision/rounding issues in CSV I/O**

I recommend adding diagnostic print statements to both codes to compare intermediate values at the same time steps.
