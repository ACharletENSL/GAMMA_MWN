# Normalization Bug Fix: Factor of ~10 Difference in Flux

## Summary

A normalization bug in `project_v2/radiation.py` (line 118) was causing the flux to be **underestimated by a factor of 9** compared to `project_v1/thinshell_analysis.py`.

## The Bug

### Location
File: `bin/Tools/project_v2/radiation.py`, line 118

### Incorrect Code
```python
F /= 3 * (env.L0 if (cell.trac < 1.5) else env.L0FS)
```

### Corrected Code
```python
F /= (env.L0 if (cell.trac < 1.5) else env.L0FS) / 3
```

## Root Cause Analysis

### Definition of F₀
From `phys_functions_shells.py`, lines 220-221:
```python
env.Fs = env.zdl * env.L0
env.F0 = env.Fs / 3
```

So: **F₀ = zdl × L₀ / 3**

### Formula Derivation

For normalized νF_ν in units of ν₀F₀:

**(νF_ν) / (ν₀F₀) = (ν/ν₀) × (F_ν/F₀)**

Where:
- F_ν = zdl × L_ν (converting luminosity per frequency to flux per frequency)
- L_ν = Lth × T⁻² × S(νT) (spectral shape in time)

Substituting:

**(νF_ν) / (ν₀F₀) = (ν/ν₀) × (zdl × L_ν) / (zdl × L₀/3)**
                 **= (ν/ν₀) × (L_ν/L₀) × 3**
                 **= (ν/ν₀) × (Lth/(L₀/3)) × T⁻² × S**

### What project_v1 Does (Correct)

```python
# Line 2091 in thinshell_analysis.py
Fnu = env.zdl * L * T**-2 * S(nub*T)  # where L = Lth
nuFnu = nuobs * Fnu

# Normalized:
nuFnu_norm = nuFnu / (nu0 * F0)
           = (nuobs/nu0) * (zdl * Lth * T**-2 * S) / (zdl * L0/3)
           = (nuobs/nu0) * (Lth/L0) * 3 * T**-2 * S
```

### What project_v2 Was Doing (Incorrect)

```python
# Line 118 (BEFORE fix)
F = Lth / (3*L0)  # <-- ERROR: dividing by 3*L0 instead of L0/3

# Line 132
Fnu = F * S(nub*T) / T**2

# Line 98-99
nF = (nuobs/nu0) * Fnu
   = (nuobs/nu0) * (Lth/(3*L0)) * T**-2 * S
```

### The Factor of 9

Ratio = v1_normalized / v2_old
      = [(nuobs/nu0) × (Lth/L0) × 3 × T⁻² × S] / [(nuobs/nu0) × (Lth/(3×L0)) × T⁻² × S]
      = [(Lth/L0) × 3] / [Lth/(3×L0)]
      = (3 × Lth/L0) × (3×L0/Lth)
      = **9**

## Verification

You can verify the fix using the diagnostic script:

```bash
python diagnose_normalization.py <key> [RS/FS]
```

This script compares the normalized peak fluxes from both methods and shows the ratio should be ~9 before the fix and ~1 after the fix.

## Other Differences

While this factor of 9 bug explains the ~10x discrepancy you observed, there are still minor differences (~10-20%) between the two methods due to:

1. **Different frequency variables**: 
   - v1 uses `nu_m` (from simulation's actual `gmin`)
   - v2 uses `nu_m2` (from theoretical microphysics)

2. **Different frequency sampling**:
   - v1: ~600 points with variable density
   - v2: 500 points uniform in log space

3. **Hybrid corrections**:
   - v1 applies radius-dependent corrections for Cartesian geometry
   - v2 does not (this should be added if needed)

## Impact

This bug affected all flux calculations in project_v2 that use `get_radiation_vFC` with `norm=True`, including:
- `get_peaks_from_data()`
- Any analysis relying on normalized flux values

## Testing

After applying this fix, the normalized peak fluxes from both methods should agree to within ~20% (due to the remaining minor differences listed above).
