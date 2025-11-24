# How to Fix the Peak Detection Discrepancies

## Problem Summary

The two methods `get_pksnunFnu` (project_v1) and `get_peaks_from_data` (project_v2) give different results due to:
1. **Different units** (CGS vs normalized)
2. **Normalization bug** in project_v2
3. **Different frequency variables** (nu_m vs nu_m2)
4. **Different data preprocessing**
5. **Geometry-dependent corrections** only in project_v1

## Immediate Fixes Required

### Fix 1: Correct the Normalization Bug in project_v2

**File**: `bin/Tools/project_v2/radiation.py`, line 67

**Current (INCORRECT)**:
```python
z, nu0, T0 = (1, env.nu0FS, env.T0FS) if (front == 'RS') else (4, env.nu0, env.T0)
```

**Fixed (CORRECT)**:
```python
z, nu0, T0 = (4, env.nu0, env.T0) if (front == 'RS') else (1, env.nu0FS, env.T0FS)
```

**Explanation**: The tracer value for RS is around 4, and for FS is around 1. The normalization frequencies should match the front being analyzed.

---

### Fix 2: Standardize Units for Comparison

Add a conversion function to make results comparable:

```python
def convert_to_same_units(nu_pk, nF_pk, env, front='RS', to_cgs=True):
    """
    Convert between normalized and CGS units
    
    Parameters:
    - nu_pk: peak frequencies (array)
    - nF_pk: peak fluxes (array)  
    - env: environment object
    - front: 'RS' or 'FS'
    - to_cgs: If True, convert normalized to CGS. If False, convert CGS to normalized
    
    Returns:
    - nu_pk, nF_pk in target units
    """
    if front == 'RS':
        nu0 = env.nu0
        F0 = env.zdl * env.L0 / 3
    else:
        nu0 = env.nu0FS
        F0 = env.zdl * env.L0FS / 3
    
    if to_cgs:
        # Convert normalized to CGS
        nu_pk_out = nu_pk * nu0
        nF_pk_out = nF_pk * (nu0 * F0)
    else:
        # Convert CGS to normalized
        nu_pk_out = nu_pk / nu0
        nF_pk_out = nF_pk / (nu0 * F0)
    
    return nu_pk_out, nF_pk_out
```

---

## Long-term Recommendations

### 1. Choose Frequency Variable Consistently

**Decision needed**: Use simulation `gmin` or theoretical `gmin`?

- **Simulation `gmin` (nu_m)**: More accurate if simulation properly tracks electron cooling
- **Theoretical `gmin` (nu_m2)**: More consistent with microphysics assumptions

**Recommendation**: Use `nu_m2` (theoretical) for consistency, but validate against simulation when available.

---

### 2. Standardize Frequency Grid

Create a common frequency grid function:

```python
def standard_frequency_grid(env, Nnu=500, lognu_min=-3, lognu_max=2):
    """Standard frequency grid for both methods"""
    nu_normalized = np.logspace(lognu_min, lognu_max, Nnu)
    nu_obs = nu_normalized * env.nu0
    return nu_obs, nu_normalized
```

---

### 3. Document Hybrid Corrections

The hybrid correction in project_v1 should be:
- Clearly documented
- Made optional with a flag
- Applied consistently in both methods if needed

```python
def apply_hybrid_correction(nu_m, L, r, env, apply=True):
    """
    Apply radius-dependent corrections for cartesian geometry
    """
    if apply and env.geometry == 'cartesian':
        factor_r = r * c_ / env.R0
        nu_m_corrected = nu_m * factor_r**(-1)
        L_corrected = L * factor_r**(1)
        return nu_m_corrected, L_corrected
    return nu_m, L
```

---

### 4. Standardize Data Processing

Create a unified data loader:

```python
def load_contributions(key, front='RS', use_preprocessed=True, 
                       extrapolate=False, smooth=False):
    """
    Unified data loading with consistent preprocessing
    
    Parameters:
    - key: run identifier
    - front: 'RS' or 'FS'
    - use_preprocessed: Use thinshell_*.csv vs run_data_*.csv
    - extrapolate: Apply early-time extrapolation
    - smooth: Apply smoothing
    """
    if use_preprocessed:
        path = get_contribspath(key)
        data = pd.read_csv(f"{path}{front}.csv")
    else:
        z = 4 if front == 'RS' else 1
        data = open_rundata(key, z)
        data = data.drop_duplicates(subset='i', keep='first')
    
    if extrapolate:
        data = extrapolate_early(data, env)
    
    if smooth:
        data = smooth_contribs(data, env)
    
    return data
```

---

## Comparison Script

Create a script to properly compare the two methods:

```python
def compare_peak_methods(key, front='RS'):
    """
    Compare peak detection methods with proper unit conversion
    """
    env = MyEnv(key)
    
    # Method 1 (project_v1)
    nupks_v1, nuFnupks_v1 = get_pksnunFnu(key, func='Band', force=False)
    nu_pk_v1 = nupks_v1[0 if front=='RS' else 1]
    nF_pk_v1 = nuFnupks_v1[0 if front=='RS' else 1]
    
    # Method 2 (project_v2)
    nu_pk_v2, nF_pk_v2 = get_peaks_from_data(key, front=front)
    
    # Convert v2 to CGS for comparison
    nu_pk_v2_cgs, nF_pk_v2_cgs = convert_to_same_units(
        nu_pk_v2, nF_pk_v2, env, front=front, to_cgs=True
    )
    
    # Compare
    nu_ratio = nu_pk_v2_cgs / nu_pk_v1
    nF_ratio = nF_pk_v2_cgs / nF_pk_v1
    
    print(f"Peak frequency ratio (v2/v1): mean={np.mean(nu_ratio):.3f}, std={np.std(nu_ratio):.3f}")
    print(f"Peak flux ratio (v2/v1): mean={np.mean(nF_ratio):.3f}, std={np.std(nF_ratio):.3f}")
    
    # Plot comparison
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    
    # Frequencies
    axs[0].loglog(nu_pk_v1, nu_pk_v2_cgs, 'o', alpha=0.5, label='Data')
    axs[0].loglog(nu_pk_v1, nu_pk_v1, 'k--', label='1:1 line')
    axs[0].set_xlabel('ν_pk from method 1 (Hz)')
    axs[0].set_ylabel('ν_pk from method 2 (Hz)')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # Fluxes  
    axs[1].loglog(nF_pk_v1, nF_pk_v2_cgs, 'o', alpha=0.5, label='Data')
    axs[1].loglog(nF_pk_v1, nF_pk_v1, 'k--', label='1:1 line')
    axs[1].set_xlabel('νF_ν,pk from method 1 (erg/s/cm²)')
    axs[1].set_ylabel('νF_ν,pk from method 2 (erg/s/cm²)')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'peak_comparison_{key}_{front}.png', dpi=150)
    
    return nu_ratio, nF_ratio
```

---

## Testing Checklist

After applying fixes:

- [ ] Fix normalization bug in project_v2/radiation.py line 67
- [ ] Add unit conversion function
- [ ] Run comparison script on test case
- [ ] Verify frequency ratios are closer to 1.0
- [ ] Verify flux ratios are closer to 1.0
- [ ] Document remaining differences (if any)
- [ ] Choose standard frequency grid
- [ ] Document when to use which method

---

## Expected Results After Fixes

After fixing the normalization bug and converting to same units:

1. **Peak frequencies** should be similar (within 10-20%) if using same frequency variable
2. **Peak fluxes** should be similar (within 10-30%) accounting for:
   - Different preprocessing (extrapolation/smoothing)
   - Different cell selection (all vs first occurrence)
   - Hybrid corrections (if geometry is cartesian)

**Remaining differences** will be due to legitimate methodological choices, not bugs.
