# Summary: Hybrid Analysis Result Discrepancies

## Your Question
Why don't you get similar results (once properly normalized) for peak frequency and flux in the 'hybrid' (cartesian) case between:
- `project_v1/thinshell_analysis.py`
- `project_v2/run_analysis.py`

## TL;DR - The Likely Cause

After detailed analysis, the normalization schemes are **mathematically equivalent**, so the discrepancy is most likely due to:

1. **Variable storage vs. on-the-fly calculation**: v1 stores `nu_m` and `Lth` in CSV files and reads them back, while v2 calculates them fresh from hydro variables
2. **Potential variable name confusion**: v1 extracts `'nu_m'` but calculates `'nu_m2'` 
3. **Different cell selection**: The two versions may select slightly different cells or handle the downstream region differently

## Detailed Analysis

### The Hybrid Corrections Are Identical ✓

Both implementations apply the SAME corrections for cartesian geometry:

```python
# Both do:
nu_m /= (r/R0)    # Frequency divided by radius ratio
L *= (r/R0)       # Luminosity multiplied by radius ratio
```

This is correct for converting from planar to spherical geometry.

### The Normalizations Are Equivalent ✓

While they LOOK different, the normalizations are mathematically equivalent:

**v1 approach:**
- Multiplies flux by `env.zdl` 
- Later divides by `env.nu0F0` when plotting
- Net effect: `(nuobs * zdl * L) / (nu0 * zdl * L0 / 3) = (nuobs/nu0) * (3*L/L0)`

**v2 approach:**
- Divides by `(env.L0 or env.L0FS) / 3`
- Multiplies by `nuobs/nu0` 
- Net effect: `(nuobs/nu0) * (L / (L0/3)) = (nuobs/nu0) * (3*L/L0)`

**These are identical!**

### The Real Issue: Data Handling

The discrepancy likely comes from HOW the data is processed:

| Aspect | project_v1 | project_v2 |
|--------|-----------|-----------|
| **Variable calculation** | Once, during `analysis_hydro_thinshell()` | On-the-fly in `get_Fnu_vFC()` |
| **Storage** | Stored in CSV files | Not stored |
| **Variable name** | Extracts `'nu_m'` but calculates as `'nu_m2'` | Always uses `'nu_m2'` |
| **Cell selection** | `df_get_cellsBehindShock()` | `drop_duplicates(subset='i', keep='first')` |
| **Potential issues** | - CSV precision loss<br>- Variable name mismatch<br>- Stored values may be stale | - Fresh calculation each time<br>- Different cells selected? |

### Specific Concerns

#### 1. Variable Name Discrepancy in v1
In `extract_contribs()` line 2293:
```python
varlist = ['r', 'ish', 'v', 'rho', 'p', 'ShSt', 'Ton', 'Tej', 'Tth', 'lfac', 'nu_m', 'Lth', 'Lp']
```

But in `df_get_all_thinshell_v1()` line 2560:
```python
numList = [0. if cell.empty else get_variable(cell, 'nu_m2') for cell in [RS, FS]]
```

**Is `'nu_m'` the same as `'nu_m2'`?** If not, this could be your problem!

#### 2. CSV Precision
v1 writes to CSV and reads back. This could introduce:
- Rounding errors
- Format conversion issues
- Loss of precision

#### 3. Cell Selection
Different methods for selecting which cells contribute could lead to different results if:
- The shock position is ambiguous
- Multiple cells have similar properties
- Edge cases are handled differently

## What To Do Next

### Step 1: Run the Diagnostic Script

I've created a diagnostic tool at `/workspace/bin/Tools/diagnostic_hybrid_comparison.py`.

Run it on your hybrid case:

```bash
cd /workspace/bin/Tools
python diagnostic_hybrid_comparison.py HPC_cart --shock RS
```

This will:
1. Compare stored vs. freshly calculated `nu_m` and `Lth` values
2. Show the hybrid corrections being applied
3. Compare the final peak values

### Step 2: Check for Variable Name Issues

Verify that `'nu_m'` and `'nu_m2'` are indeed the same by checking:
```python
# In your Python environment
from project_v1.df_funcs import var2func
print('nu_m' in var2func)
print('nu_m2' in var2func)
```

### Step 3: Test Without Hybrid Corrections

Run both analyses on a **spherical** case (no hybrid corrections) to see if they agree. This isolates whether the issue is in:
- The base calculation (if they differ even without corrections)
- The hybrid correction itself (if they only differ with corrections)

### Step 4: Add Debugging Output

Temporarily modify both scripts to print intermediate values:

**In v1's `df_get_Fnu_thinshell()` before line 2089:**
```python
if it == some_test_iteration:  # Pick a specific iteration
    print(f"V1: nu_m before = {nu_m:.6e}, L before = {L:.6e}")
    print(f"V1: r/R0 = {r*c_/env.R0:.6f}")
    if hybrid and env.geometry == 'cartesian':
        print(f"V1: nu_m after = {nu_m:.6e}, L after = {L:.6e}")
```

**In v2's `get_Fnu_vFC()` after line 72:**
```python
if cell.it == some_test_iteration:  # Same iteration
    print(f"V2: nu_m before = {num:.6e}, F before = {F:.6e}")
    print(f"V2: r/R0 = {rfac:.6f}")
    if env.geometry == 'cartesian':
        print(f"V2: nu_m after = {num:.6e}, F after = {F:.6e}")
```

Compare the outputs for the SAME iteration.

## Files Created for You

1. **`/workspace/hybrid_analysis_discrepancy_findings.md`**
   - Detailed technical analysis of the code differences
   - Line-by-line comparison of both implementations

2. **`/workspace/bin/Tools/diagnostic_hybrid_comparison.py`**
   - Automated diagnostic tool to compare the two methods
   - Can be run directly on your data

3. **`/workspace/ANALYSIS_SUMMARY.md`** (this file)
   - High-level summary and action items

## Quick Reference: Key Code Locations

### project_v1 (thinshell_analysis.py)
- **Peak extraction**: `get_pksnunFnu()` starting at line 199
- **Flux calculation**: `df_get_Fnu_thinshell()` starting at line 2062
- **Hybrid correction**: Lines 2085-2087
- **Data storage**: `analysis_hydro_thinshell()` at line 2387
- **Extraction**: `extract_contribs()` at line 2283

### project_v2 (run_analysis.py & radiation.py)
- **Peak extraction**: `get_peaks_from_data()` at line 154 of run_analysis.py
- **Flux calculation**: `get_Fnu_vFC()` at line 55 of radiation.py
- **Hybrid correction**: Lines 68-72 of radiation.py

## Expected Outcome

If the two implementations are working correctly, they should produce **identical results** (within numerical precision) after proper normalization. 

If they don't, the diagnostic script should help you identify:
- Which variable is causing the discrepancy (`nu_m`, `Lth`, or something else)
- At which stage the discrepancy occurs (storage, calculation, or correction)
- How large the discrepancy is (and whether it's significant)

---

**Need more help?** The diagnostic script output will give specific numbers. Share those numbers and I can help interpret them further.
