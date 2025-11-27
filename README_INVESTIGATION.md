# Investigation of Hybrid Analysis Discrepancies

## Overview

This investigation analyzes why peak frequency and flux results differ between `project_v1/thinshell_analysis.py` and `project_v2/run_analysis.py` for the 'hybrid' (cartesian) geometry case.

## Files Created

### 1. Documentation Files

#### `/workspace/ANALYSIS_SUMMARY.md` - **START HERE**
High-level summary with:
- Quick explanation of the likely causes
- Comparison of the two implementations
- Action items and next steps
- Quick reference guide

#### `/workspace/hybrid_analysis_discrepancy_findings.md`
Detailed technical analysis including:
- Line-by-line code comparison
- Mathematical equivalence proofs
- Specific code locations
- Potential issues identified

### 2. Diagnostic Tools

#### `/workspace/bin/Tools/quick_check_hybrid.py` - **Run This First**
Quick verification script that compares peak values between v1 and v2.

**Usage:**
```bash
cd /workspace/bin/Tools

# Check default run (HPC_cart, RS)
python quick_check_hybrid.py

# Check specific run and shock
python quick_check_hybrid.py --key plan_fid --shock FS
```

**Output:**
- Loads peak values from both versions
- Shows peak frequency and flux
- Calculates percentage difference
- Gives verdict: agree, investigate, or significant discrepancy

**Typical output:**
```
Quick Check: Comparing HPC_cart, shock RS
======================================================================

✓ V1 loaded successfully
  Geometry: cartesian
  Peak frequency (normalized): 1.234567e-01
  Peak flux (normalized):      2.345678e-02

✓ V2 loaded successfully
  Geometry: cartesian
  Peak frequency (normalized): 1.234500e-01
  Peak flux (normalized):      2.345600e-02

======================================================================
COMPARISON:
======================================================================

Peak Frequency:
  V1: 1.234567e-01
  V2: 1.234500e-01
  Difference: +0.05%

Peak Flux:
  V1: 2.345678e-02
  V2: 2.345600e-02
  Difference: +0.33%

======================================================================
✓ VERDICT: Results agree within 1% - differences likely numerical
======================================================================
```

#### `/workspace/bin/Tools/diagnostic_hybrid_comparison.py` - **Run For Detailed Investigation**
Comprehensive diagnostic tool for deep investigation.

**Usage:**
```bash
cd /workspace/bin/Tools

# Run all tests
python diagnostic_hybrid_comparison.py HPC_cart

# Run specific test
python diagnostic_hybrid_comparison.py HPC_cart --test stored --samples 20
python diagnostic_hybrid_comparison.py HPC_cart --test corrections
python diagnostic_hybrid_comparison.py HPC_cart --test peaks

# Check forward shock instead
python diagnostic_hybrid_comparison.py HPC_cart --shock FS
```

**Tests performed:**

1. **Stored vs Calculated Variables** (`--test stored`)
   - Compares `nu_m` and `Lth` values stored in v1's CSV vs freshly calculated
   - Shows percentage differences for multiple sample points
   - Identifies if CSV storage is causing precision loss

2. **Hybrid Corrections** (`--test corrections`)
   - Shows how hybrid corrections are applied to raw values
   - Displays before/after values for frequency and luminosity
   - Verifies correction factors

3. **Peak Values** (`--test peaks`)
   - Compares final peak values from both versions
   - Shows first few values and peak-of-peak
   - Calculates mean, std, and max differences

**Typical output:**
```
================================================================================
TEST 1: Comparing stored vs calculated variables
================================================================================

Iter     r/R0       nu_m_stored     nu_m_calc       Δ%         Lth_stored      Lth_calc        Δ%        
--------------------------------------------------------------------------------------------------------------
100      1.0234     1.2345e+15      1.2346e+15     +0.008     5.6789e+42      5.6788e+42     -0.002    
200      1.0456     1.3456e+15      1.3456e+15     +0.000     5.7890e+42      5.7890e+42     +0.000    
...

==============================================================================================================
SUMMARY STATISTICS:
  nu_m differences: mean = 0.004%, std = 0.010%, max = 0.023%
  Lth differences:  mean = -0.001%, std = 0.008%, max = 0.015%
==============================================================================================================
```

## Workflow

### Step 1: Quick Check (30 seconds)
```bash
cd /workspace/bin/Tools
python quick_check_hybrid.py --key YOUR_KEY --shock RS
```

**If verdict says "Results agree within 1%":**
→ No significant discrepancy! Differences are likely numerical precision.

**If verdict says "investigate with diagnostic":**
→ Proceed to Step 2

**If verdict says "Significant discrepancy":**
→ Definitely proceed to Step 2

### Step 2: Detailed Diagnostic (2-3 minutes)
```bash
python diagnostic_hybrid_comparison.py YOUR_KEY --shock RS
```

This will:
1. Check if stored values match calculated values
2. Show how hybrid corrections are applied
3. Compare final peak arrays in detail

**Interpret the results:**
- If TEST 1 shows large differences → CSV storage issue
- If TEST 2 shows incorrect corrections → Geometry handling issue
- If TEST 3 shows large differences → Full pipeline issue

### Step 3: Manual Investigation (as needed)

Based on diagnostic results, you may need to:

**If stored vs calculated differ significantly:**
- Check variable name mismatch (nu_m vs nu_m2)
- Verify CSV write/read precision
- Compare cell selection methods

**If corrections look wrong:**
- Verify geometry is correctly set to 'cartesian'
- Check R0 normalization constant
- Verify r values are correct

**If peaks differ but intermediates match:**
- Check time array alignment
- Verify normalization constants (nu0, F0)
- Check spectral shape function

## Key Findings Summary

### What's The Same ✓
1. Hybrid corrections: Both divide frequency by (r/R0) and multiply luminosity by (r/R0)
2. Normalization: Both schemes are mathematically equivalent when fully traced
3. Spectral shape: Both use Band function with same parameters

### What's Different ⚠️
1. **Data handling**: v1 stores in CSV, v2 calculates on-the-fly
2. **Variable naming**: v1 uses 'nu_m', v2 uses 'nu_m2'
3. **Cell selection**: Different methods may select different cells
4. **Timing**: v1 stores at analysis time, v2 calculates when needed

### Most Likely Cause
Based on code analysis, the discrepancy most likely comes from:
1. Variable name mismatch in v1 ('nu_m' vs 'nu_m2')
2. CSV precision loss in v1
3. Different cells being selected by the two methods

## Quick Fix Suggestions

### If CSV storage is the issue:
```python
# In project_v1/thinshell_analysis.py, line 2316
# Increase CSV precision
df.to_csv(outpath+sh+'.csv', index_label='it', float_format='%.12e')
```

### If variable name is the issue:
```python
# In project_v1/thinshell_analysis.py, line 2293
# Change 'nu_m' to 'nu_m2' to be consistent
varlist = ['r', 'ish', 'v', 'rho', 'p', 'ShSt', 'Ton', 'Tej', 'Tth', 'lfac', 'nu_m2', 'Lth', 'Lp']
```

### If cell selection is the issue:
Need to investigate which cells each method selects and why they differ.

## Contact/Support

If the diagnostic tools don't resolve the issue:
1. Save the diagnostic output
2. Check the specific code locations mentioned in hybrid_analysis_discrepancy_findings.md
3. Add print statements to compare intermediate values
4. Consider running on a spherical case (no hybrid corrections) to isolate the issue

## Troubleshooting

**"ModuleNotFoundError" when running scripts:**
- Make sure you're in `/workspace/bin/Tools` directory
- Check that project_v1 and project_v2 directories exist
- Verify Python path includes the Tools directory

**"File not found" errors:**
- Ensure you've run `analysis_hydro_thinshell()` and `extract_contribs()` for v1
- Verify the run key name is correct
- Check that results directory exists

**"Both versions failed":**
- Verify data files exist for your run key
- Check GAMMA_dir path is correctly set
- Ensure phys_input.ini is accessible

---

**Last updated**: Based on analysis of codebase as of the investigation date
**Tools version**: 1.0
