# Comparison: `get_pksnunFnu` vs `get_peaks_from_data`

## Overview
These two methods compute peak frequencies and fluxes from radiation data, but they differ in their data sources, processing approaches, and normalization schemes.

## Method 1: `get_pksnunFnu` (project_v1/thinshell_analysis.py, lines 199-228)

### Data Source:
- Uses CSV files from `get_contribspath(key)` → `results/{key}/thinshell_RS.csv` and `thinshell_FS.csv`
- These files are created by `extract_contribs()` function
- Data is preprocessed and may include extrapolation and smoothing

### Processing Flow:
1. Calls `nuFnu_thinshell_run()` (line 216) which:
   - Reads from CSV files (RS.csv, FS.csv)
   - Iterates through each row in the dataframe
   - Calls `df_get_Fnu_thinshell()` for each contribution
   - Accumulates contributions across all time steps

2. Finds peaks (lines 220-223):
   ```python
   for i in range(NT):
       ipk = np.argmax(nuFnu[i])
       nupks[k][i] = nuobs[ipk]
       nuFnupks[k][i] = nuFnu[i,ipk]
   ```

### Key Characteristics:
- Works with full accumulated flux from all contributions
- Uses preprocessed data that may have been extrapolated/smoothed
- Returns raw frequency and flux values (not explicitly normalized)

## Method 2: `get_peaks_from_data` (project_v2/radiation.py, lines 29-45)

### Data Source:
- Uses npz files via `get_radfile_thinshell()` → `results/{key}/thinshell_rad_{front}.npz`
- If file doesn't exist, generates from `open_rundata()` which reads CSV files: `run_data_{z}.csv`
- Data processing uses `drop_duplicates(subset='i', keep='first')` to keep only first time step per cell

### Processing Flow:
1. Calls `get_radiation_vFC()` (line 38) which:
   - Either loads from npz file or generates new data
   - Calls `run_nuFnu_vFC()` which processes raw run data
   - Uses `drop_duplicates(subset='i', keep='first')` approach

2. Finds peaks (lines 41-44):
   ```python
   for j in range(NT):
       i = np.argmax(nF[j])
       nu_pk[j] += nu[i]
       nF_pk[j] += nF[j][i]
   ```

### Key Characteristics:
- Works with normalized data (`norm=True`)
- Uses raw simulation data filtered by first occurrence
- Explicitly normalizes by `nu0` and `T0`

## CRITICAL BUG IDENTIFIED in `get_radiation_vFC`

**Location:** project_v2/radiation.py, lines 67 and 89

### The Problem:
There's an **inconsistent normalization** between the function setup and internal processing:

**Line 67:**
```python
z, nu0, T0 = (1, env.nu0FS, env.T0FS) if (front == 'RS') else (4, env.nu0, env.T0)
```
- If `front == 'RS'`: sets `nu0 = env.nu0FS`
- If `front == 'FS'`: sets `nu0 = env.nu0`

**Line 89 (inside `run_nuFnu_vFC`):**
```python
nu0 = (env.nu0 if (trac < 1.5) else env.nu0FS)
```
- If `trac < 1.5` (RS region): sets `nu0 = env.nu0`
- If `trac > 1.5` (FS region): sets `nu0 = env.nu0FS`

### The Contradiction:
- For RS front: Line 67 uses `env.nu0FS`, but line 89 uses `env.nu0` (MISMATCH!)
- For FS front: Line 67 uses `env.nu0`, but line 89 uses `env.nu0FS` (MISMATCH!)

The normalizations are **swapped**!

## Main Differences Summary:

1. **Data Source**:
   - `get_pksnunFnu`: Uses preprocessed CSV files (thinshell_RS.csv, thinshell_FS.csv)
   - `get_peaks_from_data`: Uses raw run data (run_data_{z}.csv) or cached npz files

2. **Data Filtering**:
   - `get_pksnunFnu`: Processes all contributions from `extract_contribs()`
   - `get_peaks_from_data`: Uses `drop_duplicates(subset='i', keep='first')`

3. **Normalization**:
   - `get_pksnunFnu`: Works with unnormalized data, normalization happens implicitly
   - `get_peaks_from_data`: Explicitly normalizes, **but has a bug causing incorrect normalization**

4. **Preprocessing**:
   - `get_pksnunFnu`: May include extrapolation (via `extrapolate_early()`) and smoothing
   - `get_peaks_from_data`: No explicit extrapolation or smoothing mentioned

5. **Frequency Array**:
   - `get_pksnunFnu`: Uses frequency array from `get_radEnv()` 
   - `get_peaks_from_data`: Uses `Nnu=500` by default

## Recommendation:

The **normalization bug** in `get_radiation_vFC` (line 67) should be fixed. The condition should be:
```python
z, nu0, T0 = (4, env.nu0, env.T0) if (front == 'RS') else (1, env.nu0FS, env.T0FS)
```

Additionally, the different data sources and preprocessing steps will naturally lead to different results. You should verify:
1. Which data source is more accurate for your use case
2. Whether the extrapolation/smoothing in project_v1 is necessary
3. Whether the `drop_duplicates` approach in project_v2 correctly represents the physics
