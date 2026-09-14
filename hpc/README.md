# Running post-processing on the cluster

**Policy (HPC staff, 2026-09-14).** The file server was taken down by jobs that read and
write large numbers of files. Such jobs must now copy what they need to the compute
node's local temp directory, run there, and copy the outputs back to `~/work` at the end.

Our post-processing is exactly that job. One hi-res cell history is 18 MB and a shell
pass opens 20 000 of them, once per sweep point; the file **open** alone costs ~2.3 s per
cell over NFS, so a sweep is latency against the file server rather than computation. The
extraction is worse: it re-reads all 153 224 snapshots once per cell block, from 32
processes at a time.

GAMMA runs are *not* in scope and `submit.sh` is unchanged: a simulation writes one 3 MB
snapshot every few seconds, which is not a load worth restructuring around.

## Use

```bash
KEY=<run> sbatch [slurm opts] hpc/submit_analysis.sh '<command>'
```

The command runs on a compute node with its working directory inside a **copy of this
tree on the node's `/tmp`**, with `GAMMA_DIR` pointing at it, so every `results/`,
`figures/` and `extracted_data/` path the analysis touches is local. What was written is
copied back when the job ends -- on failure too.

Cell extraction (writes 331 GiB, more than `/tmp` holds, so it is drained back as it
goes):

```bash
KEY=cooling_g100_hires STAGE_CELLS=out STAGE_DUMPS=1 \
sbatch --cpus-per-task=32 --mem=180G hpc/submit_analysis.sh \
  'python -c "import analysis_hydro as a; a.extract_data_cells(\"$KEY\", None, nproc=32, cell_block=5000)"'
```

A sweep -- the case staging is really for, since it reads every cell once per point:

```bash
KEY=cooling_g100_hires STAGE_CELLS=in \
sbatch --cpus-per-task=32 --mem=120G hpc/submit_analysis.sh \
  'python -c "import sweep_gammacm as s; s.main(key=\"$KEY\", nproc=32)"'
```

Both shells' hi-res cells are 331 GiB and do not fit; one shell (10 000 x 18 MB = 165 GiB)
does, so say which shell the job is for and only that one is staged:

```bash
STAGE_SHELL=4      # or 1 -- the index range comes from the run's own grid
```

`CELLS_FILTER` takes raw rsync filters if you ever need something `STAGE_SHELL` cannot
express. A partial cell copy is never kept either way: it would be silently wrong, summing
emission over whichever cells happened to arrive, so a failed copy is thrown away and the
cells are read from `~/work` instead.

## The sweep launchers

`hpc/sweep_prep.sh` and `hpc/sweep_point.sh` are the staged versions of the `*_k.sh`
scripts, driven exactly as before:

```bash
P=$(sbatch --parsable --export=ALL,RUNKEY=$KEY,ZSH=$Z,METHOD=data+rarcut hpc/sweep_prep.sh)
sbatch --dependency=afterok:$P --export=ALL,RUNKEY=$KEY,ZSH=$Z,METHOD=data+rarcut \
       --array=0-7 hpc/sweep_point.sh
```

That array is the job the file server went down under: eight tasks, each opening every
cell of the run. Each task now stages its own shell onto its own node. Two tasks landing
on the same node will not both fit (165 GiB against ~208 GiB); the second notices and
reads from work, which is what the old behaviour was anyway.

Any other job script adopts the same thing in four lines:

```bash
STAGE_WORK="${SLURM_SUBMIT_DIR:-$PWD}"; . "$STAGE_WORK/hpc/stage.sh"
stage_open "name.$RUNKEY"; stage_analysis_in "$RUNKEY" || exit 2
stage_analysis_cd || exit 1
...work...; stage_analysis_out "$RUNKEY"
```

## Knobs

| variable | default | meaning |
|---|---|---|
| `KEY` | *required* | which run to stage |
| `STAGE_CELLS` | `in` | `in` copy the cells in (reading jobs) / `out` start empty and drain back (extraction) / `link` read over NFS |
| `STAGE_DUMPS` | `0` | `1` copies `phys*.out` in as well -- 502 GiB at hi-res, so it usually falls back to symlinks |
| `STAGE_SHELL` | unset | stage only shell 4 or 1's cells, the range read from the run's grid |
| `CELLS_FILTER` | empty | raw rsync filters for the cells copy, when `STAGE_SHELL` will not do |
| `DRAIN_SECS` | `600` | how often written cells are pushed back (`STAGE_CELLS=out`) |
| `GAMMA_STAGE` | `1` | `0` runs straight out of `~/work`, as before |
| `GAMMA_STAGE_ROOT` | auto | node-local root, if `/tmp` is ever not the right answer |
| `GAMMA_STAGE_RESERVE_GB` | `20` | free space to leave on the node's disk |

## What it does when things do not fit

Nothing here fails the job. A path that will not fit in the local disk is **symlinked**
to the work copy instead and read over NFS, exactly as before staging existed; the job is
slower and still correct. The `[stage]` lines in `slurm-<id>.out` say which paths were
copied and which were linked, so the log tells you whether the job actually got the
benefit.

Two guards decide: an up-front size estimate (one `readdir` plus a sample, never a full
`du` -- that walk is itself the load we are avoiding), and the free space watched while
each copy runs, which is what stops `/tmp` filling under a wrong estimate.

## What gets staged

The analysis code (7 MB, which also pins the job to one commit), `phys_input.ini`, the
run's own directory, `extracted_data`, and the figures tree -- minus the other runs'
figure folders, which are 93% of it (`fiducial` 769 MB, `hires` 347 MB) and which a run
never reads, since `method_outdir` lives under `figdir(key)` precisely so two runs cannot
share a cache. Those are symlinked, so anything that does read them still can.

The snapshots and the cells follow `STAGE_DUMPS` / `STAGE_CELLS` above.

## The node's local disk (measured 2026-09-14, cn01)

- `/tmp`, xfs on `/dev/mapper/vg_os-lv_root`: **219 G volume, ~208 G free** on an idle
  node. `TMPDIR` is set to it by Slurm.
- Every node reports `TmpDisk=0`, so the scheduler neither reserves this space nor cleans
  it up: check before filling it and remove what you wrote. `stage_close` does that on
  exit, including when the job is cancelled.
- `/dev/shm` is 95 G but RAM-backed, and would be charged to the job's memory.
- `/scratch` (30 T) is **NFS**, not a local disk -- staging there moves the load instead
  of removing it. `stage.sh` warns if it ever resolves to a network filesystem.

## Why `GAMMA_DIR` exists

`IO.py` and `environment.py` used to find the project root by taking the first path
element containing `GAMMA`. A staged copy cannot be named that way safely -- any
directory above it whose name contains `GAMMA` captures the match and every `results/`
path silently resolves to the wrong tree (this happened on the first test). Both modules
now honour `GAMMA_DIR` when it is set and fall back to the old rule when it is not, and
`submit_analysis.sh` verifies the root the stack actually resolved to before starting
work -- if it is not the staged tree, the job drops back to `~/work` rather than
analysing the wrong directory.
