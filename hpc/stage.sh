# ---------------------------------------------------------------------------------------
# hpc/stage.sh -- run post-processing out of the compute node's LOCAL disk.
#
# SITE POLICY (HPC staff mail, 2026-09-14, after the file server was knocked over):
# a job that reads or writes a large number of files must copy what it needs to the
# node's local temp directory, run there, and copy the outputs back afterwards.
#
# Our post-processing is the textbook case. One hi-res cell history is 18 MB and the
# shell pass opens 20 000 of them, once per sweep point; the file OPEN alone costs
# ~2.3 s per cell over NFS (IO.open_cellframe's measurement), so a sweep is latency
# against the file server, not computation. GAMMA itself is not the problem -- it writes
# one 3 MB snapshot every few seconds -- so submit.sh is deliberately left alone.
#
# MEASURED ON THIS CLUSTER (2026-09-14, cn01):
#   local temp = /tmp, xfs on /dev/mapper/vg_os-lv_root, 219 G volume, ~208 G free idle.
#   Slurm's TmpFS is /tmp but every node reports TmpDisk=0, so the scheduler neither
#   reserves it nor cleans it: check the free space and remove what you wrote, which is
#   what this file does. /dev/shm is 95 G but RAM-backed (charged to the job).
#   /scratch is NFS (10.1.1.16:/export/scratch) -- NOT a local disk, staging there
#   moves the load rather than removing it.
#   What fits: one shell's hi-res cells (10 000 x 18 MB = 165 GiB) just fits; BOTH
#   shells (331 GiB) and the raw snapshots (502 GiB) do not -- those get symlinked.
#
# Source it, do not execute it:
#     source "$SLURM_SUBMIT_DIR/hpc/stage.sh"
#     stage_open cells                       # sets STAGE_DIR, arms the cleanup trap
#     stage_in   "results/$KEY/cells"        # copy (or symlink if it will not fit)
#     stage_link "results/$KEY"              # read through to the work copy
#     ( cd "$STAGE_DIR/bin/Tools/project_v2" && python ... )
#     stage_out  "results/$KEY/cells"        # copy back what the job wrote
#
# Every path is RELATIVE TO THE GAMMA ROOT and mirrored on both sides, so the staged
# tree is the work tree and a caller reads the same whether staging is on or off.
#
# Knobs, all optional, all env:
#   GAMMA_STAGE=0           staging off: STAGE_DIR becomes the work root and every
#                           stage_* call is a no-op, so a script still runs unchanged
#   GAMMA_STAGE_ROOT=DIR    node-local root to use instead of auto-detecting
#   GAMMA_STAGE_RESERVE_GB  free space to leave on that disk (default 20) -- /tmp is the
#                           node's OS volume and is shared with whoever else is on it
#   GAMMA_STAGE_KEEP=1      do not delete the staged tree when the job ends (debugging)
# ---------------------------------------------------------------------------------------

STAGE_ACTIVE=0
STAGE_DIR=""
STAGE_WORK="${STAGE_WORK:-${SLURM_SUBMIT_DIR:-$PWD}}"   # the GAMMA root the job came from
STAGE_PARENT=""

stage_log(){ printf '[stage] %s\n' "$*" >&2; }
STAGE_PY="$(command -v python || command -v python3 || echo python)"

stage_fstype(){ stat -f -c %T "$1" 2>/dev/null || echo unknown; }
stage_avail_kb(){ df -Pk "$1" 2>/dev/null | awk 'NR==2{print $4+0}'; }

# How big is this directory, roughly? du over NFS on 150 000 files is itself the load we
# are trying to avoid, so estimate instead: one readdir (ls -U, unsorted, no stat) for the
# count, times the mean size of a handful of entries. Our big directories are flat and
# uniform -- 153 224 snapshots of 3.3 MB, 20 000 cell histories of 18 MB -- which is
# exactly the case this is accurate for, and it is only ever used to skip a copy that is
# hopeless. The live check in stage_in is what actually protects the disk.
stage_estimate_kb(){
  local dir="$1" n sample
  [ -d "$dir" ] || { du -sk "$dir" 2>/dev/null | awk '{print $1+0}'; return; }
  n=$(ls -U "$dir" 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] || { echo 0; return; }
  sample=$(ls -U "$dir" 2>/dev/null | head -20 \
           | (cd "$dir" && xargs -d '\n' du -sk 2>/dev/null) \
           | awk '{s+=$1; k++} END{if(k) print s/k; else print 0}')
  awk -v n="$n" -v m="$sample" 'BEGIN{printf "%d", n*m}'
}

stage_is_network(){
  case "$(stage_fstype "$1")" in
    nfs|nfs4|lustre|gpfs|beegfs|smb2|cifs|ceph) return 0 ;;
    *) return 1 ;;
  esac
}

# Candidate local roots, best first. On this cluster it resolves to /tmp (TMPDIR is set
# to it by Slurm); the rest are here so the script survives a different node image.
stage_resolve_root(){
  local d
  for d in "${GAMMA_STAGE_ROOT:-}" "${SLURM_TMPDIR:-}" "${TMPDIR:-}" \
           /local /localscratch /scratch/local /tmp; do
    [ -n "$d" ] || continue
    [ -d "$d" ] && [ -w "$d" ] || continue
    printf '%s\n' "$d"
    return 0
  done
  return 1
}

# stage_open NAME -- create the staging tree and arm the cleanup trap.
# Never fails the job: if there is no usable local disk it leaves STAGE_ACTIVE=0 and
# points STAGE_DIR at the work root, which turns every other call into a no-op.
stage_open(){
  local name="${1:-job}" root fs avail

  if [ "${GAMMA_STAGE:-1}" = "0" ]; then
    stage_log "staging DISABLED (GAMMA_STAGE=0) -- running in $STAGE_WORK"
    STAGE_DIR="$STAGE_WORK"; STAGE_ACTIVE=0; return 0
  fi
  if ! command -v rsync >/dev/null 2>&1; then
    stage_log "no rsync on $(hostname -s) -- running in $STAGE_WORK"
    STAGE_DIR="$STAGE_WORK"; STAGE_ACTIVE=0; return 0
  fi
  if ! root="$(stage_resolve_root)"; then
    stage_log "no writable local temp directory -- running in $STAGE_WORK"
    STAGE_DIR="$STAGE_WORK"; STAGE_ACTIVE=0; return 0
  fi

  # The parent is deliberately lower-case: the Python side finds the project root by
  # taking the first path element containing 'GAMMA' (IO.py), so an upper-case element
  # above the mirror would hijack it and every results path would resolve too high. The
  # mirror itself keeps the work root's own name, so the two trees are interchangeable.
  STAGE_PARENT="$root/stage.${USER:-$(id -un)}.${SLURM_JOB_ID:-$$}.$name"
  STAGE_DIR="$STAGE_PARENT/$(basename "$STAGE_WORK")"
  if ! mkdir -p "$STAGE_DIR" 2>/dev/null; then
    stage_log "cannot create $STAGE_DIR -- running in $STAGE_WORK"
    STAGE_DIR="$STAGE_WORK"; STAGE_ACTIVE=0; STAGE_PARENT=""; return 0
  fi
  STAGE_ACTIVE=1
  trap 'stage_close' EXIT

  fs="$(stage_fstype "$STAGE_DIR")"
  avail="$(stage_avail_kb "$STAGE_DIR")"; avail="${avail:-0}"
  stage_log "node $(hostname -s): staging in $STAGE_DIR (fs=$fs, free=$(( avail / 1048576 )) GiB)"
  case "$fs" in
    tmpfs|ramfs)
      stage_log "WARNING: $root is RAM-backed ($fs) -- anything staged is charged to the"
      stage_log "         job's memory. Point GAMMA_STAGE_ROOT at a real disk." ;;
  esac
  if stage_is_network "$STAGE_DIR"; then
    stage_log "WARNING: $root is a NETWORK filesystem ($fs) -- this does not take load off"
    stage_log "         the file server. Point GAMMA_STAGE_ROOT at a real local disk."
  fi
  return 0
}

# Free space minus the reserve, in KiB. The reserve matters: /tmp is the node's OS
# volume, so filling it hurts every other job on the node, not just this one.
stage_free_kb(){
  local avail reserve
  avail="$(stage_avail_kb "$STAGE_DIR")"; avail="${avail:-0}"
  reserve=$(( ${GAMMA_STAGE_RESERVE_GB:-20} * 1048576 ))
  echo $(( avail - reserve ))
}

# stage_link REL -- mirror a work path into the staged tree as a SYMLINK.
# For inputs too big to copy (the 502 GiB of snapshots) and as the fallback whenever a
# copy will not fit: the job still reads them, just over NFS as it always did.
stage_link(){
  [ "$STAGE_ACTIVE" = 1 ] || return 0
  local rel="${1%/}"
  [ -e "$STAGE_WORK/$rel" ] || { stage_log "link: $rel does not exist, skipped"; return 0; }
  rm -rf "$STAGE_DIR/$rel"
  mkdir -p "$(dirname "$STAGE_DIR/$rel")"
  ln -s "$STAGE_WORK/$rel" "$STAGE_DIR/$rel"
  stage_log "link   $rel -> work (read over NFS)"
}

# stage_in REL [rsync opts...] -- copy a work path into the staged tree.
# A WHOLE directory that will not fit is symlinked instead: a post-processing job should
# read slowly over NFS rather than die at minute zero. Two guards, because neither alone
# is enough:
#   - an up-front estimate, so 502 GiB of snapshots are never even started;
#   - the free space watched WHILE rsync runs, which is what stops the node's /tmp from
#     being filled when the estimate guessed wrong.
# A FILTERED copy (extra rsync args) never falls back to a symlink, and neither does a
# single file: the caller is staging part of a directory, and replacing the whole thing
# with a link to ~/work would both hide what was already staged there and turn a local
# write into a write straight back onto the file server. Those return 1 instead and leave
# what was copied, for the caller to decide about.
stage_in(){
  [ "$STAGE_ACTIVE" = 1 ] || return 0
  local rel="${1%/}"; shift
  local src="$STAGE_WORK/$rel" est free pid rc
  [ -e "$src" ] || { stage_log "in:    $rel does not exist, skipped"; return 0; }

  local filtered=0
  [ "$#" -gt 0 ] || [ ! -d "$src" ] && filtered=1
  free="$(stage_free_kb)"
  if [ "$filtered" = 0 ]; then
    est="$(stage_estimate_kb "$src")"
    stage_log "in     $rel  (~$(( est / 1048576 )) GiB, $(( free / 1048576 )) GiB usable)"
    if [ "$est" -ge "$free" ]; then
      stage_log "       WILL NOT FIT -- linking to the work copy instead"
      stage_link "$rel"
      return 0
    fi
  else
    stage_log "in     $rel  ($(( free / 1048576 )) GiB usable)"
  fi

  mkdir -p "$(dirname "$STAGE_DIR/$rel")"
  if [ -d "$src" ]; then
    mkdir -p "$STAGE_DIR/$rel"
    rsync -a "$@" "$src/" "$STAGE_DIR/$rel/" &
  else
    rsync -a "$@" "$src" "$STAGE_DIR/$rel" &
  fi
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    if [ "$(stage_free_kb)" -le 0 ]; then
      stage_log "       local disk down to the reserve -- aborting the copy"
      kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
      stage_in_fallback "$rel" "$filtered"
      return $?
    fi
    sleep 10
  done
  wait "$pid"; rc=$?
  [ "$rc" -eq 0 ] && return 0
  stage_log "       copy failed (rsync $rc)"
  stage_in_fallback "$rel" "$filtered"
}

stage_in_fallback(){
  local rel="$1" filtered="$2"
  if [ "$filtered" = 1 ]; then
    stage_log "       leaving the partial copy in place -- caller decides (rel=$rel)"
    return 1
  fi
  rm -rf "${STAGE_DIR:?}/$rel"
  stage_link "$rel"
  return 0
}

# stage_out REL [rsync opts...] -- copy a staged path back to the work tree.
# No --delete, ever: the work copy may hold results this job did not produce.
stage_out(){
  [ "$STAGE_ACTIVE" = 1 ] || return 0
  local rel="${1%/}"; shift
  local src="$STAGE_DIR/$rel"
  [ -d "$src" ] && [ ! -L "$src" ] || return 0     # a symlink was never staged
  mkdir -p "$STAGE_WORK/$rel"
  stage_log "out    $rel -> work"
  rsync -a "$@" "$src/" "$STAGE_WORK/$rel/"
}

# stage_drain REL -- copy back the finished files and delete the local copies.
# For outputs bigger than the local disk (a hi-res cell extraction writes 331 GiB into
# 208 GiB of /tmp): call it periodically and the local footprint stays bounded. Only
# files untouched for QUIET minutes are moved, so a file still being written is left.
stage_drain(){
  [ "$STAGE_ACTIVE" = 1 ] || return 0
  local rel="${1%/}" quiet="${2:-2}"
  local src="$STAGE_DIR/$rel"
  [ -d "$src" ] && [ ! -L "$src" ] || return 0
  mkdir -p "$STAGE_WORK/$rel"
  local list; list="$(mktemp)"
  ( cd "$src" && find . -type f -mmin "+$quiet" -print ) > "$list"
  if [ -s "$list" ]; then
    stage_log "drain  $rel: $(wc -l < "$list") files -> work"
    rsync -a --remove-source-files --files-from="$list" "$src/" "$STAGE_WORK/$rel/"
  fi
  rm -f "$list"
}

stage_close(){
  [ "$STAGE_ACTIVE" = 1 ] || return 0
  if [ "${GAMMA_STAGE_KEEP:-0}" = "1" ]; then
    stage_log "GAMMA_STAGE_KEEP=1 -- leaving $STAGE_PARENT on $(hostname -s)"
    return 0
  fi
  # Nothing here is policed by Slurm (TmpDisk=0), so leaving the tree behind would
  # silently shrink the next job's /tmp. Guard the pattern: never rm an unset path.
  case "$STAGE_PARENT" in
    */stage.*) stage_log "removing $STAGE_PARENT"; rm -rf "$STAGE_PARENT" ;;
    *) stage_log "refusing to remove '$STAGE_PARENT' (not a staging directory)" ;;
  esac
}

# ---------------------------------------------------------------------------------------
# The analysis tree: what every post-processing job stages, in one place, so a job script
# is four lines rather than a copy of this. Honours STAGE_CELLS / STAGE_DUMPS /
# CELLS_FILTER exactly as hpc/README.md documents them.
# ---------------------------------------------------------------------------------------

stage_analysis_in(){
  local key="$1"
  local cells="${STAGE_CELLS:-in}" dumps="${STAGE_DUMPS:-0}" src="$STAGE_WORK/results/$1"

  # The code travels with the job: 7 MB, and it pins the analysis to one commit for the
  # job's lifetime. A filtered stage_in never falls back to a symlink on its own (that
  # would mask what it had already copied), so these two calls say what they want.
  stage_in "bin/Tools/project_v2" --exclude='__pycache__' --exclude='results' \
    || stage_link "bin/Tools/project_v2"
  stage_in "phys_input.ini"
  stage_figs_in "$key"
  stage_in "extracted_data"

  [ "$STAGE_ACTIVE" = 1 ] && mkdir -p "$STAGE_DIR/results/$key"
  # phys_input.ini and field_correction.json decide what the physics IS: if they are
  # missing from the staged run directory the analysis silently falls back to the repo
  # root's .ini and reports a different run. Link them rather than lose them.
  stage_in "results/$key" --exclude='cells/' --exclude='phys[0-9]*.out' \
    || find "$src" -maxdepth 1 -mindepth 1 ! -name 'phys[0-9]*.out' ! -name cells \
            -exec ln -sf -t "$STAGE_DIR/results/$key/" {} +

  # The snapshots are handled here rather than through stage_in, because the fallback has
  # to apply to THEM and not to the directory that now holds the files just staged.
  # 502 GiB at hi-res against ~200 GiB of /tmp, so the symlink branch is the usual one.
  if [ "$STAGE_ACTIVE" = 1 ]; then
    if [ "$dumps" = "1" ] && [ "$(stage_estimate_kb "$src")" -lt "$(stage_free_kb)" ]; then
      stage_log "copying the snapshots in"
      rsync -a --include='phys[0-9]*.out' --exclude='*' "$src/" "$STAGE_DIR/results/$key/"
    else
      [ "$dumps" = "1" ] && stage_log "snapshots do not fit -- symlinking them"
      # one find with a batched exec, NOT a shell loop: there are 153 224 snapshots in a
      # hi-res run and a loop would fork ln that many times
      find "$src" -maxdepth 1 -name 'phys[0-9]*.out' \
           -exec ln -sf -t "$STAGE_DIR/results/$key/" {} +
    fi
  fi

  case "$cells" in
    in)   stage_cells_in "$key" ;;
    out)  [ "$STAGE_ACTIVE" = 1 ] && mkdir -p "$STAGE_DIR/results/$key/cells" ;;
    link) stage_link "results/$key/cells" ;;
    *)    stage_log "unknown STAGE_CELLS='$cells'"; return 2 ;;
  esac
  return 0
}

# figures/ holds the sweep point caches, so a job that cannot see them recomputes every
# point instead of reloading it -- but 93% of the tree is the OTHER runs' folders
# (fiducial 769 MB, hires 347 MB locally), and a run never reads another run's cache:
# method_outdir is under figdir(key) precisely so two runs cannot share one. Copy this
# run's, symlink the rest -- anything that does read them still can, over NFS.
stage_figs_in(){
  [ "$STAGE_ACTIVE" = 1 ] || return 0
  local key="$1" others d
  local args=()
  others="$(stage_other_run_folders "$key")"
  if [ -z "$others" ]; then
    stage_in "bin/Tools/figures" || stage_link "bin/Tools/figures"
    return 0
  fi
  for d in $others; do args+=(--exclude="/$d/"); done
  stage_log "figures: copying all but $(echo $others | tr '\n' ' ')"
  stage_in "bin/Tools/figures" "${args[@]}" || { stage_link "bin/Tools/figures"; return 0; }
  for d in $others; do stage_link "bin/Tools/figures/$d"; done
}

# The figure folders of every run that is NOT this one. environment.run_folder is the
# authority on the name ('fiducial' / 'hires' / the key itself), so this does not have to
# know the aliases.
stage_other_run_folders(){
  local key="$1"
  ( cd "$STAGE_WORK/bin/Tools/project_v2" 2>/dev/null || exit 1
    GAMMA_DIR="$STAGE_WORK" STAGE_WORK="$STAGE_WORK" "$STAGE_PY" -c '
import os, sys
from environment import run_folder
root = os.environ["STAGE_WORK"]
keep = run_folder(sys.argv[1])
figs = os.path.join(root, "bin", "Tools", "figures")
try:
    runs = {run_folder(k) for k in os.listdir(os.path.join(root, "results"))}
except OSError:
    sys.exit(0)
for d in sorted(runs - {keep}):
    if os.path.isdir(os.path.join(figs, d)):
        print(d)
' "$key" ) 2>/dev/null
}

# The cells, which are the whole point. A PARTIAL copy would be silently wrong -- the
# emission would be summed over whichever cells happened to arrive -- so any failure here
# throws the copy away and reads the work tree instead.
stage_cells_in(){
  [ "$STAGE_ACTIVE" = 1 ] || return 0
  local key="$1" rel="results/$1/cells" list="" ok=0
  if [ -n "${STAGE_SHELL:-}" ]; then
    list="$(stage_cells_list "$key" "$STAGE_SHELL")" || list=""
  fi
  if [ -n "$list" ]; then
    stage_log "cells: shell $STAGE_SHELL only, $(wc -l < "$list") files"
    stage_in "$rel" --files-from="$list" && ok=1
    rm -f "$list"
  elif [ -n "${CELLS_FILTER:-}" ]; then
    stage_in "$rel" ${CELLS_FILTER} && ok=1
  else
    stage_in "$rel" && ok=1        # unfiltered: stage_in links it itself if it will not fit
    return 0
  fi
  [ "$ok" = 1 ] && return 0
  stage_log "       partial cell copy discarded -- reading them from work"
  rm -rf "${STAGE_DIR:?}/$rel"
  stage_link "$rel"
}

# The cell files of one shell, as an rsync --files-from list. The index range comes from
# the RUN's own grid (shell_cells.shell_cell_range reads its phys_input.ini), not from a
# hand-written glob, and it is intersected with what is actually in the directory -- one
# readdir, and no missing entries for rsync to fail on. This is what makes a hi-res sweep
# stageable at all: both shells are 331 GiB and do not fit, one shell is 165 GiB and does.
stage_cells_list(){
  local key="$1" z="$2" cells="$STAGE_WORK/results/$1/cells" out
  [ -d "$cells" ] || return 1
  out="$(mktemp)"
  ls -U "$cells" 2>/dev/null | ( cd "$STAGE_WORK/bin/Tools/project_v2" 2>/dev/null || exit 1
    GAMMA_DIR="$STAGE_WORK" "$STAGE_PY" -c '
import sys, re
from shell_cells import shell_cell_range
k0, k1, _ = shell_cell_range(sys.argv[1], int(sys.argv[2]))
for name in sys.stdin.read().split():
    m = re.match(r"0*(\d+)", name)
    if m and k0 <= int(m.group(1)) <= k1:
        print(name)
' "$key" "$z" ) > "$out" 2>/dev/null
  if [ -s "$out" ]; then printf '%s\n' "$out"; else rm -f "$out"; return 1; fi
}

# Name the staged tree explicitly and check it took. Without GAMMA_DIR the analysis stack
# infers its root from the cwd by the first path element containing 'GAMMA', which is
# right here but is not something to rely on for a job that writes 300 GiB into it. If
# the stack resolves anywhere else, drop back to ~/work rather than analysing the wrong
# directory: slow is a nuisance, wrong is a lost job.
stage_analysis_cd(){
  local seen
  export GAMMA_DIR="$STAGE_DIR"
  cd "$STAGE_DIR/bin/Tools/project_v2" || return 1
  seen="$("$STAGE_PY" -c 'import IO; print(IO.GAMMA_dir)' 2>/dev/null)"
  if [ "$STAGE_ACTIVE" = 1 ] && [ "$seen" != "$STAGE_DIR" ]; then
    stage_log "ERROR: the analysis stack resolves to '$seen', not '$STAGE_DIR'"
    stage_log "       falling back to $STAGE_WORK (no staging)"
    stage_close
    STAGE_ACTIVE=0
    STAGE_DIR="$STAGE_WORK"
    export GAMMA_DIR="$STAGE_WORK"
    cd "$STAGE_WORK/bin/Tools/project_v2" || return 1
  fi
  stage_log "cwd=$PWD"
  stage_log "root=$GAMMA_DIR"
  return 0
}

# Copy back what the job produced. Call it even when the job failed: a run that died after
# twenty hours of extraction still wrote cells worth keeping. No --delete anywhere -- the
# work copy holds results this job never produced. --no-links skips the symlinked inputs,
# which are already in ~/work and must never be copied back over themselves.
stage_analysis_out(){
  local key="$1"
  cd "$STAGE_WORK" || return 1
  stage_out "results/$key" --exclude='phys[0-9]*.out' --no-links
  stage_out "bin/Tools/figures" --no-links
  stage_out "extracted_data" --no-links
}
