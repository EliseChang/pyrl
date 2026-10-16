#!/bin/bash
# Train weak-start (rho 0.6) pagan_hist networks from this repo. Portable: Linux workstation or
# an Apple Silicon Mac (see tools/launch/README_local_launch.md).
#
#   launch_weak_start.sh <model> <arm> <seeds...>
#
# <model> is the pagan_hist_<model>.py stem: curriculum | antibias | effort2 | stages | tscost.
# <arm> is the arm tag; the run folder is pagan_hist_<model>_g06<arm>_w<seed padded to 3>, which
# is what build_network_explorer.py's ARMS table pattern-matches on. Per-arm task switches are
# environment variables (do.py --config cannot set a task module's own globals) and are derived
# from the arm tag in the dispatch block below.
#
# Environment:
#   PYRL_REPO     repo root                        (default $HOME/pyrl)
#   PYRL_PYTHON   the Python 2.7 + Theano binary   (default: first of the candidates below)
#   JOBS          how many seeds to train at once  (default: physical/performance cores, capped 8)
#   MAX_ITER      updates per run                  (default 2800 = 201,600 trials)
#   NVAL          validation trials per checkpoint (default: the model's own, 720)
#   NICE          niceness wrapper                 (default "nice -n 10"; set NICE="" for none)
#
# NVAL is passed as --config n_validation, which `do.py train` applies to training only and
# records in the savefile. It changes the cost of a run (a checkpoint's validation batch is
# NVAL/checkfreq trials per update, against 72 training trials), the precision of the curriculum
# gates that read it, and - because validation draws consume the same stimulus RNG stream as
# training draws - the trial sequence itself. Runs with different NVAL start from identical
# weights but diverge immediately, so they are NOT seed-comparable and must not be pooled.
#
# This is the portable successor to Marino's models/scripts/launch_weak_start.sh in share_rl_rnn,
# which hardcodes ~/pyrl_work, ~/rat_pretraining and his pyrl27 env. The CFG string is the weak
# start verbatim: the only difference from the pre-13-Oct strong start is rho=0.6 (measured
# spectral radius about 0.8, the value of the Nature networks), with the critic left at
# baseline_rho=1.15.
set -u

if [ $# -lt 3 ]; then
  echo "usage: $0 <model> <arm> <seeds...>" >&2; exit 2
fi
MODEL=$1; ARM=$2; shift 2; SEEDS="$@"

REPO=${PYRL_REPO:-$HOME/pyrl}
LOGS=$REPO/logs/weak

# Python 2.7 + Theano. Override with PYRL_PYTHON; otherwise take the first that exists.
if [ -z "${PYRL_PYTHON:-}" ]; then
  for c in "$HOME/anaconda3/envs/pyrl/bin/python" \
           "$HOME/miniforge3/envs/pyrl27/bin/python" \
           "$HOME/miniconda3/envs/pyrl27/bin/python" \
           "$HOME/mambaforge/envs/pyrl27/bin/python"; do
    [ -x "$c" ] && PYRL_PYTHON=$c && break
  done
fi
if [ -z "${PYRL_PYTHON:-}" ] || [ ! -x "$PYRL_PYTHON" ]; then
  echo "no Python 2.7 found; set PYRL_PYTHON=/path/to/env/bin/python" >&2; exit 2
fi
P=$PYRL_PYTHON
case "$("$P" -c 'import sys; print(sys.version_info[0])' 2>/dev/null)" in
  2) ;;
  *) echo "PYRL_PYTHON=$P is not Python 2: $("$P" --version 2>&1)" >&2; exit 2 ;;
esac

# Default concurrency: performance cores on macOS, physical cores on Linux, capped at 8 so a
# laptop is not pinned flat. Each run is single-threaded by the OMP/MKL settings below.
if [ -z "${JOBS:-}" ]; then
  if [ "$(uname -s)" = Darwin ]; then
    JOBS=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.physicalcpu)
  else
    JOBS=$(nproc 2>/dev/null || echo 4)
  fi
  [ "$JOBS" -gt 8 ] && JOBS=8
fi

cd "$REPO/examples" || exit 2
export PYTHONPATH=$REPO OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export THEANO_FLAGS="base_compiledir=$HOME/.theano"
mkdir -p "$LOGS"

CFG="--config Wrec_init=gaussian --config gates=False --config f_hidden=tanh --config tau=100 \
 --config rho=0.6 --config baseline_rho=1.15 --config p0=0.5 --config baseline_p0=1 \
 --config var_rec=0 --config baseline_var_rec=0 --config L2_x=0.001 --config lr=0.004 \
 --config baseline_lr=0.004 --config beta_entropy=0 --config Win=1 \
 --config max_iter=${MAX_ITER:-2800} --config checkfreq=5"
[ -n "${NVAL:-}" ] && CFG="$CFG --config n_validation=$NVAL"

# arm tag -> task switches. Order matters only in that the cost prefix match must come first.
ENV=""
case $ARM in *ctrl)              ENV="PAGAN_PRETRAIN=0" ;; esac
case $ARM in abneg*)             ENV="$ENV PAGAN_AB_BETA=-5" ;; esac
case $ARM in e2[0-9][0-9]*)      ENV="$ENV PAGAN_EFFORT=0.${ARM:2:2}" ;; esac
case $ARM in in[0-9][0-9]*)      ENV="$ENV PAGAN_EFFORT=0.${ARM:2:2}" ;; esac
case $ARM in st[0-9][0-9]short)  ENV="$ENV PAGAN_EFFORT=0.${ARM:2:2} PAGAN_STAGES=short" ;; esac
case $ARM in st[0-9][0-9]full)   ENV="$ENV PAGAN_EFFORT=0.${ARM:2:2} PAGAN_STAGES=full" ;; esac
# per-timestep engagement cost (version 4). PAGAN_STIM0=50 starts the duration schedule at four
# input bins, where the accuracy ceiling is 0.833 rather than the 0.653 of a one-bin window, so
# the performance-gated arm has real headroom over its gate from the first task checkpoint.
# Never 10: at a 10 ms window pagan.get_condition yields zero input bins and no input is
# delivered at all, not even the go cue, which is the defect the abandoned fixcost arms had.
# `tc` is the full run (four arms, the model's own 720-trial validation batch); `tl` is the
# slimmed local run (three arms, NVAL=360) - a separate run with its own names, so the two can
# never be pooled by the explorer or by a records/ glob. Both read the cost from chars 2-3.
case $ARM in t[cl][0-9][0-9]*)    ENV="$ENV PAGAN_TSCOST=0.${ARM:2:2} PAGAN_STIM0=50" ;; esac
case $ARM in t[cl][0-9][0-9]off*) ENV="$ENV PAGAN_RAMP=off" ;; esac
case $ARM in t[cl][0-9][0-9]fix*) ENV="$ENV PAGAN_RAMP=fixed" ;; esac
case $ARM in t[cl][0-9][0-9]cho*) ENV="$ENV PAGAN_RAMP=choice" ;; esac
case $ARM in t[cl][0-9][0-9]per*) ENV="$ENV PAGAN_RAMP=perf" ;; esac

echo "launch_weak_start: model=$MODEL arm=$ARM seeds=$SEEDS jobs=$JOBS nval=${NVAL:-model default}"
echo "  python=$P"
echo "  env=[$ENV]"
for w in $SEEDS; do
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 10; done
  s=$(printf %03d $w)
  env $ENV nohup ${NICE-nice -n 10} "$P" -u do.py --suffix _g06${ARM}_w$s \
    --config weight_seed=$w --config stim_seed=$w --config action_seed=$w \
    $CFG models/pagan_hist_$MODEL.py train \
    > "$LOGS/g06${ARM}_w$s.log" 2>&1 &
  echo "  started g06${ARM}_w$s"
  sleep 8
done
wait
echo "finished $(date)" > "$LOGS/DONE_g06${ARM}"
echo "launch_weak_start: $MODEL $ARM done $(date)"
