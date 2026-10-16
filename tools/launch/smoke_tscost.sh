#!/bin/bash
# Smoke test for pagan_hist_tscost: one seed per ramp mode, 60 updates at checkfreq=5.
#
# 60 updates covers the 30-update nose-poke pre-training (6 checkpoints), the hand-off, and about
# six growth opportunities, so the log shows the phase transitions, the presentation held at 20 ms
# through pre-training, the hand-off, and the schedule either moving or not.
#
#   smoke_tscost.sh [modes...]      default: off fixed choice perf
#
# Runs the modes SEQUENTIALLY and heavily nice'd, so it can be run on a busy box: one core at a
# time at the lowest priority. Run this FIRST on a new machine -- it is the cheapest proof that
# the Python 2.7 + Theano environment there actually works (about 100 s per mode).
# Environment: PYRL_REPO, PYRL_PYTHON, MAX_ITER, PAGAN_STIM0. Run folders are named _smoke_tc<mode>, i.e. without the g06 tag,
# so RUN_FILTER="_g06" in the rollout scheduler excludes them; they are moved to scratch at the
# end rather than left in work/data.
set -u

MODES=${@:-off fixed choice perf}
REPO=${PYRL_REPO:-$HOME/pyrl}
if [ -z "${PYRL_PYTHON:-}" ]; then
  for c in "$HOME/anaconda3/envs/pyrl/bin/python" \
           "$HOME/miniforge3/envs/pyrl27/bin/python" \
           "$HOME/miniconda3/envs/pyrl27/bin/python" \
           "$HOME/mambaforge/envs/pyrl27/bin/python"; do
    [ -x "$c" ] && PYRL_PYTHON=$c && break
  done
fi
[ -x "${PYRL_PYTHON:-}" ] || { echo "set PYRL_PYTHON=/path/to/env/bin/python" >&2; exit 2; }
P=$PYRL_PYTHON
LOGS=$REPO/logs/weak
SCR=${SCRATCH:-$HOME/scratch}

cd $REPO/examples
export PYTHONPATH=$REPO OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export THEANO_FLAGS="base_compiledir=$HOME/.theano"
mkdir -p $LOGS $SCR/smoke

CFG="--config Wrec_init=gaussian --config gates=False --config f_hidden=tanh --config tau=100 \
 --config rho=0.6 --config baseline_rho=1.15 --config p0=0.5 --config baseline_p0=1 \
 --config var_rec=0 --config baseline_var_rec=0 --config L2_x=0.001 --config lr=0.004 \
 --config baseline_lr=0.004 --config beta_entropy=0 --config Win=1 \
 --config max_iter=${MAX_ITER:-60} --config checkfreq=5"

for M in $MODES; do
  echo "=== smoke $M: $(date) ==="
  PAGAN_TSCOST=0.55 PAGAN_STIM0=${PAGAN_STIM0:-50} PAGAN_RAMP=$M \
    nice -n 19 $P -u do.py --suffix _smoke_tc$M \
    --config weight_seed=1 --config stim_seed=1 --config action_seed=1 \
    $CFG models/pagan_hist_tscost.py train > $LOGS/smoke_tc$M.log 2>&1
  grep -E "Traceback|Error|phase .* ->|pagan_hist_tscost|val trials|mean cost paid|P\(choice\)|P\(correct" \
    $LOGS/smoke_tc$M.log | tail -60
done

for M in $MODES; do
  [ -d work/data/pagan_hist_tscost_smoke_tc$M ] && \
    mv work/data/pagan_hist_tscost_smoke_tc$M $SCR/smoke/ && echo "moved smoke_tc$M to $SCR/smoke"
done
