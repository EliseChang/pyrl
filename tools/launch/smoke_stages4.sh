#!/bin/bash
# Smoke test for pagan_hist_stages4: the stage ladder on the per-timestep engagement cost.
#
# Three runs, each proving a different thing:
#
#   block    STIM0=50,  MAX_ITER=80   -- the duration schedule steps 50 -> 100 -> ... and stage 1 is
#                                        HELD once it reaches criterion ("duration schedule still
#                                        running"). About a minute.
#   promote  STIM0=50,  MAX_ITER=400  -- long enough for the schedule to complete (125 updates
#                                        after the 30-update pre-training) and for a real
#                                        promotion out of stage 1 to follow. The only cheap proof
#                                        that the ladder actually advances. Five to ten minutes.
#   jump     STIM0=1300, MAX_ITER=80  -- the control: with no schedule the window goes straight
#                                        from 20 to 1,300 ms at the hand-off and engagement
#                                        collapses to ZERO attempted trials, so the ladder never
#                                        starts. Note that this means promotion cannot be smoked by
#                                        pre-completing the schedule -- the two are incompatible,
#                                        which is itself the result the `off` arm was for.
#
#   smoke_stages4.sh [block|promote|jump ...]      default: all three
#
# Sequential and heavily nice'd, so it can run on a busy box. Run folders are _smoke_s4<which>,
# i.e. without the g06 tag, so RUN_FILTER="_g06" in the rollout scheduler excludes them; they are
# moved to scratch at the end.
# Environment: PYRL_REPO, PYRL_PYTHON, MAX_ITER.
set -u

WHICH=${@:-block promote jump}
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

cd "$REPO/examples" || exit 2
export PYTHONPATH=$REPO OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export THEANO_FLAGS="base_compiledir=$HOME/.theano"
mkdir -p "$LOGS" "$SCR/smoke"

CFG="--config Wrec_init=gaussian --config gates=False --config f_hidden=tanh --config tau=100 \
 --config rho=0.6 --config baseline_rho=1.15 --config p0=0.5 --config baseline_p0=1 \
 --config var_rec=0 --config baseline_var_rec=0 --config L2_x=0.001 --config lr=0.004 \
 --config baseline_lr=0.004 --config beta_entropy=0 --config Win=1 --config n_validation=360 \
 --config max_iter=80 --config checkfreq=5"

for W in $WHICH; do
  case $W in
    block)   STIM0=50;   ITER=${MAX_ITER:-80} ;;
    promote) STIM0=50;   ITER=${MAX_ITER:-400} ;;
    jump)    STIM0=1300; ITER=${MAX_ITER:-80} ;;
    *) echo "unknown: $W (want block|promote|jump)" >&2; exit 2 ;;
  esac
  CFG_W="$(echo "$CFG" | sed "s/--config max_iter=[0-9]*/--config max_iter=$ITER/")"
  echo "=== smoke stages4 $W (PAGAN_STIM0=$STIM0, max_iter=$ITER): $(date) ==="
  PAGAN_TSCOST=0.55 PAGAN_STIM0=$STIM0 PAGAN_RAMP=choice PAGAN_STAGES=full \
    nice -n 19 "$P" -u do.py --suffix _smoke_s4$W \
    --config weight_seed=1 --config stim_seed=1 --config action_seed=1 \
    $CFG_W models/pagan_hist_stages4.py train > "$LOGS/smoke_s4$W.log" 2>&1
  grep -E "Traceback|Error|phase .* ->|pagan_hist_stages4|val trials|presentation|P\(choice\)|P\(correct" \
    "$LOGS/smoke_s4$W.log" | tail -45
done

for W in $WHICH; do
  [ -d "work/data/pagan_hist_stages4_smoke_s4$W" ] && \
    rm -rf "$SCR/smoke/pagan_hist_stages4_smoke_s4$W" && \
    mv "work/data/pagan_hist_stages4_smoke_s4$W" "$SCR/smoke/" && echo "moved smoke_s4$W to $SCR/smoke"
done
