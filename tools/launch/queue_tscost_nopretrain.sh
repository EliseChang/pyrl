#!/bin/bash
# The no-pre-training control for the per-timestep engagement cost (17 Oct 2026).
#
# Straight into the full task from the first trial: no nose-poke pre-training, no stage ladder,
# no duration schedule (PAGAN_RAMP=off from a network that was never pre-trained, so the window is
# the full 1,300 ms from trial 1 rather than growing into it). This is the control the original
# requirements deferred ("comparison with no pre-training ... is deferred for following runs") --
# it isolates what the pre-training buys an engagement network, on top of the schedule collapse
# already seen in the pre-trained arms (tc55off's hand-off collapse, the stages4 `jump` smoke).
#
#   arm (1 x 8 seeds = 8 runs)
#     tl55offctrl   no pre-training (PAGAN_PRETRAIN=0), no schedule (PAGAN_RAMP=off), cost 0.55
#
# Expected outcome: collapse in every seed. At init P(DECLINE) = 0.25/step, so
# P(reaching a potent choice) = 0.75^130 ~ 1e-16 at the full 1,300 ms window, and nothing here
# ever shortens that window -- there is no pre-training phase to learn to hold fixation in, and
# no schedule to grow it gradually. Run anyway, as the control: a network that never engages is a
# result, not a wasted run, and it is the cleanest test of what the two-step pre-training window
# and the duration schedule are actually buying the other arms.
#
# max_iter is cut to 1,000 (from the usual 2,800): if the collapse is as predicted there is
# nothing to learn from updates 1,000-2,800, so cutting the budget saves about 65% of the compute
# for seeds expected to need none of it. n_validation is left at the model's own default (720,
# not the local family's 360) so the collapse trajectory is as well resolved as the tc55 arms' --
# compute is already cheap at this iteration count.
#
#   queue_tscost_nopretrain.sh [seeds...]          default seeds 1..8
#
# ~5 core-hours (8 runs x 1000 updates x ~2.25 s/update at the default n_validation=720), about
# 40 minutes wall at JOBS=8. Environment: PYRL_REPO, PYRL_PYTHON, JOBS, MAX_ITER, NICE (see
# launch_weak_start.sh). Resumable: an arm with a DONE marker in logs/weak/ is skipped.
#
# No launcher changes were needed: `tl55offctrl` already composes correctly out of
# launch_weak_start.sh's existing dispatch rules --
#   *ctrl               -> PAGAN_PRETRAIN=0
#   t[cl][0-9][0-9]*     -> PAGAN_TSCOST=0.55 PAGAN_STIM0=50
#   t[cl][0-9][0-9]off*  -> PAGAN_RAMP=off           (matches "offctrl")
# giving PAGAN_PRETRAIN=0 PAGAN_TSCOST=0.55 PAGAN_STIM0=50 PAGAN_RAMP=off. PAGAN_STIM0=50 is inert
# here: with PAGAN_RAMP=off, pagan_hist_tscost's _task_length() returns stimulus_max regardless of
# STIM0, confirmed by import: phase()=='task', stimulus==1300 (dec_t0==130) from the first trial.
set -u

SEEDS=${@:-1 2 3 4 5 6 7 8}
REPO=${PYRL_REPO:-$HOME/pyrl}
LOGS=$REPO/logs/weak
mkdir -p "$LOGS"

export MAX_ITER=${MAX_ITER:-1000}

echo "queue_tscost_nopretrain: repo=$REPO seeds=[$SEEDS] jobs=${JOBS:-auto} max_iter=$MAX_ITER"
for ARM in tl55offctrl; do
  if [ -f "$LOGS/DONE_g06$ARM" ]; then
    echo "queue_tscost_nopretrain: $ARM already done ($(cat "$LOGS/DONE_g06$ARM")), skipping"
    continue
  fi
  echo "queue_tscost_nopretrain: starting $ARM at $(date)"
  bash "$REPO/tools/launch/launch_weak_start.sh" tscost $ARM $SEEDS 2>&1 \
    | tee -a "$LOGS/q_$ARM.log"
done
echo "queue_tscost_nopretrain: finished $(date)" | tee "$LOGS/DONE_tscost_nopretrain_queue"
