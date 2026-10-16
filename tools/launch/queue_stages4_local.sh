#!/bin/bash
# The stage-ladder run on the per-timestep engagement cost (16 Oct 2026), local.
#
# A third, separate run of the engagement line. The other two hand off from the nose-poke
# pre-training straight to the full task and compare how the stimulus-presentation window is grown
# (queue_tscost.sh, four arms, for 0071; queue_tscost_local.sh, three arms, on the Mac). This one
# hands off to the rats' 10-stage TaskSwitch ladder instead, so it is the arm that asks whether the
# graded stage structure protects engagement where the straight hand-off does not.
#
#   arm (1 x 8 seeds = 8 runs)
#     sc55full   pre-training -> stages 1-10, cost 0.55, the duration schedule P(choice)-gated and
#                confined to stage 1
#
# The duration schedule runs in stage 1 only and once in all of training: pagan_hist_stages4 blocks
# promotion out of any stage while the presentation window is still growing, and once the window
# reaches 1,300 ms the schedule is a permanent no-op. Expect stage 1 to hold until about update 155
# (30 updates of pre-training, then 25 growth opportunities at one per checkpoint), then the ladder
# proper. If the schedule ever stalls the seed stays in stage 1 for the whole run -- accepted
# deliberately, and interpretable: the ladder never started because engagement never scaled.
#
#   queue_stages4_local.sh [seeds...]          default seeds 1..8
#
# ~9 core-hours (8 runs x 2800 updates x ~1.50 s/update at NVAL=360), so about 1.2 h wall at
# JOBS=8. Environment: PYRL_REPO, PYRL_PYTHON, JOBS, MAX_ITER, NICE (see launch_weak_start.sh).
# Resumable: an arm with a DONE marker in logs/weak/ is skipped.
#
# Deferred, each one launcher invocation away now that the model file exists: the `short` ladder
# (arm sc55short), the no-pre-training control (sc55fullctrl, which picks up PAGAN_PRETRAIN=0 for
# free), the other two ramp criteria crossed with the ladder, and a PAGAN_RAMP=off stage arm.
set -u

SEEDS=${@:-1 2 3 4 5 6 7 8}
REPO=${PYRL_REPO:-$HOME/pyrl}
LOGS=$REPO/logs/weak
mkdir -p "$LOGS"

export NVAL=${NVAL:-360}

echo "queue_stages4_local: repo=$REPO seeds=[$SEEDS] jobs=${JOBS:-auto} nval=$NVAL"
for ARM in sc55full; do
  if [ -f "$LOGS/DONE_g06$ARM" ]; then
    echo "queue_stages4_local: $ARM already done ($(cat "$LOGS/DONE_g06$ARM")), skipping"
    continue
  fi
  echo "queue_stages4_local: starting $ARM at $(date)"
  bash "$REPO/tools/launch/launch_weak_start.sh" stages4 $ARM $SEEDS 2>&1 \
    | tee -a "$LOGS/q_$ARM.log"
done
echo "queue_stages4_local: finished $(date)" | tee "$LOGS/DONE_stages4_local_queue"
