#!/bin/bash
# The per-timestep engagement cost run, SLIMMED LOCAL VERSION (16 Oct 2026).
#
# A separate run from the four-arm one in queue_tscost.sh, trimmed to fit a laptop. That full run
# stays queued for mvm-lin-0071 as it is, and nothing here changes it. Two cuts:
#
#   1. three arms, not four: the no-schedule control (`off`) is dropped, because the smoke test
#      already answered it - at the hand-off it goes from 0.996 of trials reaching a choice to
#      0.000 within one checkpoint, essentially all of it mid-window withdrawal at a median step
#      in the 30s-40s. The control's job was to show that jumping straight to the full window
#      destroys engagement, and it does.
#   2. n_validation 360, not 720, which is where most of the saving is: a checkpoint's validation
#      batch is the larger half of the per-update cost (720/5 = 144 validation trials per update
#      against 72 training trials), so halving it takes the run from 2.25 to 1.50 s/update.
#
#   arms (3 x 8 seeds = 24 runs)
#     tl55fix   grow +50 ms at every checkpoint, regardless of performance
#     tl55cho   grow iff P(choice) >= 0.70                               (what pagan.update() does)
#     tl55per   grow iff P(choice) >= 0.70 and P(correct|choice) > 0.55  (reliably above chance)
#
# The `tl` prefix (against `tc` for the full run) keeps the two runs' folder names, records and
# explorer classes apart, so they can never be pooled. They are NOT comparable seed-for-seed: the
# same weight_seed gives the same starting weights, but validation draws consume the same stimulus
# RNG stream as training draws, so a 360-trial batch puts every later trial out of step with the
# 720-trial run.
#
# What the smaller batch costs, accepted deliberately. The curriculum gates read one validation
# batch, so their precision halves: with ~359 decided trials the standard error of P(correct) at
# chance is 0.026 against 0.019 at 720. A network exactly at chance therefore clears the perf
# arm's 0.55 gate on 2.9% of checkpoints rather than 0.4% - about 16 growths' worth of noise over
# 560 checkpoints, against the 25 growths that complete the schedule. So `tl55per` may drift most
# of the way to the full window without ever being reliably above chance, and its schedule trace
# has to be read alongside its accuracy rather than as evidence on its own.
#
#   queue_tscost_local.sh [seeds...]          default seeds 1..8
#
# ~28 core-hours (24 runs x 2800 updates x 1.50 s/update), so about 3.5 h wall at JOBS=8 or 7 h at
# JOBS=4. Environment: PYRL_REPO, PYRL_PYTHON, JOBS, MAX_ITER, NICE (see launch_weak_start.sh).
# Resumable: an arm with a DONE marker in logs/weak/ is skipped.
set -u

SEEDS=${@:-1 2 3 4 5 6 7 8}
REPO=${PYRL_REPO:-$HOME/pyrl}
LOGS=$REPO/logs/weak
mkdir -p "$LOGS"

export NVAL=${NVAL:-360}

echo "queue_tscost_local: repo=$REPO seeds=[$SEEDS] jobs=${JOBS:-auto} nval=$NVAL"
for ARM in tl55fix tl55cho tl55per; do
  if [ -f "$LOGS/DONE_g06$ARM" ]; then
    echo "queue_tscost_local: $ARM already done ($(cat "$LOGS/DONE_g06$ARM")), skipping"
    continue
  fi
  echo "queue_tscost_local: starting $ARM at $(date)"
  bash "$REPO/tools/launch/launch_weak_start.sh" tscost $ARM $SEEDS 2>&1 \
    | tee -a "$LOGS/q_$ARM.log"
done
echo "queue_tscost_local: all three arms finished $(date)" | tee "$LOGS/DONE_tscost_local_queue"
