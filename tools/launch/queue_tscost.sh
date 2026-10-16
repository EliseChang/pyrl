#!/bin/bash
# The per-timestep engagement cost run (version 4, 16 Oct 2026): four arms x 8 seeds = 32
# networks, cost 0.55, all pre-trained, differing only in how the stimulus-presentation window is
# grown after the hand-off from the nose-poke pre-training to the task.
#
#   tc55off   no duration schedule: full 1,300 ms from the hand-off on (the control)
#   tc55fix   grow +50 ms at every checkpoint, regardless of performance
#   tc55cho   grow iff P(choice) >= 0.70                                  (what pagan.update() does)
#   tc55per   grow iff P(choice) >= 0.70 and P(correct|choice) > 0.55     (reliably above chance)
#
# The schedule starts at PAGAN_STIM0=50 ms (four input bins). Deferred to a following run: the
# no-pre-training controls (the same arms with a `ctrl` suffix, which picks up PAGAN_PRETRAIN=0
# for free) and the staged curriculum.
#
#   queue_tscost.sh [seeds...]          default seeds 1..8
#
# Runs the four arms SEQUENTIALLY, each with JOBS seeds in flight, so JOBS is the global
# concurrency and the whole run is ~96 core-hours of single-threaded work (32 runs x 2,800
# updates x ~2.3 s/update). At JOBS=8 that is about 12 hours wall; at JOBS=4, about 24.
#
# Environment: PYRL_REPO, PYRL_PYTHON, JOBS, MAX_ITER, NICE (see launch_weak_start.sh).
# Resumable: an arm already carrying a DONE marker in logs/weak/ is skipped, so re-running this
# after an interruption picks up where it stopped. Delete the marker to force an arm to re-run.
set -u

SEEDS=${@:-1 2 3 4 5 6 7 8}
REPO=${PYRL_REPO:-$HOME/pyrl}
LOGS=$REPO/logs/weak
mkdir -p "$LOGS"

echo "queue_tscost: repo=$REPO seeds=[$SEEDS] jobs=${JOBS:-auto}"
for ARM in tc55off tc55fix tc55cho tc55per; do
  if [ -f "$LOGS/DONE_g06$ARM" ]; then
    echo "queue_tscost: $ARM already done ($(cat "$LOGS/DONE_g06$ARM")), skipping"
    continue
  fi
  echo "queue_tscost: starting $ARM at $(date)"
  bash "$REPO/tools/launch/launch_weak_start.sh" tscost $ARM $SEEDS 2>&1 \
    | tee -a "$LOGS/q_$ARM.log"
done
echo "queue_tscost: all four arms finished $(date)" | tee "$LOGS/DONE_tscost_queue"
