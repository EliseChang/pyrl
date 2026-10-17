#!/bin/bash
# Measure one trained run end to end: behaviour rollout -> record, activity rollout -> summary,
# mechanism. The portable replacement for Marino's rollout_extract.sh / rollout_activity.sh, which
# hardcode his layout (`cd ~/pyrl_work/examples`, `PYTHONPATH=$HOME/pyrl_work`, and sibling scripts
# at `~/rat_pretraining/*.py`) and so will not run from this repo.
#
#   measure_run.sh <run> <modelfile> [steps]
#
#   <run>        the run folder name under examples/work/data, e.g.
#                pagan_hist_tscost_g06tl55cho_w002
#   <modelfile>  path relative to examples/, e.g. models/pagan_hist_tscost.py
#   [steps]      comma-separated subset of: behaviour,record,activity,summary,mechanism
#                default: all five
#
# Environment:
#   PYRL_REPO       repo root                              (default $HOME/pyrl)
#   PYRL_PYTHON     Python 2.7 + Theano, for the rollouts  (default: first candidate found)
#   PYRL_PYTHON3    Python 3 + numpy/scipy/matplotlib, for the npz steps
#                   (default: first candidate found -- NOT plain anaconda3, see below)
#   SCRATCH         rollout scratch                        (default $HOME/scratch)
#   RECORDS/ACTIVITY/MECHANISM   output dirs               (default $HOME/{records,activity,mechanism})
#   TPC             trials per condition for the behaviour rollout (default 10, as rollout_arm.sh)
#   TPC_A           trials per condition for the activity rollout  (default 2)
#   MAX_CKPTS       checkpoints for the mechanism search           (default 24)
#   N_ACTIVITY      checkpoints for the activity rollout           (default 12)
#
# A NOTE ON THE PYTHON 3 INTERPRETER. `~/anaconda3/bin/python3` on mvm-lin-0071 pairs numpy 2.0.2
# with scipy 1.7.3, so `scipy.special` raises "numpy.dtype size changed, may indicate binary
# incompatibility" -- which breaks the record and mechanism steps. Use `~/anaconda3/envs/findr` or
# `~/anaconda3/envs/rnn`, both of which have a consistent numpy/scipy. The candidate list below
# prefers those and deliberately does not fall back to the base environment.
#
# WHAT THE ROLLOUT MEASURES. The env below pins every checkpoint to the plain full task:
# PAGAN_PHASE=task (no pre-training phase), PAGAN_ANTIBIAS=0 (no bias schedule), PAGAN_STAGE=10
# (full task for the stage models), and PAGAN_STIM0 deliberately unset so pagan_hist_tscost's
# default is stimulus_max -- the full 1,300 ms window, not whatever the duration schedule had
# reached at that checkpoint. That is what makes checkpoints comparable across arms and runs.
#
# Checkpoint policy for the behaviour rollout is Marino's: every checkpoint up to iteration 300,
# every 20th after, plus the last -- 186 of 561 for a 2,800-update run at checkfreq 5.
#
# Measured cost on mvm-lin-0071, per run, at TPC=10 (one core, 186 checkpoints):
#   behaviour ~12.5 min and 2.4 GB of scratch, record ~22 s, activity ~20 s and 0.5 GB,
#   summary ~2 s, mechanism ~70 s. About 15 min and 2.9 GB per run; the ~25 MB of npz is what
#   you keep. At TPC=25 the behaviour rollout is ~30 min and 6 GB instead.
set -u

if [ $# -lt 2 ]; then
  echo "usage: $0 <run> <modelfile relative to examples/> [steps]" >&2
  echo "  e.g. $0 pagan_hist_tscost_g06tl55cho_w002 models/pagan_hist_tscost.py" >&2
  exit 2
fi
run=$1; model=$2; steps=${3:-behaviour,record,activity,summary,mechanism}

REPO=${PYRL_REPO:-$HOME/pyrl}
SCRATCH=${SCRATCH:-$HOME/scratch}
RECORDS=${RECORDS:-$HOME/records}
ACTIVITY=${ACTIVITY:-$HOME/activity}
MECHANISM=${MECHANISM:-$HOME/mechanism}
TOOLS=$REPO/tools/measurement
LOGS=$REPO/logs/measure

if [ -z "${PYRL_PYTHON:-}" ]; then
  for c in "$HOME/anaconda3/envs/pyrl/bin/python" "$HOME/miniforge3/envs/pyrl27/bin/python" \
           "$HOME/miniconda3/envs/pyrl27/bin/python" "$HOME/mambaforge/envs/pyrl27/bin/python"; do
    [ -x "$c" ] && PYRL_PYTHON=$c && break
  done
fi
[ -x "${PYRL_PYTHON:-}" ] || { echo "no Python 2.7 found; set PYRL_PYTHON" >&2; exit 2; }

if [ -z "${PYRL_PYTHON3:-}" ]; then
  for c in "$HOME/anaconda3/envs/findr/bin/python" "$HOME/anaconda3/envs/rnn/bin/python"; do
    [ -x "$c" ] && PYRL_PYTHON3=$c && break
  done
fi
[ -x "${PYRL_PYTHON3:-}" ] || { echo "no usable Python 3 found; set PYRL_PYTHON3 (not base anaconda3, see header)" >&2; exit 2; }

P=$PYRL_PYTHON; PR=$PYRL_PYTHON3
D=$REPO/examples/work/data/$run
[ -d "$D" ] || { echo "no such run folder: $D" >&2; exit 2; }
[ -f "$D/$run.pkl" ] || { echo "missing savefile: $D/$run.pkl" >&2; exit 2; }
nck=$(ls "$D/checkpoints" 2>/dev/null | grep -c '^iter_[0-9]*\.pkl$' || true)
[ "$nck" -gt 0 ] || { echo "no checkpoints in $D/checkpoints -- nothing to roll out" >&2; exit 2; }

# the suffix do.py needs: the run name minus the model-file stem
suffix=_${run#pagan_hist_*_}

mkdir -p "$LOGS" "$RECORDS" "$ACTIVITY" "$MECHANISM"
cd "$REPO/examples" || exit 2
export PYTHONPATH=$REPO OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export THEANO_FLAGS="base_compiledir=$HOME/.theano"
export SCRATCH
export PAGAN_PHASE=task PAGAN_ANTIBIAS=0 PAGAN_STAGE=10
case $run in *ctrl*) export PAGAN_PRETRAIN=0 ;; esac

has() { case ",$steps," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
ckdir=$SCRATCH/work/pyrl/examples/$run/checkpoints

echo "measure_run: $run"
echo "  model=$model  suffix=$suffix  checkpoints on disk=$nck"
echo "  py2=$P"
echo "  py3=$PR"
echo "  steps=$steps  TPC=${TPC:-10}"

# ---------------------------------------------------------------- the checkpoint list
ck=$(ls "$D/checkpoints" | grep -o "iter_[0-9]*" | sed 's/iter_0*//; s/^$/0/' | sort -n)
last=$(echo "$ck" | tail -1)
want=$(echo "$ck" | awk -v last="$last" '{ if ($1 <= 300 || $1 % 20 == 0 || $1 == last) print $1 }')
echo "  behaviour checkpoints wanted: $(echo $want | wc -w) (last=$last)"

# ---------------------------------------------------------------- behaviour rollout
if has behaviour; then
  todo=""
  for i in $want; do
    d=$ckdir/$(printf iter_%06d $i)
    [ -f "$d/trials_behavior.pkl" ] || todo="$todo,$i"
  done
  todo=${todo#,}
  n=$(echo $todo | tr ',' ' ' | wc -w)
  if [ -n "$todo" ]; then
    echo "  [behaviour] rolling out $n new checkpoints -> $LOGS/behaviour_$run.log"
    $P -u do.py --suffix "$suffix" "$model" run analysis/pagan_hist.py \
       trials-b "${TPC:-10}" iters=$todo > "$LOGS/behaviour_$run.log" 2>&1 \
      || { echo "  [behaviour] FAILED, see $LOGS/behaviour_$run.log" >&2; tail -5 "$LOGS/behaviour_$run.log" >&2; exit 1; }
    echo "  [behaviour] done ($(ls -d $ckdir/iter_*/ 2>/dev/null | wc -l) checkpoints on disk)"
  else
    echo "  [behaviour] all $(echo $want | wc -w) already rolled out, skipping"
  fi
fi

# ---------------------------------------------------------------- the compact record
if has record; then
  echo -n "  [record] "
  $PR "$TOOLS/psytrack_on_network.py" --extract --run "$run" --scratch "$SCRATCH" --out "$RECORDS" 2>&1 | tail -1
fi

# ---------------------------------------------------------------- activity rollout
if has activity; then
  # N_ACTIVITY checkpoints spread over training, always including the first and the last
  act=$(echo "$want" | awk -v n="${N_ACTIVITY:-12}" '{a[NR]=$1} END {
          if (NR<=n) { for(i=1;i<=NR;i++) print a[i] }
          else { for(j=0;j<n;j++) print a[1+int(j*(NR-1)/(n-1))] } }' | sort -n -u)
  todo=""
  for i in $act; do
    d=$ckdir/$(printf iter_%06d $i)
    [ -f "$d/trials_activity.pkl" ] || todo="$todo,$i"
  done
  todo=${todo#,}
  if [ -n "$todo" ]; then
    echo "  [activity] rolling out $(echo $todo | tr ',' ' ' | wc -w) checkpoints -> $LOGS/activity_$run.log"
    $P -u do.py --suffix "$suffix" "$model" run analysis/pagan_hist.py \
       trials-a "${TPC_A:-2}" iters=$todo > "$LOGS/activity_$run.log" 2>&1 \
      || { echo "  [activity] FAILED, see $LOGS/activity_$run.log" >&2; tail -5 "$LOGS/activity_$run.log" >&2; exit 1; }
  else
    echo "  [activity] already rolled out, skipping"
  fi
fi

if has summary; then
  echo -n "  [summary] "
  OMP_NUM_THREADS=2 $PR "$TOOLS/activity_summary.py" --run "$run" --scratch "$SCRATCH" \
     --out "$ACTIVITY" 2>&1 | tail -1
fi

# ---------------------------------------------------------------- mechanism
if has mechanism; then
  echo "  [mechanism] fixed points over up to ${MAX_CKPTS:-24} checkpoints"
  OMP_NUM_THREADS=2 $PR "$TOOLS/pyrl_mechanism.py" --run "$run" --max-ckpts "${MAX_CKPTS:-24}" \
     --data "$REPO/examples/work/data" --activity-scratch "$SCRATCH" --out "$MECHANISM" 2>&1 | tail -2
fi

echo "measure_run: $run finished $(date)"
echo "  record:    $RECORDS/${run}_record.npz"
echo "  activity:  $ACTIVITY/${run}_activity.npz"
echo "  mechanism: $MECHANISM/${run}_policy_mechanism.npz"
echo "  scratch (deletable once the npz above exist): $ckdir"
