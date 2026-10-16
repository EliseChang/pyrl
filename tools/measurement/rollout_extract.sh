#!/bin/bash
# Roll out the checkpoints of one long run on the TASK distribution (schedule off, no
# pre-training phase) and extract the compact PsyTrack/GLM record, so that only the 6 MB record
# has to travel between machines (the rollouts themselves are ~30 MB per checkpoint).
#   rollout_extract.sh <run> <modelfile> <pyrl_python> <record_python> <scratch> <out_records>
# Checkpoints: every one up to iteration 300 (checkfreq 5 then) and every 20th after, plus the
# last. Rollouts that already exist are skipped by pyrl (trials-b rewrites them; we filter).
run=$1; model=$2; P=$3; PR=$4; SCRATCH=$5; OUT=$6
cd ~/pyrl_work/examples
export PYTHONPATH=$HOME/pyrl_work OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 SCRATCH=$SCRATCH
export PAGAN_PHASE=task PAGAN_ANTIBIAS=0 PAGAN_STAGE=10   # PAGAN_STAGE: full task for the stage-curriculum model
[ -n "$THEANO_EXTRA" ] && export THEANO_FLAGS="$THEANO_EXTRA"
[ "$run" != "${run/_ctrl_/}" ] || [ "$run" != "${run/abctrl/}" ] && export PAGAN_PRETRAIN=0
ck=$(ls work/data/$run/checkpoints/ | grep -o "iter_[0-9]*" | sed 's/iter_0*//; s/^$/0/' | sort -n)
last=$(echo "$ck" | tail -1)
want=$(echo "$ck" | awk -v last=$last '{ if ($1 <= 300 || $1 % 20 == 0 || $1 == last) print $1 }')
todo=""
for i in $want; do d=$SCRATCH/work/pyrl/examples/$run/checkpoints/$(printf iter_%06d $i); [ -f $d/trials_behavior.pkl ] || todo="$todo,$i"; done
todo=${todo#,}
echo "$run: $(echo $want | wc -w) checkpoints wanted, rolling out $(echo $todo | tr ',' ' ' | wc -w) new"
if [ -n "$todo" ]; then
  $P -u do.py --suffix _${run#pagan_hist_*_} $model run analysis/pagan_hist.py trials-b ${TPC:-25} iters=$todo > ~/pyrl_work/logs/rollout_$run.log 2>&1
fi
mkdir -p $OUT
$PR ~/rat_pretraining/psytrack_on_network.py --extract --run $run --scratch $SCRATCH --out $OUT 2>&1 | tail -1
