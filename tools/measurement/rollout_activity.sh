#!/bin/bash
# Activity rollouts (hidden states of actor and critic, critic values) at up to 12 checkpoints
# of one run, then the compact summary. Task distribution, schedule off, no pre-training phase.
#   rollout_activity.sh <run> <modelfile> <pyrl_python> <summary_python> <scratch> <out_dir>
run=$1; model=$2; P=$3; PS=$4; SCRATCH=$5; OUT=$6
cd ~/pyrl_work/examples
export PYTHONPATH=$HOME/pyrl_work OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 SCRATCH=$SCRATCH
export PAGAN_PHASE=task PAGAN_ANTIBIAS=0 PAGAN_STAGE=10   # PAGAN_STAGE: full task for the stage-curriculum model
[ -n "$THEANO_EXTRA" ] && export THEANO_FLAGS="$THEANO_EXTRA"
[ "$run" != "${run/ctrl/}" ] && export PAGAN_PRETRAIN=0
ck=$(ls work/data/$run/checkpoints/ | grep -o "iter_[0-9]*" | sed 's/iter_0*//; s/^$/0/' | sort -n)
n=$(echo "$ck" | wc -l)
# 12 checkpoints spread over training, always including the first and the last
want=$(echo "$ck" | awk -v n=$n 'BEGIN{k=12} {a[NR]=$1} END{if(n<=k){for(i=1;i<=n;i++)print a[i]} else {for(j=0;j<k;j++){i=int(1+j*(n-1)/(k-1)+0.5); print a[i]}}}' | sort -n | uniq)
todo=""
for i in $want; do d=$SCRATCH/work/pyrl/examples/$run/checkpoints/$(printf iter_%06d $i); [ -f $d/trials_activity.pkl ] || todo="$todo,$i"; done
todo=${todo#,}
echo "$run: activity at $(echo $want | wc -w) checkpoints, $(echo $todo | tr ',' ' ' | wc -w) new"
[ -n "$todo" ] && $P -u do.py --suffix _${run#pagan_hist_*_} $model run analysis/pagan_hist.py trials-a 2 iters=$todo > ~/pyrl_work/logs/rolloutA_$run.log 2>&1
mkdir -p $OUT
$PS ~/rat_pretraining/activity_summary.py --run $run --scratch $SCRATCH --out $OUT 2>&1 | tail -3
