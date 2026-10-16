#!/bin/bash
# mvm-lin-0071: activity rollouts + summaries for the 16 base pairs (copied from rebel), the
# 16 anti-bias and the 8 inverted-anti-bias runs, 12 checkpoints each, 12 at a time.
export THEANO_EXTRA="cxx=$HOME/miniconda3/envs/pyrl27/bin/x86_64-conda-linux-gnu-g++,gcc.cxxflags=-I$HOME/miniconda3/envs/pyrl27/include,base_compiledir=$HOME/.theano"
P=$HOME/miniconda3/envs/pyrl27/bin/python; PS=$HOME/psytrack_env/bin/python; SCR=$HOME/pyrl_scratch; OUT=$HOME/activity
cd ~/pyrl_work/examples
{ ls work/data | grep "^pagan_hist_curriculum_" ; ls work/data | grep "^pagan_hist_antibias_"; } | grep -v _copy > /tmp/act_runs.txt
echo "$(wc -l < /tmp/act_runs.txt) runs"
xargs -P 12 -I{} bash -c 'run={}; case $run in *curriculum*) m=models/pagan_hist_curriculum.py;; *) m=models/pagan_hist_antibias.py;; esac; bash ~/rat_pretraining/rollout_activity.sh $run $m '"$P $PS $SCR $OUT"' > ~/pyrl_work/logs/act_$run.log 2>&1' < /tmp/act_runs.txt
echo "activity done $(date)" > ~/pyrl_work/logs/ACTIVITY_DONE
