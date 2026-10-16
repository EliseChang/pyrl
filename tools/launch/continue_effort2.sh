#!/bin/bash
# Continue every effort-v2 run on this machine from its savefile to 5600 updates (403,200
# trials): the question is whether the slower engagement networks get there with a rat-sized
# budget doubled. `train recover` resumes weights, optimiser and RNG; PAGAN_PHASE=task keeps the
# curriculum clock in the task phase; the EFFORT value is read back from the run name.
#   continue_effort2.sh <machine: rebel|ws> [N_PAR]
M=$1
cd ~/pyrl_work/examples
export PYTHONPATH=$HOME/pyrl_work OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PAGAN_PHASE=task
if [ "$M" = rebel ]; then P=$HOME/anaconda3/envs/pyrl27/bin/python
else P=$HOME/miniconda3/envs/pyrl27/bin/python; export THEANO_FLAGS="cxx=$HOME/miniconda3/envs/pyrl27/bin/x86_64-conda-linux-gnu-g++,gcc.cxxflags=-I$HOME/miniconda3/envs/pyrl27/include,base_compiledir=$HOME/.theano"; fi
CFG="--config Wrec_init=gaussian --config gates=False --config f_hidden=tanh --config tau=100 --config rho=1.15 --config baseline_rho=1.15 --config p0=0.5 --config baseline_p0=1 --config var_rec=0 --config baseline_var_rec=0 --config L2_x=0.001 --config lr=0.004 --config baseline_lr=0.004 --config beta_entropy=0 --config Win=1 --config max_iter=5600 --config checkfreq=10"
mkdir -p ~/pyrl_work/logs/effort2
for run in $(ls work/data | grep "^pagan_hist_effort2_" | grep -v _copy | grep -v test | grep -E "${RUN_FILTER:-.}"); do   # RUN_FILTER: only the runs trained on this machine
  suf=${run#pagan_hist_effort2_}                       # e255curr_w003
  tag=${suf:2:2}; E="0.$tag"; [ "$tag" = "40" ] && E=0.4; [ "$tag" = "70" ] && E=0.7
  w=$((10#${suf##*_w})); SEED="--config weight_seed=$w --config stim_seed=$w --config action_seed=$w"
  PRE=""; [ "$suf" != "${suf/ctrl/}" ] && PRE="PAGAN_PRETRAIN=0"
  env PAGAN_EFFORT=$E $PRE nohup $P -u do.py --suffix _$suf $SEED $CFG models/pagan_hist_effort2.py train recover > ~/pyrl_work/logs/effort2/${suf}_cont.log 2>&1 &
  sleep 6
done
wait
echo "continuation finished $(date)" > ~/pyrl_work/logs/effort2/CONT_DONE${CONT_TAG:-}
