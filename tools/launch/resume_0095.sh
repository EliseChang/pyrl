#!/bin/bash
# On mvm-lin-0095: resume seeds 1-5 (both arms) of the long runs, moved here from rebel.
cd ~/pyrl_work/examples
export PYTHONPATH=$HOME/pyrl_work OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export THEANO_FLAGS="cxx=$HOME/miniconda3/envs/pyrl27/bin/x86_64-conda-linux-gnu-g++,gcc.cxxflags=-I$HOME/miniconda3/envs/pyrl27/include,base_compiledir=$HOME/.theano"
P=$HOME/miniconda3/envs/pyrl27/bin/python
CFG="--config Wrec_init=gaussian --config gates=False --config f_hidden=tanh --config tau=100 --config rho=1.15 --config baseline_rho=1.15 --config p0=0.5 --config baseline_p0=1 --config var_rec=0 --config baseline_var_rec=0 --config L2_x=0.001 --config lr=0.004 --config baseline_lr=0.004 --config beta_entropy=0 --config Win=1 --config max_iter=2800 --config checkfreq=10"
for w in 1 2 3 4 5; do
  s=$(printf %03d $w)
  PAGAN_PHASE=task nohup $P -u do.py --suffix _curr_w$s --config weight_seed=$w --config stim_seed=$w --config action_seed=$w $CFG models/pagan_hist_curriculum.py train recover > ~/pyrl_work/logs/long/curr_w$s.log 2>&1 &
  PAGAN_PRETRAIN=0 nohup $P -u do.py --suffix _ctrl_w$s --config weight_seed=$w --config stim_seed=$w --config action_seed=$w $CFG models/pagan_hist_curriculum.py train recover > ~/pyrl_work/logs/long/ctrl_w$s.log 2>&1 &
  sleep 15
done
wait
echo "all 10 finished $(date)" > ~/pyrl_work/logs/long/ALL_DONE
