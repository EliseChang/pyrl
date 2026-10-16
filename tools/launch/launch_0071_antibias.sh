#!/bin/bash
# mvm-lin-0071 (32 cores): the anti-bias experiment, beta 5 (the lab's pipeline value), tau 50.
# Arms: pretrained + anti-bias (abcurr) and no_pretrain + anti-bias (abctrl), seeds 1-8, from
# scratch to 2800 updates = 201,600 trials, checkpoints every 5 updates (phase lengths are in
# checkpoints). Same seeds as the existing pretrained / no_pretrain runs, so each new network
# starts from the same weights as its beta-0 twin.
cd ~/pyrl_work/examples
export PYTHONPATH=$HOME/pyrl_work OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export THEANO_FLAGS="cxx=$HOME/miniconda3/envs/pyrl27/bin/x86_64-conda-linux-gnu-g++,gcc.cxxflags=-I$HOME/miniconda3/envs/pyrl27/include,base_compiledir=$HOME/.theano"
export PAGAN_AB_BETA=5 PAGAN_AB_TAU=50
P=$HOME/miniconda3/envs/pyrl27/bin/python
CFG="--config Wrec_init=gaussian --config gates=False --config f_hidden=tanh --config tau=100 --config rho=1.15 --config baseline_rho=1.15 --config p0=0.5 --config baseline_p0=1 --config var_rec=0 --config baseline_var_rec=0 --config L2_x=0.001 --config lr=0.004 --config baseline_lr=0.004 --config beta_entropy=0 --config Win=1 --config max_iter=2800 --config checkfreq=5"
mkdir -p ~/pyrl_work/logs/antibias
for w in 1 2 3 4 5 6 7 8; do
  s=$(printf %03d $w)
  nohup $P -u do.py --suffix _abcurr_w$s --config weight_seed=$w --config stim_seed=$w --config action_seed=$w $CFG models/pagan_hist_antibias.py train > ~/pyrl_work/logs/antibias/abcurr_w$s.log 2>&1 &
  PAGAN_PRETRAIN=0 nohup $P -u do.py --suffix _abctrl_w$s --config weight_seed=$w --config stim_seed=$w --config action_seed=$w $CFG models/pagan_hist_antibias.py train > ~/pyrl_work/logs/antibias/abctrl_w$s.log 2>&1 &
  sleep 20
done
wait
echo "all finished $(date)" > ~/pyrl_work/logs/antibias/ALL_DONE
