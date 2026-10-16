#!/bin/bash
# mvm-lin-0095 (32 cores): seeds 9-16 of the rat pre-training pilot, both arms, trained from
# scratch to 2800 updates = 201,600 trials, checkpoints every 5 updates (the curriculum's phase
# lengths are counted in checkpoints, so checkfreq must stay 5 for the pretrained arm).
# Environment: ~/miniconda3/envs/pyrl27 (conda-forge: python 2.7.18, numpy 1.16.5, scipy 1.2.1,
# gcc 11 via conda, libxcrypt for crypt.h; pip Theano 0.8.2). Theano needs the conda g++ and the
# env's include dir passed explicitly. About 6 s per update here (rebel: 15 s).
cd ~/pyrl_work/examples
export PYTHONPATH=$HOME/pyrl_work OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export THEANO_FLAGS="cxx=$HOME/miniconda3/envs/pyrl27/bin/x86_64-conda-linux-gnu-g++,gcc.cxxflags=-I$HOME/miniconda3/envs/pyrl27/include,base_compiledir=$HOME/.theano"
P=$HOME/miniconda3/envs/pyrl27/bin/python
CFG="--config Wrec_init=gaussian --config gates=False --config f_hidden=tanh --config tau=100 --config rho=1.15 --config baseline_rho=1.15 --config p0=0.5 --config baseline_p0=1 --config var_rec=0 --config baseline_var_rec=0 --config L2_x=0.001 --config lr=0.004 --config baseline_lr=0.004 --config beta_entropy=0 --config Win=1 --config max_iter=2800 --config checkfreq=5"
mkdir -p ~/pyrl_work/logs/long
for w in 9 10 11 12 13 14 15 16; do
  s=$(printf %03d $w)
  nohup $P -u do.py --suffix _curr_w$s --config weight_seed=$w --config stim_seed=$w --config action_seed=$w $CFG models/pagan_hist_curriculum.py train > ~/pyrl_work/logs/long/curr_w$s.log 2>&1 &
  PAGAN_PRETRAIN=0 nohup $P -u do.py --suffix _ctrl_w$s --config weight_seed=$w --config stim_seed=$w --config action_seed=$w $CFG models/pagan_hist_curriculum.py train > ~/pyrl_work/logs/long/ctrl_w$s.log 2>&1 &
  sleep 20   # stagger the Theano compiles, which share one compile lock
done
wait
echo "all 16 finished $(date)" > ~/pyrl_work/logs/long/ALL_DONE
