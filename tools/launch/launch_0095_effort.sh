#!/bin/bash
# mvm-lin-0095 (32 cores): model 3, option C, the effort-cost sweep. EFFORT in {0.4, 0.55, 0.7}
# (total cost of attending one trial; chance pays 0.5), arms pretrained (effcurr) and
# no_pretrain (effctrl), seeds 1-4: 24 runs from scratch to 2800 updates = 201,600 trials,
# checkpoints every 5 updates. Same seeds as the base pairs, so every network starts from
# the same weights as its no-effort twin.
cd ~/pyrl_work/examples
export PYTHONPATH=$HOME/pyrl_work OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export THEANO_FLAGS="cxx=$HOME/miniconda3/envs/pyrl27/bin/x86_64-conda-linux-gnu-g++,gcc.cxxflags=-I$HOME/miniconda3/envs/pyrl27/include,base_compiledir=$HOME/.theano"
P=$HOME/miniconda3/envs/pyrl27/bin/python
CFG="--config Wrec_init=gaussian --config gates=False --config f_hidden=tanh --config tau=100 --config rho=1.15 --config baseline_rho=1.15 --config p0=0.5 --config baseline_p0=1 --config var_rec=0 --config baseline_var_rec=0 --config L2_x=0.001 --config lr=0.004 --config baseline_lr=0.004 --config beta_entropy=0 --config Win=1 --config max_iter=2800 --config checkfreq=5"
mkdir -p ~/pyrl_work/logs/effort
for E in 0.4 0.55 0.7; do
  tag=$(echo $E | sed 's/0\.//; s/^\([0-9]\)$/\10/')      # 0.4 -> 40, 0.55 -> 55, 0.7 -> 70
  for w in 1 2 3 4; do
    s=$(printf %03d $w)
    PAGAN_EFFORT=$E nohup $P -u do.py --suffix _eff${tag}curr_w$s --config weight_seed=$w --config stim_seed=$w --config action_seed=$w $CFG models/pagan_hist_effort.py train > ~/pyrl_work/logs/effort/eff${tag}curr_w$s.log 2>&1 &
    PAGAN_EFFORT=$E PAGAN_PRETRAIN=0 nohup $P -u do.py --suffix _eff${tag}ctrl_w$s --config weight_seed=$w --config stim_seed=$w --config action_seed=$w $CFG models/pagan_hist_effort.py train > ~/pyrl_work/logs/effort/eff${tag}ctrl_w$s.log 2>&1 &
    sleep 15
  done
done
wait
echo "all finished $(date)" > ~/pyrl_work/logs/effort/ALL_DONE
