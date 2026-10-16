#!/bin/bash
# rebel (12 cores): effort model v2 (DECLINE action, lump cost), EFFORT 0.4, both
# arms, seeds 1-8: 8 runs (seeds 5-8) to 2800 updates = 201,600 trials. See pagan_hist_effort2.py.
cd ~/pyrl_work/examples
export PYTHONPATH=$HOME/pyrl_work OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
P=$HOME/anaconda3/envs/pyrl27/bin/python
CFG="--config Wrec_init=gaussian --config gates=False --config f_hidden=tanh --config tau=100 --config rho=1.15 --config baseline_rho=1.15 --config p0=0.5 --config baseline_p0=1 --config var_rec=0 --config baseline_var_rec=0 --config L2_x=0.001 --config lr=0.004 --config baseline_lr=0.004 --config beta_entropy=0 --config Win=1 --config max_iter=2800 --config checkfreq=5"
mkdir -p ~/pyrl_work/logs/effort2
for E in 0.4; do
  tag=$(echo $E | sed 's/0\.//; s/^\([0-9]\)$/\10/')      # 0.4 -> 40, 0.55 -> 55, 0.7 -> 70
  for w in 5 6 7 8; do
    s=$(printf %03d $w)
    PAGAN_EFFORT=$E nohup $P -u do.py --suffix _e2${tag}curr_w$s --config weight_seed=$w --config stim_seed=$w --config action_seed=$w $CFG models/pagan_hist_effort2.py train > ~/pyrl_work/logs/effort2/e2${tag}curr_w$s.log 2>&1 &
    PAGAN_EFFORT=$E PAGAN_PRETRAIN=0 nohup $P -u do.py --suffix _e2${tag}ctrl_w$s --config weight_seed=$w --config stim_seed=$w --config action_seed=$w $CFG models/pagan_hist_effort2.py train > ~/pyrl_work/logs/effort2/e2${tag}ctrl_w$s.log 2>&1 &
    sleep 8
  done
done
wait
echo "all finished $(date)" > ~/pyrl_work/logs/effort2/ALL_DONE
