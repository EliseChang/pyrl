#!/bin/bash
# On a workstation: stop the anti-bias runs, clear their outputs, relaunch with the given script.
LAUNCH=$1
pkill -u mpagan -f "pagan_hist_antibias.py train"
pkill -u mpagan -f "bash launch_"
sleep 3
echo "running after kill: $(pgrep -f 'do.py' | grep -v pgrep | wc -l)"
rm -rf ~/pyrl_work/examples/work/data/pagan_hist_antibias_ab* ~/pyrl_work/examples/work/figs/pagan_hist_antibias_ab*
mkdir -p ~/pyrl_work/logs/antibias && rm -f ~/pyrl_work/logs/antibias/*.log ~/pyrl_work/logs/antibias/ALL_DONE
cd ~/pyrl_work && nohup setsid bash $LAUNCH > logs/antibias/launcher.log 2>&1 < /dev/null &
sleep 2; echo "relaunched $LAUNCH"
