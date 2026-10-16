#!/bin/bash
# On rebel: stop seeds 1-5 (both arms) of the long runs, copy their run folders to mvm-lin-0095,
# and resume them there. Seeds 6-8 stay on rebel.
H=mpagan@100.123.125.55
ssh $H "mkdir -p ~/pyrl_work/examples/work/data ~/pyrl_work/logs/long"
for w in 1 2 3 4 5; do for arm in curr ctrl; do
  s=$(printf %03d $w); run=pagan_hist_curriculum_${arm}_w$s
  pkill -f "[s]uffix _${arm}_w$s " && echo "stopped $run" || echo "no process for $run"
done; done
sleep 5
for w in 1 2 3 4 5; do for arm in curr ctrl; do
  s=$(printf %03d $w); run=pagan_hist_curriculum_${arm}_w$s
  rsync -aq ~/pyrl_work/examples/work/data/$run/ $H:pyrl_work/examples/work/data/$run/ && echo "copied $run ($(du -sh ~/pyrl_work/examples/work/data/$run | cut -f1))"
done; done
scp -q ~/pyrl_work/resume_0095.sh $H:pyrl_work/
ssh $H "cd ~/pyrl_work && (nohup setsid bash resume_0095.sh > logs/long/resume_launcher.log 2>&1 < /dev/null &); sleep 150; echo 0095 running=\$(pgrep -f '[d]o.py' | wc -l); grep -h 'resume override: max_iter' logs/long/*_w00[1-5].log | wc -l; grep -l Traceback logs/long/*_w00[1-5].log | head -3; grep -h 'After [0-9]* updates' logs/long/curr_w001.log | tail -1"
echo "rebel running=$(pgrep -f '[d]o.py' | wc -l)"
