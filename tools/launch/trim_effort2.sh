#!/bin/bash
# Keep only the 8 probe runs of effort v2 (EFFORT 0.55, both arms, seeds 1-4); stop the rest.
KEEP="e255curr_w001 e255curr_w002 e255curr_w003 e255curr_w004 e255ctrl_w001 e255ctrl_w002 e255ctrl_w003 e255ctrl_w004"
for pid in $(pgrep -f "pagan_hist_effort2.py train"); do
  cmd=$(tr '\0' ' ' < /proc/$pid/cmdline); suf=$(echo "$cmd" | grep -o "suffix _e2[0-9a-z_]*" | cut -d_ -f2-)
  keep=0; for k in $KEEP; do [ "$suf" = "$k" ] && keep=1; done
  [ $keep = 0 ] && kill $pid && echo "stopped $suf"
done
pkill -f "bash launch_.*effort2" 2>/dev/null
sleep 2; echo "still training: $(pgrep -f 'pagan_hist_effort2.py train' | wc -l)"
