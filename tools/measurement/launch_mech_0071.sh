#!/bin/bash
# mvm-lin-0071: seeded slow-point analysis for the base pairs, anti-bias and inverted runs (40)
cd ~/rat_pretraining
{ ls ~/pyrl_work/examples/work/data | grep "^pagan_hist_curriculum_"; ls ~/pyrl_work/examples/work/data | grep "^pagan_hist_antibias_"; } | grep -v _copy > /tmp/mech_all.txt
echo "$(wc -l < /tmp/mech_all.txt) runs"
rm -f ~/pyrl_work/logs/MECH_ALL_DONE
xargs -P 14 -I{} env OMP_NUM_THREADS=2 ~/psytrack_env/bin/python pyrl_mechanism.py --run {} --max-ckpts 24 --data ~/pyrl_work/examples/work/data --activity-scratch ~/pyrl_scratch --out ~/mechanism < /tmp/mech_all.txt > ~/pyrl_work/logs/mech_all.log 2>&1
echo done > ~/pyrl_work/logs/MECH_ALL_DONE
