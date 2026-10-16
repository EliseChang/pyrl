#!/bin/bash
# On rebel: gather the compact records, mechanism and activity summaries from the three
# workstations, copy any run folders not yet on rebel (the builder reads training history and
# checkpoints from them), rebuild the explorer, and leave the site ready to rsync to the Mac.
# Idempotent; run as often as wanted.
set -e
REC=/mnt/big/pyrl_work/psytrack/records
mkdir -p $REC /mnt/big/pyrl_work/mechanism /mnt/big/pyrl_work/activity
for h in 100.123.125.55; do   # 12 Oct: only 0095 is ours now (z4 is Maja's, 0071 Elise's)
  rsync -aq --include="*_record.npz" --exclude="*" mpagan@$h:records/ $REC/ 2>/dev/null && echo "records from $h: ok"
  rsync -aq --include="*_mechanism.npz" --exclude="*" mpagan@$h:mechanism/ /mnt/big/pyrl_work/mechanism/ 2>/dev/null || true
  rsync -aq --include="*_activity.npz" --exclude="*" mpagan@$h:activity/ /mnt/big/pyrl_work/activity/ 2>/dev/null || true
  rsync -aq mpagan@$h:pyrl_work/logs/stages/ ~/pyrl_work/logs/stages/ 2>/dev/null || true    # stage-probe training logs (the stage per checkpoint is read from them)
  rsync -aq --include="pagan_hist_curriculum_*/***" --include="pagan_hist_antibias_*/***" --include="pagan_hist_effort*_*/***" --include="pagan_hist_stages_*/***" --exclude="*" mpagan@$h:pyrl_work/examples/work/data/ ~/pyrl_work/examples/work/data/ 2>/dev/null || true
done
echo "records on rebel: $(ls $REC/*_record.npz | wc -l); mechanism $(ls /mnt/big/pyrl_work/mechanism/*_mechanism.npz 2>/dev/null | wc -l); activity $(ls /mnt/big/pyrl_work/activity/*_activity.npz 2>/dev/null | wc -l)"
cd ~/rat_pretraining
rm -f /mnt/big/pyrl_work/explorer_site_v2/img/pairs/*.png
OMP_NUM_THREADS=2 ~/rat_strategy_discovery/.venv/bin/python build_network_explorer.py --site /mnt/big/pyrl_work/explorer_site_v2 --workers 10 2>&1 | grep -v Warning | grep -E "FAILED|explorer:|networks$" | head -8
