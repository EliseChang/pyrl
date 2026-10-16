#!/bin/bash
# Re-roll the new checkpoints and re-extract the record of every run of PREFIX on this machine
# (rollout_extract is idempotent: existing checkpoint rollouts are kept, new ones added).
#   refresh_records.sh <machine: rebel|0095|z4|0071> <prefix> <modelfile> [N_PAR] ; honours RUN_FILTER
M=$1; PREFIX=$2; MODEL=$3; NP=${4:-8}
case $M in
  rebel) P=$HOME/anaconda3/envs/pyrl27/bin/python; PR=$HOME/rat_strategy_discovery/.venv/bin/python; SCR=/mnt/big/pyrl_work/scratch; OUT=/mnt/big/pyrl_work/psytrack/records ;;
  *)     P=$HOME/miniconda3/envs/pyrl27/bin/python; PR=$HOME/psytrack_env/bin/python; SCR=$HOME/pyrl_scratch; OUT=$HOME/records
         export THEANO_EXTRA="cxx=$HOME/miniconda3/envs/pyrl27/bin/x86_64-conda-linux-gnu-g++,gcc.cxxflags=-I$HOME/miniconda3/envs/pyrl27/include,base_compiledir=$HOME/.theano" ;;
esac
export TPC=10
cd ~/pyrl_work/examples
ls work/data | grep "^${PREFIX}_" | grep -v _copy | grep -v test | grep -E "${RUN_FILTER:-.}" > /tmp/refresh_$PREFIX.txt
xargs -P $NP -I{} bash -c "bash ~/rat_pretraining/rollout_extract.sh {} $MODEL $P $PR $SCR $OUT > ~/pyrl_work/logs/rx_{}.log 2>&1" < /tmp/refresh_$PREFIX.txt
echo done > ~/pyrl_work/logs/REFRESH_${PREFIX}_DONE
