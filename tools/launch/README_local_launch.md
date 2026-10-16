# The slimmed local `tscost` run (Apple Silicon Mac, `/Users/s2758080/pyrl`)

Written 16 Oct 2026 on `mvm-lin-0071`, where the run could not be started: Marino was holding all
32 cores with a rolling `pagan_hist_evo.py` sweep.

**There are two runs, and this file is about the second one.**

| | full run | slimmed local run |
|---|---|---|
| queue | `queue_tscost.sh` | `queue_tscost_local.sh` |
| where | mvm-lin-0071, when it frees up | the Mac, now |
| arms | `tc55off` `tc55fix` `tc55cho` `tc55per` | `tl55fix` `tl55cho` `tl55per` |
| networks | 32 | 24 |
| validation batch | 720 | 360 |
| explorer class | `e11` | `e12` |
| cost | ~56 core-hours | **~28 core-hours** |

The full run **stays queued for 0071 exactly as specified** — nothing in the slimming touches it.
The two are separate runs on separate samples, not two halves of one: the same weight seed gives
the same starting weights, but validation draws consume the same stimulus RNG stream as training
draws, so a 360-trial batch puts every later trial out of step with the 720-trial one. They must
not be pooled, which is why the arm tags, run folders and explorer classes differ.

The scripts are portable — `$HOME/pyrl` resolves to `/Users/s2758080/pyrl`, so no path edits.
Everything machine-specific goes through environment variables (`PYRL_PYTHON`, `JOBS`, `MAX_ITER`,
`NVAL`, `NICE`, `PYRL_REPO`).

---

## 1. What was cut, and what it costs

**Dropped: the `off` arm (no duration schedule).** The smoke test already answered it. At the
hand-off to the task phase it goes from 0.996 of trials reaching a choice to 0.000 within one
checkpoint, essentially all of it *mid-window* withdrawal at a median step of 42 falling to 31 —
the network has learned not to refuse a trial outright but cannot hold a 130-step window. The
control existed to show that jumping straight to the full window destroys engagement; it does, and
24 more networks would not say it better. The smoke logs are the evidence
(`logs/weak/smoke_tcoff.log`).

**Halved: the validation batch, 720 → 360.** This is where the saving is. A checkpoint's
validation batch is the larger part of the per-update cost — 720/5 = 144 validation trials per
update against 72 training trials — so halving it takes a run from 2.25 to 1.50 s/update, a third
off. Passed as `--config n_validation=360`, which `do.py train` applies to training only and
records in the savefile.

**What the smaller batch costs, accepted deliberately.** The curriculum gates read one validation
batch, so their precision halves: with ~359 decided trials the standard error of accuracy at
chance is 0.026 against 0.019 at 720. A network *exactly* at chance therefore clears the `perf`
arm's 0.55 gate on **2.9%** of checkpoints rather than 0.4% — about 16 growths' worth of noise over
560 checkpoints, against the 25 growths that complete the schedule. So `tl55per` may drift most of
the way to the full window without ever being reliably above chance. Read its schedule trace
alongside its accuracy, never on its own. The P(choice) gate is unaffected in practice, since
P(choice) sits near 1.0 once pre-training is done.

## 2. The budget

| quantity | full run | slim local run | where it comes from |
|---|---|---|---|
| trials per update | 216 | 144 | 72 training + `n_validation`/5 |
| seconds per update | 2.25 | 1.50 | at ~10.4 ms/trial |
| hours per run | 1.75 | 1.16 | × 2,800 updates |
| runs | 32 | 24 | arms × 8 seeds |
| **core-hours** | **~56** | **~28** | single-threaded, embarrassingly parallel |
| wall at `JOBS=8` | ~7 h | **~3.5 h** | |
| wall at `JOBS=4` | ~14 h | ~7 h | |
| disk, training output | ~3.9 GB | **~2.9 GB** | 122 MB/run, measured on `share_rl_rnn/runs/pagan_hist_effort2_g06e255curr_w001` (a 20 MB `<run>.pkl` + 561 checkpoints at ~150 KB) |
| disk, if you roll out too | +10–20 GB scratch | | then 12 MB/run of `records/`; mechanism 3.0 MB, activity 0.6 MB (means over the 326/277/177 in the share) |

The 10.4 ms/trial is measured from `logs/pagan_hist_hist100_w010.log` on *this* box: 230 updates in
17m12s at 432 trials/update. (An earlier draft of this file said ~96 core-hours for the full run;
that was wrong — it multiplied 32 runs by a loose "~3 h" instead of the 1.75 h the per-update
figure actually gives. The table above is the corrected arithmetic.)

Treat it as the right order of magnitude, not a promise: it is a different model on different
hardware. An M-series performance core is likely faster per core than this Xeon, but under
Rosetta 2 (section 3) some of that is given back. **Measure it yourself** — the smoke test in
section 4 prints elapsed time per checkpoint and pins the real number in two minutes.

Thermals matter more than the arithmetic. 8 cores at 100% for several hours on a laptop will
throttle; keep it plugged in and well ventilated, and expect to exceed the ideal wall time.

## 3. The environment, which is the real risk

`pyrl` is Python 2.7 + Theano 0.8.2. On this box that is
`/home/echang/anaconda3/envs/pyrl/bin/python` (2.7.15, numpy 1.15.4), working with no special
`THEANO_FLAGS`. **There is no Python 2.7 build for `osx-arm64`** — Python 2.7 predates Apple
Silicon and neither `defaults` nor `conda-forge` ships one. Options, best first:

1. **`osx-64` conda env under Rosetta 2.** Python 2.7, numpy 1.15 and a compiler all exist for
   `osx-64`. The only route I would expect to work first time:
   ```bash
   softwareupdate --install-rosetta --agree-to-license      # if not already present
   CONDA_SUBDIR=osx-64 conda create -n pyrl27 python=2.7 numpy=1.15 scipy matplotlib
   conda activate pyrl27
   conda config --env --set subdir osx-64                   # keep the env on osx-64
   pip install 'theano==0.8.2'
   ```
   `tools/launch/pyrl27_explicit.txt` is Marino's `conda create --file` spec — a useful version
   reference, but it is `linux-64` and cannot be used directly.
2. **An `arm64` Linux VM** (Lima, UTM, OrbStack) — native speed, but Python 2.7 and numpy 1.15
   would have to be built from source for `aarch64` and Theano 0.8.2 there is untested. Only worth
   it if Rosetta performance disappoints badly.
3. **`linux/amd64` Docker/colima** — QEMU emulation, 5–20× slower. Not viable.

Theano needs a working C++ compiler; without one it silently falls back to a pure-Python backend
orders of magnitude slower. **Check for that explicitly** — if the smoke takes minutes per
checkpoint instead of seconds, that is what happened. Install the Command Line Tools
(`xcode-select --install`), and if needed point Theano at the conda compiler with
`THEANO_FLAGS="cxx=/path/to/clang++,base_compiledir=$HOME/.theano"`.

## 4. The launch sequence

```bash
cd ~/pyrl
export PYRL_PYTHON=$HOME/miniforge3/envs/pyrl27/bin/python     # whatever section 3 produced

# (a) the offline tests: seconds, no Theano, prove the model survived the transfer
$PYRL_PYTHON tests/test_pagan_hist_tscost.py
for m in off fixed choice perf; do $PYRL_PYTHON tests/test_pagan_hist_tscost_ramp.py $m; done

# (b) the smoke test: ~100 s per mode, proves Theano works and pins the per-update cost
MAX_ITER=60 bash tools/launch/smoke_tscost.sh

# (c) one real run, to confirm naming and timing before committing the budget
JOBS=1 NVAL=360 bash tools/launch/launch_weak_start.sh tscost tl55cho 1
#     -> examples/work/data/pagan_hist_tscost_g06tl55cho_w001/, log in logs/weak/
#     ...then Ctrl-C, and delete that folder and logs/weak/DONE_g06tl55cho

# (d) the run
JOBS=8 NICE="" nohup bash tools/launch/queue_tscost_local.sh > logs/weak/queue_local.log 2>&1 &
```

Step (a) must pass before (b): it is the cheapest check that nothing was corrupted in transfer and
it needs no Theano. Expected smoke output, from this box:

- the window held at 20 ms through six pre-training checkpoints, the choice fraction climbing
  0.42 → 1.00;
- `fixed` and `choice` stepping 50 → 100 → 150 … with the choice fraction staying above 0.996;
- `perf` holding at 50 ms (accuracy 0.47–0.52, under its 0.55 gate) — **expected and accepted**,
  see section 6;
- `off` collapsing at the hand-off, if you smoke it: that arm is not in the local run but
  `smoke_tscost.sh` still covers all four modes, which is a useful check in itself.

`queue_tscost_local.sh` runs the three arms in sequence with `JOBS` seeds in flight, and skips any
arm whose `logs/weak/DONE_g06<arm>` marker exists, so it survives an interruption, a reboot or a
closed lid. To force an arm to re-run, delete its marker *and* its run folders. Set `NICE=""` on a
dedicated laptop; the default `nice -n 10` exists for the shared box.

## 5. What is missing and must be checked locally

Nothing below can be determined from `mvm-lin-0071`, in rough order of how much it could change
the plan:

1. **Does a Python 2.7 + Theano 0.8.2 env build on the Mac at all** (section 3), with the C
   backend active rather than the Python fallback. The one real go/no-go.
2. **Chip, performance-core count, RAM** — sets `JOBS` and the wall time:
   `sysctl -n machdep.cpu.brand_string hw.perflevel0.physicalcpu hw.memsize`. The scripts
   auto-detect performance cores and cap `JOBS` at 8, but the right value depends on what else the
   machine is doing. Each run holds roughly 200–400 MB resident, so 8 concurrent want ~3 GB plus
   headroom.
3. **Free disk**: ≥ 3 GB to train, ≥ 25 GB if the rollouts are done locally too.
4. **macOS version**, for Rosetta 2 availability.
5. **Whether this branch is on the Mac.** It is committed in `/home/echang/pyrl` on branch
   `tscost-engagement-cost`; the remote is `github.com/EliseChang/pyrl`. It has **not** been
   pushed — that is an outward action left to you. Either push from here
   (`git push -u origin tscost-engagement-cost`) and pull on the Mac, or copy the files in
   section 7. Compare `git log --oneline -1` on both sides.
6. **Whether the analysis chain runs locally.** For training, not needed — the queue only trains.
   The rollout → `records/` → explorer chain needs a *separate* Python 3 env with numpy, scipy and
   matplotlib, and `tools/measurement/rollout_extract.sh` still calls its siblings by Marino's
   `~/rat_pretraining/` paths, so those need adjusting before it runs anywhere else
   (`README_for_elise.md` section 8). All the pieces are present in `tools/measurement/`,
   `solution_space.py` included — README section 5 wrongly says that one is missing. Simplest
   division of labour: **train locally, copy the 24 run folders back, measure on a Linux box.**
7. **Whether 0071 frees up**, which changes the calculus for the *full* run (28 core-hours is
   ~1 h of wall time on an idle 32-core box): `ps -eo pcpu,args | grep do.py`.

## 6. Decisions already settled, so they are not relitigated locally

- per-step cost `0.55/130`, so the full window costs exactly 0.55;
- `PAGAN_STIM0=50` ms, four input bins, accuracy ceiling 0.833 (0.653 at one bin);
- `PAGAN_PRESTIM=20` ms, the pre-training window is one presentation step, so the hand-off steps
  20 → 50 ms;
- `+50 ms` per checkpoint, 25 growths to the full length;
- `PAGAN_RAMP_PCORRECT=0.55`, deliberately not 0.50: at 720 validation trials a chance network
  clears 0.50 on 48.5% of checkpoints, so the schedule would complete on validation noise alone.
  The `perf` arm may therefore hold a short window for a long time, which is accepted — a stalled
  performance gate is an informative result, not a failure. (At 360 trials even 0.55 is cleared on
  2.9% of chance checkpoints; see section 1.)
- `DECLINE` terminates at any timestep and the accrued cost is never refunded;
- `R_ABORTED=0`, no timeout penalty (timeouts are 0.000 of trials in practice anyway);
- initial action probabilities not handcrafted (`bout` at the pyrl default).

## 7. The files these runs consist of

| file | |
|---|---|
| `examples/models/pagan_hist_tscost.py` | the model, shared by both runs |
| `tests/test_pagan_hist_tscost.py` | reward structure, 44 assertions |
| `tests/test_pagan_hist_tscost_ramp.py` | the duration schedule, one mode per process |
| `tools/launch/launch_weak_start.sh` | the weak-start trainer: portable, `JOBS` pool, `NVAL` |
| `tools/launch/queue_tscost.sh` | the **full** run, four arms, 720 — for 0071, unchanged |
| `tools/launch/queue_tscost_local.sh` | the **slim local** run, three arms, 360 |
| `tools/launch/smoke_tscost.sh` | the environment check |
| `tools/launch/README_local_launch.md` | this file |
| `tools/measurement/build_network_explorer.py` | `--tree elise`; `tc55*` → `e11`, `tl55*` → `e12` |
| `README_for_elise.md` | sections 9 (the runs) and 10 (the two trees) |
