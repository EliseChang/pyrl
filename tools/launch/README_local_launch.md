# Launching the `tscost` run locally (Apple Silicon Mac, `/Users/s2758080/pyrl`)

Written 16 Oct 2026 on `mvm-lin-0071`, where the run could not be started: Marino was holding all
32 cores with a rolling `pagan_hist_evo.py` sweep. The model, launch scripts and tests are
committed and verified; this file is what the local session needs.

The scripts are portable — `$HOME/pyrl` resolves to `/Users/s2758080/pyrl` on the Mac, so no path
edits are needed. Everything machine-specific goes through environment variables
(`PYRL_PYTHON`, `JOBS`, `MAX_ITER`, `NICE`, `PYRL_REPO`).

---

## 1. The cost of the run, so the decision is informed

| quantity | value | where it comes from |
|---|---|---|
| work | **~96 core-hours**, single-threaded, embarrassingly parallel | 32 runs × 2,800 updates × ~2.3 s/update |
| per update | ~2.3 s | 216 trials/update (72 training + 720 validation every 5 updates) at ~10.4 ms/trial, measured from `logs/pagan_hist_hist100_w010.log` (230 updates in 17m12s at 432 trials/update) on this box |
| per run | ~1.8 h on an uncontended core | 2,800 × 2.3 s |
| wall time | ~12 h at `JOBS=8`, ~24 h at `JOBS=4`, ~48 h at `JOBS=2` | 96 core-hours ÷ JOBS |
| disk, training output | **~3.9 GB** (122 MB/run: a 20 MB `<run>.pkl` + 561 checkpoints at ~150 KB) | measured on `share_rl_rnn/runs/pagan_hist_effort2_g06e255curr_w001` |
| disk, if you also roll out | +10–20 GB of scratch, then 12 MB/run of `records/` | mean record npz over the 326 in the share is 12.1 MB; mechanism 3.0 MB; activity 0.6 MB |

The per-update figure is from a *different* model on *this* hardware, so treat it as the right
order of magnitude, not a promise. An M-series performance core is likely faster per-core than
this Xeon, but under Rosetta 2 (section 2) some of that is given back. **Measure it yourself**: the
smoke test in section 3 prints elapsed time per checkpoint, which pins the real number in two
minutes.

Thermals matter more than the arithmetic. 8 cores at 100% for 12 hours on a laptop will throttle;
keep it plugged in, lid open or on a stand, and expect the wall time to exceed the ideal. If the
Mac has only 4 performance cores, prefer running **two arms locally and two elsewhere** over a
48-hour laptop job.

## 2. The environment, which is the real risk

`pyrl` is Python 2.7 + Theano 0.8.2. On this box that is
`/home/echang/anaconda3/envs/pyrl/bin/python` (2.7.15, numpy 1.15.4), and it works with no special
`THEANO_FLAGS`. **There is no Python 2.7 build for `osx-arm64`** — Python 2.7 predates Apple
Silicon and neither `defaults` nor `conda-forge` ships one. The options, best first:

1. **`osx-64` conda env under Rosetta 2.** Python 2.7, numpy 1.15 and a compiler all exist for
   `osx-64`, and Rosetta 2 runs them. This is the only route I would expect to work first time:
   ```bash
   softwareupdate --install-rosetta --agree-to-license      # if not already present
   CONDA_SUBDIR=osx-64 conda create -n pyrl27 python=2.7 numpy=1.15 scipy matplotlib
   conda activate pyrl27
   conda config --env --set subdir osx-64                   # keep the env on osx-64
   pip install 'theano==0.8.2'
   ```
   `tools/launch/pyrl27_explicit.txt` is Marino's `conda create --file` spec for his Linux env —
   useful as a version reference, but it is `linux-64` and cannot be used directly.
2. **An `arm64` Linux VM** (Lima, UTM, OrbStack) — native speed, but Python 2.7 and numpy 1.15
   would have to be built from source for `aarch64`, and Theano 0.8.2 on `aarch64` is untested.
   Only worth it if Rosetta performance disappoints badly.
3. **`linux/amd64` Docker/colima** — works, but QEMU emulation is 5–20× slower. Not viable for
   96 core-hours.

Theano needs a working C++ compiler; if it cannot find one it silently falls back to a pure-Python
backend that is orders of magnitude slower. **Check for that explicitly** — if the smoke test takes
minutes per checkpoint instead of seconds, that is what has happened. Install the Command Line
Tools (`xcode-select --install`) and if needed point Theano at the conda compiler via
`THEANO_FLAGS="cxx=/path/to/clang++,base_compiledir=$HOME/.theano"`.

## 3. The launch sequence

```bash
cd ~/pyrl
export PYRL_PYTHON=$HOME/miniforge3/envs/pyrl27/bin/python     # whatever section 2 produced

# (a) the offline tests: seconds, no Theano, prove the model is intact after the transfer
$PYRL_PYTHON tests/test_pagan_hist_tscost.py
for m in off fixed choice perf; do $PYRL_PYTHON tests/test_pagan_hist_tscost_ramp.py $m; done

# (b) the smoke test: ~100 s per mode, proves Theano works and pins the per-update cost
MAX_ITER=60 bash tools/launch/smoke_tscost.sh

# (c) one real run, to confirm naming and timing before committing 96 core-hours
JOBS=1 bash tools/launch/launch_weak_start.sh tscost tc55cho 1
#     -> examples/work/data/pagan_hist_tscost_g06tc55cho_w001/, log in logs/weak/
#     ...then Ctrl-C, delete that folder and logs/weak/DONE_g06tc55cho

# (d) the run
JOBS=8 NICE="" nohup bash tools/launch/queue_tscost.sh > logs/weak/queue.log 2>&1 &
```

Step (a) must pass before (b): it is the cheapest check that the transfer did not corrupt
anything, and it needs no Theano. Expected output of (b), from this box:

- presentation held at 20 ms through six pre-training checkpoints, the choice fraction climbing
  0.42 → 1.00;
- `off` jumping to 1,300 ms at the hand-off and engagement collapsing to 0.000, essentially all of
  it mid-window withdrawal at a median step in the 30s–40s;
- `fixed` and `choice` stepping 50 → 100 → 150 … with the choice fraction staying above 0.996;
- `perf` holding at 50 ms (accuracy 0.47–0.52, under its 0.55 gate). **Expected and accepted** —
  see section 5.

`queue_tscost.sh` runs the four arms in sequence with `JOBS` seeds in flight, and skips any arm
whose `logs/weak/DONE_g06<arm>` marker exists, so it is safe to re-run after an interruption,
a reboot or a closed lid. To force an arm to re-run, delete its marker *and* its run folders.

Set `NICE=""` on a dedicated laptop (the default `nice -n 10` exists for the shared box).

## 4. What is missing and must be retrieved locally

Nothing here can be determined from `mvm-lin-0071`. In rough order of how much they could change
the plan:

1. **Does a Python 2.7 + Theano 0.8.2 env build on the Mac at all** (section 2), and is the Theano
   C backend active rather than the Python fallback. This is the one real go/no-go.
2. **The Mac's chip, performance-core count and RAM** — sets `JOBS` and therefore the wall time.
   `sysctl -n machdep.cpu.brand_string hw.perflevel0.physicalcpu hw.memsize`. The scripts
   auto-detect performance cores and cap `JOBS` at 8, but the *right* value depends on whether the
   machine is otherwise in use. Each run holds roughly 200–400 MB resident, so 8 concurrent runs
   want ~3 GB plus headroom.
3. **Free disk**: ≥ 4 GB for training output alone, ≥ 25 GB if the rollouts are done locally too.
4. **macOS version**, for Rosetta 2 availability.
5. **Whether this branch is on the Mac.** It is committed here as `tscost-engagement-cost` in
   `/home/echang/pyrl`, and the remote is `github.com/EliseChang/pyrl`. It has **not** been
   pushed — pushing is an outward action I have not taken. Either push it from here
   (`git push -u origin tscost-engagement-cost`) and pull on the Mac, or copy the eight files in
   section 6 across. Check `git log --oneline -1` on both sides to confirm they match.
6. **Whether the analysis chain will run locally.** For training, no: `queue_tscost.sh` only
   trains. The rollout → `records/` → explorer chain needs a *separate* Python 3 env with numpy,
   scipy and matplotlib, and `tools/measurement/rollout_extract.sh` still calls its siblings by
   Marino's `~/rat_pretraining/` paths, so it needs its paths adjusted before it will run anywhere
   else (README_for_elise.md section 8 flags this). The pieces are all present in
   `tools/measurement/` — including `solution_space.py`, which README section 5 wrongly says is
   missing. The simplest division of labour is: **train locally, copy the 32 run folders back,
   measure on a Linux box.**
7. **Whether `mvm-lin-0071` frees up in the meantime**, which would make all of this moot —
   `ps -eo pcpu,args | grep do.py`. 96 core-hours is ~3 hours of wall time on an idle 32-core box
   against ~12 on the laptop.

## 5. Decisions already settled, so they are not relitigated locally

- per-step cost `0.55/130`, so the full window costs exactly 0.55;
- `PAGAN_STIM0=50` ms, four input bins, accuracy ceiling 0.833 (0.653 at one bin);
- `PAGAN_PRESTIM=20` ms, the pre-training window is one presentation step, so the hand-off steps
  20 → 50 ms;
- `+50 ms` per checkpoint, 25 growths to the full length;
- `PAGAN_RAMP_PCORRECT=0.55`. Deliberately not 0.50: a network exactly at chance clears 0.50 on
  48.5% of checkpoints (SE 0.019 on ~718 decided trials), so the schedule would complete on
  validation noise alone in ~52 checkpoints; 0.55 is cleared on 0.4%. The `perf` arm may therefore
  hold a short window for a long time, which is accepted — a stalled performance gate is an
  informative result, not a failure;
- `DECLINE` terminates at any timestep and the accrued cost is never refunded;
- `R_ABORTED=0`, no timeout penalty (timeouts turn out to be 0.000 of trials anyway);
- initial action probabilities not handcrafted (`bout` at the pyrl default).

## 6. The files this run consists of

| file | |
|---|---|
| `examples/models/pagan_hist_tscost.py` | the model |
| `tests/test_pagan_hist_tscost.py` | reward structure, 44 assertions |
| `tests/test_pagan_hist_tscost_ramp.py` | the duration schedule, one mode per process |
| `tools/launch/launch_weak_start.sh` | the weak-start trainer, portable, with a `JOBS` pool |
| `tools/launch/queue_tscost.sh` | the four arms, resumable |
| `tools/launch/smoke_tscost.sh` | the environment check |
| `tools/launch/README_local_launch.md` | this file |
| `tools/measurement/build_network_explorer.py` | `--tree elise`, the `tc55*` arms as class `e11` |
| `README_for_elise.md` | sections 9 (the run) and 10 (the two trees) |
