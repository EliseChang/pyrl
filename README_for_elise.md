# How a network gets measured, and how to make yours slot into the explorer

This file didn't exist on disk, but the `models/scripts/` folder of `share_rl_rnn` carries a
complete, working measurement pipeline for this `pyrl` stack, built by Marino (mpagan) around
the `pagan_hist*` tasks. This doc reverse-engineers that pipeline from the scripts themselves
(`build_network_explorer.py`, `pyrl_mechanism.py`, `activity_summary.py`,
`psytrack_on_network.py`, `explorer_lib.py`, and the `launch_*.sh` / `rollout_*.sh` /
`*_and_dispatch.sh` orchestration around them) so a network trained from this repo lands in the
same `explorer/` HTML pages as everything else.

**Where the scripts actually are**: the pipeline scripts described below are bundled in this
repo under [`tools/measurement/`](tools/measurement/) (`build_network_explorer.py`,
`explorer_lib.py`, `glm_vs_psytrack.py`, `pyrl_mechanism.py`, `solution_space.py`,
`activity_summary.py`, `psytrack_on_network.py`, and the `rollout_extract.sh` /
`rollout_activity.sh` / `complete_records.sh` / `refresh_records.sh` / `collect_and_build.sh` /
`launch_activity_0071.sh` / `launch_mech_0071.sh` wrappers) — on Marino's machines they live in
a separate `rat_pretraining` repo (`~/rat_pretraining/*.py`), and that's still the path baked
into some of the shell scripts' own calls to their siblings (see the caveat at the end of
section 8), but the files themselves are copied here so you don't need that other repo to read
or run them. Marino's `models/pyrl_work`'s own top-level training-launch scripts (hostnames and
paths specific to his HPC machines, but the exact commands that produced every network already
in the explorer) are bundled under [`tools/launch/`](tools/launch/). Left out of both: ~100
one-off/debug scripts that were also in `models/scripts/` (not part of the documented pipeline),
and five small patch/backup files from `models/pyrl_work`'s top level (`model.py`,
`patch_model.py`, `patch_model2.py`, `patch_pg.py`, `patch_pg2.py`) — these were Marino's own
tools for text-patching `pyrl/model.py`/`pyrl/policygradient.py` to add the `record_outcome`/
`set_validation`/resume-override hooks, and that patching is superseded now that those hooks are
merged directly into this repo's `pyrl/model.py` and `pyrl/policygradient.py`.

## 1. The five things a network needs to produce

| stage | what it is | where it comes from | npz file |
|---|---|---|---|
| training | weights at every checkpoint + training-history curve | `pyrl` itself (`do.py ... train`) | `examples/work/data/<run>/<run>.pkl`, `examples/work/data/<run>/checkpoints/iter_NNNNNN.pkl` |
| rollout (behaviour) | per-trial choices/rewards at each checkpoint, replayed sequentially so trial-history inputs are real | `do.py run analysis/pagan_hist.py trials-b` | `$SCRATCH/work/pyrl/examples/<run>/checkpoints/iter_NNNNNN/trials_behavior.pkl` |
| **records/** | compact PsyTrack/GLM regressors extracted from the behaviour rollout | `psytrack_on_network.py --extract` | `<records>/<run>_record.npz` |
| rollout (activity) | per-step hidden states of actor + critic, action probabilities, critic value | `do.py run analysis/pagan_hist.py trials-a` | `$SCRATCH/work/pyrl/examples/<run>/checkpoints/iter_NNNNNN/trials_activity.pkl` |
| **activity/** | compact critic/actor measures per checkpoint (value trajectories, participation ratio, evidence/choice axes, PCA basis) | `activity_summary.py` | `<activity>/<run>_activity.npz` |
| **mechanism/** | fixed points per context, leading eigenvalue, selection-vector angle, the Nature-2025 triangle, per checkpoint | `pyrl_mechanism.py` (reuses `solution_space.py`) | `<mechanism>/<run>_policy_mechanism.npz` |

`build_network_explorer.py` reads the four npz families above (plus `<run>.pkl` and the
`checkpoints/iter_*.pkl` for weight norms) and writes the static `explorer/` site. Nothing in
the explorer talks to a live training process or a database — it is all pre-computed npz +
pickle on disk, cached per-run under `<site>/cache/<run>.npz`.

## 2. Training, so the rollouts exist

Train with `examples/do.py`, from the repo root with `PYTHONPATH` set to it:

```bash
python examples/do.py --suffix _curr_w001 \
    --config weight_seed=1 --config stim_seed=1 --config action_seed=1 \
    --config Wrec_init=gaussian --config gates=False --config f_hidden=tanh \
    --config tau=100 --config rho=1.15 --config baseline_rho=1.15 \
    --config p0=0.5 --config baseline_p0=1 --config var_rec=0 --config baseline_var_rec=0 \
    --config L2_x=0.001 --config lr=0.004 --config baseline_lr=0.004 --config beta_entropy=0 \
    --config Win=1 --config max_iter=2800 --config checkfreq=5 \
    examples/models/pagan_hist_curriculum.py train
```

Points that matter for the downstream pipeline, not just for training:

- **`--config checkfreq=5`** (every 5 updates) is what the whole pipeline assumes when it talks
  about "checkpoints"; with `n_gradient=72` trials/update, one checkpoint = 360 training trials
  (`N_GRADIENT = 72` is hard-coded in `build_network_explorer.py`). If you train at a different
  `checkfreq`/`n_gradient`, the x-axes (`iter * N_GRADIENT`) will be wrong until that constant is
  updated too.
- **`--suffix`** is what becomes the run name on disk (`<modelfile stem><suffix>`) — see the
  naming convention in section 5, which the explorer's `ARMS` table pattern-matches on.
- **`--config weight_seed=N --config stim_seed=N --config action_seed=N`**: the paired arms
  (pretrained vs no_pretrain, antibias vs its control, etc.) are built by using the *same* three
  seeds across arms, so they start from literally the same initial weights — only an
  environment-variable switch (below) differs between the two runs of a pair.
- **Task-specific switches are environment variables, not `--config`**, because `do.py
  --config` can only set keys already in `model.config`, not a task module's own globals
  (`do.py --config` would otherwise need a code change per experiment). Each `pagan_hist_*`
  model documents its own switches in its docstring; the ones used across the current library:

  | variable | meaning | used by |
  |---|---|---|
  | `PAGAN_PRETRAIN=0` | skip the nose-poke pre-training; start straight in the task phase (the matched control) | `pagan_hist_curriculum.py` and everything built on it |
  | `PAGAN_PHASE=task` | force the task phase at import (needed because `do.py run` re-imports the model fresh — a rollout must run on the TASK distribution regardless of which checkpoint is loaded) | curriculum and descendants |
  | `PAGAN_ANTIBIAS=0` | turn the closed-loop anti-bias schedule off (rollouts must measure every arm on the same, schedule-free, distribution) | `pagan_hist_antibias.py` |
  | `PAGAN_AB_BETA`, `PAGAN_AB_TAU` | anti-bias schedule parameters (lab default: beta 5, tau 50; beta -5 inverts it) | `pagan_hist_antibias.py` |
  | `PAGAN_EFFORT=0.55` | total cost of attending one trial | `pagan_hist_effort.py`, `pagan_hist_effort2.py` |
  | `PAGAN_STAGES=full` (`none`\|`short`\|`full`) | which stage-curriculum arm to train | `pagan_hist_stages.py` |
  | `PAGAN_STAGE=10` | force the full-task stage for analysis rollouts (so every arm is measured on the same distribution) | `pagan_hist_stages.py` |
  | `PAGAN_QAB_REWARD=1` | quadrant anti-bias reward scaling on/off | `pagan_hist_stages.py` |

  The closed-loop schedules (`PAGAN_ANTIBIAS`, the stage curriculum's promotion logic) depend on
  `record_outcome`/`set_validation` hooks added to `pyrl/model.py` and
  `pyrl/policygradient.py` on 11 Oct 2026 (now merged into this repo) — without them a task
  module that defines those methods is simply never told the outcome of a trial.

Training writes `examples/work/data/<run>/<run>.pkl` (has `training_history`, a list of
per-checkpoint dicts with `iter`, `perf`, `mean_reward`) and
`examples/work/data/<run>/checkpoints/iter_NNNNNN.pkl` (has `iter`, `policy_params`,
`baseline_params`, `config`) — both consumed directly by `build_network_explorer.py`'s
`compute()` and by `pyrl_mechanism.py`. A run stops early when `terminate()` is satisfied
(0.95 correct, 99% decided, on a task-phase validation batch); `build_network_explorer.py`
reads `criterion_iter` back out of this to report whether/when a network "learned".

## 3. Rolling out and extracting `records/`

Behaviour rollouts, every checkpoint (or a subset via `iters=`), replayed sequentially per
checkpoint so the trial-history input channels are real trial-by-trial history, not shuffled:

```bash
PAGAN_PHASE=task PAGAN_ANTIBIAS=0 PAGAN_PRETRAIN=0 \
  python examples/do.py --suffix _curr_w001 examples/models/pagan_hist_curriculum.py \
      run examples/analysis/pagan_hist.py trials-b 25 iters=all
```

`25` is trials per condition (`trials_per_condition`); `iters=all` rolls out every saved
checkpoint (`iters=5,10,15` rolls out specific ones). This writes
`$SCRATCH/work/pyrl/examples/<run>/checkpoints/iter_NNNNNN/trials_behavior.pkl` — a Python-2
pickle `[trials, A, R, M, perf]` (`A`/`R`/`M` are `(Tmax, n_trials)`; `trials` is a list of
per-trial dicts with `context`, `integral_loc`, `integral_freq`, `correct_side`,
`prev_outcome`). `$SCRATCH` defaults to `~/scratch` if unset.

Then extract the compact record (this is the only thing `build_network_explorer.py` reads for
behaviour/strategy — not the raw rollout):

```bash
python psytrack_on_network.py --extract --run <run> \
    --scratch /path/to/scratch --out /path/to/records
```

This re-derives, per trial with a choice: `bias`, `rel` (z-scored signed evidence of the cued
feature, + = right), `irr` (z-scored signed evidence of the other feature), `histChoice`
(previous choice, +1 right), `wsls` (previous choice x previous reward), plus raw fields
(`raw_context`, `raw_e_loc`, `raw_e_frq`, `raw_rel_raw`, `raw_irr_raw`, `raw_reward`,
`raw_ck_iters`, `raw_ck_n_total`, `raw_ck_n_choice` — the last three give the attempt rate for
effort-model runs) — written to `<out>/<run>_record.npz` as `y`, `session` (checkpoint index per
trial), `iters`, and `X_<name>` per regressor. `build_network_explorer.RECORDS` must point at
this `--out` directory.

`rollout_extract.sh` in `models/scripts/` wraps both steps with the checkpoint-subsampling
policy actually used for the long runs (every checkpoint to iteration 300, then every 20th, plus
the last) and is idempotent (skips rollouts that already exist on disk).

Separately, `psytrack_on_network.py` (without `--extract`, or with `--record <path>`) fits an
actual PsyTrack model to the record and writes `<out>/<run>_psytrack.npz` +
`<out>/<run>_psytrack.png` — optional (used for the PsyTrack-vs-ridge-GLM comparison page), not
required for the main per-network explorer page, which fits its own per-checkpoint ridge
logistic GLM directly from the record (`glm_vs_psytrack.ridge_logistic`).

## 4. Activity rollouts and `activity/`

Activity rollouts save hidden states too (43 MB per checkpoint vs a few MB for behaviour), so
only a handful of checkpoints are usually worth it (`rollout_activity.sh` picks 12, spread over
training, always including the first and last):

```bash
PAGAN_PHASE=task PAGAN_ANTIBIAS=0 PAGAN_STAGE=10 PAGAN_PRETRAIN=0 \
  python examples/do.py --suffix _curr_w001 examples/models/pagan_hist_curriculum.py \
      run examples/analysis/pagan_hist.py trials-a 2 iters=5,50,150,...
```

Writes `.../checkpoints/iter_NNNNNN/trials_activity.pkl`: `[trials, U, Z, Z_b, A, R, M, perf,
r_p, r_v]`, where `r_p`/`r_v` are `(T, n_trials, N)` hidden states of the policy and baseline
(critic) networks. Then:

```bash
python activity_summary.py --run <run> --scratch /path/to/scratch --out /path/to/activity
```

reduces each checkpoint's rollout to critic measures (`v_start`, `v_end`, `pe_terminal`,
`v_vs_abs_evidence`, `v_vs_signed_evidence`, `v_side_gap`, `critic_pr`, `v_mean_t_hit/miss`) and
actor measures (`actor_pr`, `pca_mean`/`pca_comp`/`pca_var`, `choice_axis`, `evidence_axis`,
`evidence_axis_ctx`, `angle_choice_evidence`, `angle_evidence_ctx`, `critic_reads_evidence`,
`traj_pc`), written to `<out>/<run>_activity.npz` — one row per checkpoint rolled out.
`build_network_explorer.ACT` must point at this `--out` directory.

## 5. Mechanism: fixed points, eigenvalues, the triangle

```bash
python pyrl_mechanism.py --run <run> --max-ckpts 24 \
    --data examples/work/data --activity-scratch /path/to/scratch \
    --out /path/to/mechanism
```

For each of up to `--max-ckpts` checkpoints (subsampled evenly if there are more), this maps
`pyrl`'s leaky-RNN policy params onto `solution_space`'s discrete map
(`wR = Wrec^T, wI = Win^T, bR = bin`, readout `wO = Wout[:,RIGHT] - Wout[:,LEFT]`), searches for
slow points per context (LOC-only and FREQ-only constant input; trained RL networks essentially
never have exact fixed points at the strict tolerance, so this is a slow-point search at
`FP_TOL = 1e-2`, seeded from states the network actually visited if `--activity-scratch` points
at matching `trials_activity.pkl` rollouts — hence run activity rollouts first if you want this
warm-starting), and runs `solution_space.decompose()` to get the leading eigenvalue (mapped back
through the leak: `eig_leaky = 1 - alpha + alpha * eig_discrete`), the selection-vector angle
between contexts, and the triangle coordinates. Writes
`<out>/<run>_policy_mechanism.npz`. `build_network_explorer.MECH` must point at this `--out`
directory. (`--net baseline` runs the same analysis on the critic instead of the actor, writing
`<run>_baseline_mechanism.npz`.)

`solution_space.py` (the actual fixed-point solver / triangle decomposition) is not part of this
repo — `pyrl_mechanism.py` expects it either next to itself or at `../../solution_space`
relative to itself, inside the `gatsby_talk` tree.

## 6. Building the explorer

Once a run has `records/<run>_record.npz`, optionally `activity/<run>_activity.npz` and
`mechanism/<run>_policy_mechanism.npz`, and its `examples/work/data/<run>/` folder is reachable,
rebuild the site:

```bash
OMP_NUM_THREADS=2 python build_network_explorer.py --site /path/to/explorer_site \
    [--only <run_prefix>] [--workers 10]
```

`build_network_explorer.py` hard-codes its four input roots at the top of the file —
`RECORDS`, `DATA` (`~/pyrl_work/examples/work/data`), `MECH`, `ACT` — edit those (or symlink) to
point at wherever your `records/` / `mechanism/` / `activity/` actually live and wherever your
`pyrl` checkout's `examples/work/data` is. It computes everything per-network into
`<site>/cache/<run>.npz` (bump `CACHE_VERSION` in the script if you change what `compute()`
returns, so stale caches aren't silently reused), then writes `index.html` (population table +
sparklines), one page per network (Behaviour/Strategy/Mechanism/Critic, each with a checkpoint
slider), `compare.html`, and `methods.html` via the shared `explorer_lib.Site` builder — a
static site with no server, so figures are PNGs and any data a page needs at runtime is
embedded inline rather than fetched.

## 7. Naming your run so it slots in automatically

`build_network_explorer.py`'s `ARMS` table pattern-matches on the run-name *prefix on disk*
(`<model file stem><suffix>`) to decide the display name, which "arm" it belongs to (e.g.
`pretrained` vs `no_pretrain`), and which `CLASS` section of the site it appears under (`base`,
`antibias`, `effort`, `stages`, `library`, ...). A brand-new experiment needs a new entry in
`ARMS` (and usually a new `CLASSES` entry describing it); an experiment that is a new seed of an
*existing* arm just needs to follow the existing prefix convention so it's picked up for free,
e.g.:

- `pagan_hist_curriculum_curr_w<NNN>` / `pagan_hist_curriculum_ctrl_w<NNN>` — base pretrained /
  no_pretrain pairs, seed `NNN`.
- `pagan_hist_antibias_abcurr_w<NNN>` / `abctrl_w<NNN>` — anti-bias (beta 5), pretrained /
  no_pretrain; `abneg_w<NNN>` / `abnegctrl_w<NNN>` — inverted (beta -5).
- `pagan_hist_effort2_e<CC>curr_w<NNN>` / `e<CC>ctrl_w<NNN>` — effort v2, cost `0.<CC>` (e.g.
  `e240` = cost 0.40, `e255` = 0.55, `e270` = 0.70), pretrained / no_pretrain.
- `pagan_hist_stages_st55short_w<NNN>` / `st55full_w<NNN>` — stage curriculum on the effort-0.55
  network, short (stages 1-6 then 10) / full (stages 1-10 with the quadrant anti-bias) ladder.

i.e. `--suffix _<arm>_w<seed padded to 3 digits>` when training, matching one of the prefixes
above (or a new one you add to `ARMS`/`CLASSES`).

## 8. One-shot refresh, for many runs at once

`tools/measurement/` has orchestration wrappers for doing all of section 3 across every run of a
given model-file prefix on one machine:

- `complete_records.sh <machine> <prefix> <modelfile> [N_PAR]` — make a record for every run
  under that prefix that has none yet.
- `refresh_records.sh <machine> <prefix> <modelfile> [N_PAR]` — re-roll new checkpoints and
  re-extract the record for every run under that prefix (idempotent; existing checkpoint
  rollouts are kept).
- `rollout_activity.sh` / `launch_activity_0071.sh`, `launch_mech_0071.sh` — same idea for
  activity and mechanism.
- `collect_and_build.sh` — on the machine that hosts the explorer: rsync records/mechanism/
  activity (and the run folders themselves) in from wherever they were computed, then rebuild
  the site.

These assume Marino's lab layout (`~/pyrl_work`, `/mnt/big/pyrl_work`, specific hostnames,
`~/rat_pretraining` for their own sibling-script calls — e.g. `rollout_extract.sh` still invokes
`~/rat_pretraining/psytrack_on_network.py` rather than the copy that now sits right next to it in
`tools/measurement/`) and won't run as-is outside it. They're bundled here anyway as the clearest
record of the exact flags (checkpoint subsampling policy, `trials_per_condition`, parallelism)
actually used for the networks already in the explorer, if you want a new run measured the same
way — adjust the hardcoded paths for your own machine before running them.

`tools/launch/` has the actual training-launch scripts for the base/antibias/effort/stages
sweeps (`launch_0071_antibias.sh`, `launch_rebel_effort2*.sh`, etc.) — same caveat: hardcoded to
Marino's machines, but they're the exact `do.py train` invocations (seeds, `--config` overrides,
`PAGAN_*` env vars) that produced every network currently in the explorer.

---

## 9. The per-timestep engagement cost run (`tscost`, 16 Oct 2026)

This is the first run documented in its own explorer tree (section 10). It follows up the
fixation-cost networks (`pagan_hist_fixcost.py` / `pagan_hist_stages3.py`) that Marino tried on
13 Oct and set aside — those are in `share_rl_rnn/models/pyrl_work/examples/models/` but were
deliberately never merged here, and no output from their full launch survives.

**What went wrong with the fixation-cost version, so it is not repeated.** It charged the cost per
timestep of holding fixation, which is the right idea, but it (a) removed the dedicated decline
output and signalled withdrawal with `CHOOSE-LEFT`/`CHOOSE-RIGHT`, and (b) made a stimulus-epoch
`CHOOSE-LEFT`/`RIGHT` an aborting violation — i.e. it restored the fixation requirement that was
troubleshot out of this project early on (inherited from the original pyrl). Failure (a) is fatal
on its own: the two choice actions did double duty as the withdrawal action, so every reward
success that pushed "left" up pushed it up at step 0 too, where it means withdraw, and the policy
collapsed into declining everything (its smoke log shows P(choice) falling 0.656 → 0.160 *during*
training). There is also a third, previously unrecorded defect: those arms were launched with
`PAGAN_STIM0=10`, and at a 10 ms stimulus `pagan.get_condition`'s `stim_t0 = stimulus[0] + 1`
leaves `nbins = 0`, so **no input was ever delivered to them at all, not even the LOC go cue**.
One presentation step is `PAGAN_STIM0=20`, not 10.

**The model: `examples/models/pagan_hist_tscost.py`** (full rules in its docstring). Built on
`pagan_hist_curriculum`. Four actions — `FIXATE`, `CHOOSE-LEFT`, `CHOOSE-RIGHT`, `DECLINE`
(`Nout = 4`, and because pyrl draws `Wout` last from the weight seed, `Win`/`bin`/`Wrec` are
bit-identical to the three-action arms and to `effort2` at the same seed, so every comparison
stays pairwise). No fixation requirement; `DECLINE` is available at every timestep and ends the
trial for no reward; every timestep of the presentation window on which the network does not
decline costs `PAGAN_TSCOST/130`, so the full window costs exactly 0.55 and a trial abandoned
part-way costs only what it used, with no refund. `bout` is left at the pyrl default, so initial
action probabilities are not handcrafted. No timeout penalty (`R_ABORTED = 0`).

The presentation window is one step (20 ms, `PAGAN_PRESTIM`) through the pre-training; the
task-phase schedule then starts at `PAGAN_STIM0` = 50 ms and grows by 50 ms at one opportunity per
checkpoint, taken according to `PAGAN_RAMP` (25 growths to the full length). Early stopping is
suppressed while the schedule is still running, so a run cannot stop at criterion on a 100 ms
window.

**Why the schedule starts at 50 ms and not 20.** `correct_side` is the sign of the *realized*
pulse count (`pagan.py:154-167`), so the label is a deterministic function of what the network saw
except when that count is exactly zero, where it is a coin flip. An observer reading the pulses
perfectly therefore scores `1 - P(integral = 0)/2`. With lambda = `pulse_rate`*`dt`/1000 = 0.4
pulses per bin and `nbins = ms/10 - 1`, by exact Poisson convolution over the three strengths and
both sides:

| window | input bins | P(integral = 0) | accuracy ceiling |
|---|---|---|---|
| 20 ms | 1 | 0.694 | 0.653 |
| 30 ms | 2 | 0.515 | 0.742 |
| 50 ms | 4 | 0.334 | **0.833** |
| 100 ms | 9 | 0.185 | 0.908 |
| 1,300 ms | 129 | 0.016 | 0.992 |

A one-bin window is two-thirds coin flip and leaves the performance-gated arm only 0.10 of headroom
over its 0.55 gate; four bins leave 0.28. The ceiling is for an *unselective* responder - a network
that preferentially declined the zero-count trials could exceed it on the trials it chose, the same
P(correct|choice) inflation that late declines allow. `PAGAN_PRESTIM` stays at 20 ms, because the
pre-training window is one presentation step by design, so the hand-off steps 20 -> 50 ms. The
module default for `PAGAN_STIM0` stays at the full 1,300 ms: a rollout that silently measured a
50 ms window would corrupt every record.

New environment switches, to add to the table in section 2:

| variable | meaning | default |
|---|---|---|
| `PAGAN_TSCOST` | cost of holding the presentation window at its full length | `0.55` |
| `PAGAN_RAMP` | duration schedule: `off` \| `fixed` \| `choice` \| `perf` | `choice` |
| `PAGAN_STIM0` | where the schedule starts, in ms. Defaults to the full length so analysis rollouts measure the full task; training passes `50` (four input bins - see the ceiling table below) | `1300` |
| `PAGAN_PRESTIM` | presentation length during the pre-training phases, in ms | `20` |
| `PAGAN_RAMP_PCHOICE` | P(choice) the `choice` and `perf` schedules require | `0.70` |
| `PAGAN_RAMP_PCORRECT` | P(correct\|choice) the `perf` schedule additionally requires | `0.55` |
| `PAGAN_TIMEOUT` | reward for a trial whose buffer ends with no choice | `0` |

**The arms.** Four, all pre-trained, cost 0.55, 8 seeds each, 2,800 updates, differing only in the
schedule: `tc55off` (none — straight to 1,300 ms at the hand-off, the control), `tc55fix`
(unconditional), `tc55cho` (P(choice) ≥ 0.70), `tc55per` (P(choice) ≥ 0.70 and
P(correct|choice) > 0.55). Run folders `pagan_hist_tscost_g06tc55{off,fix,cho,per}_w001..008`,
explorer class `e11`. Deferred to a following run: the no-pre-training controls (the same arms
with a `ctrl` suffix, which picks up `PAGAN_PRETRAIN=0` for free) and the staged curriculum.

**There are two runs of this model.** The four-arm run above is queued for 0071 and stays exactly
as specified. Because 0071 was fully committed by Marino's `pagan_hist_evo` sweep on 16 Oct, a
**slimmed version was split off to run locally on an Apple Silicon Mac**: three arms (the `off`
control dropped, the smoke test having already shown that jumping to the full window collapses
engagement) and a 360-trial validation batch instead of 720, which is where most of the saving is
— a checkpoint's validation batch is the larger part of the per-update cost, so halving it takes a
run from 2.25 to 1.50 s/update. 24 runs, ~28 core-hours against ~56 for the full run.

Its arms are `tl55fix`, `tl55cho`, `tl55per` (`tl` rather than `tc`), run folders
`pagan_hist_tscost_g06tl55*_w001..008`, explorer class `e12` — separate names throughout, because
**the two runs are not comparable seed-for-seed**: the same weight seed gives the same starting
weights, but validation draws consume the same stimulus RNG stream as training draws, so a
360-trial batch puts every later trial out of step with the 720-trial run. They answer the same
question on separate samples; do not pool them. The smaller batch also halves gate precision: with
~359 decided trials the SE of accuracy at chance is 0.026 against 0.019, so a chance network
clears the `perf` arm's 0.55 gate on 2.9% of checkpoints rather than 0.4% — about 16 growths of
noise over a run, against the 25 that complete the schedule. That arm's schedule trace must be
read alongside its accuracy. Launch with `tools/launch/queue_tscost_local.sh`;
`tools/launch/README_local_launch.md` is the full procedure.

**Launching** (`tools/launch/`, the portable successor to Marino's
`models/scripts/launch_weak_start.sh`, which hardcodes his paths and env). The scripts take
`PYRL_REPO`, `PYRL_PYTHON`, `JOBS`, `MAX_ITER` and `NICE` from the environment, so they run
unchanged on a Linux box or an Apple Silicon Mac; `tools/launch/README_local_launch.md` covers the
local launch, the ~96 core-hour budget, the Python 2.7/Theano situation on arm64, and what still
has to be checked on the Mac:

```bash
bash tools/launch/launch_weak_start.sh tscost tc55cho 1 2 3 4 5 6 7 8   # one arm
bash tools/launch/queue_tscost.sh          # the full run: four arms, 720-trial validation
bash tools/launch/queue_tscost_local.sh    # the slim local run: three arms, NVAL=360
MAX_ITER=60 bash tools/launch/smoke_tscost.sh                           # 4 modes x 1 seed, nice'd
```

`launch_weak_start.sh` carries the weak start (`rho=0.6`, critic at `baseline_rho=1.15`), derives
the task switches from the arm tag, keeps at most `JOBS` seeds in flight, and takes `NVAL` as
`--config n_validation`; both queues run their arms in sequence and skip any arm with a
`DONE_g06<arm>` marker, so they are resumable. **Check the machine first**: 0071 has 32 cores and Marino's sweeps routinely
use all of them (`ps -eo pcpu,args | grep do.py`).

The `perf` gate is 0.55 and deliberately not 0.50: with ~718 decided trials per validation batch
the SE of P(correct) at chance is 0.019, so a 0.50 threshold is cleared by a chance network on
48.5% of checkpoints and the schedule would complete on validation noise alone in ~52 checkpoints.
0.55 is cleared on 0.4%. The arm may therefore hold a short window for a long time; that is
accepted, a stalled performance gate being an informative result rather than a failure.

**What the smoke test showed** (`logs/weak/smoke_tc*.log`, 60 updates, seed 1, same seed across
modes). Through the pre-training all arms are identical and engagement is learned fast: the
fraction of trials reaching a choice goes 0.42 → 0.77 → 0.96 → 0.99 → 1.00 over the six
pre-training checkpoints, at a mean cost of 0.008 (two charged steps). Then:

- `off` collapses at the hand-off, exactly as predicted: 0.996 → 0.000 within one checkpoint, with
  essentially every trial now a *mid-window* withdrawal (not a first-step refusal), at a median
  step of 42 falling to 31 over the next five checkpoints. The network has learned not to refuse a
  trial but cannot hold a 130-step window, and withdraws earlier the longer it trains.
- `fixed` and `choice` advance one step per checkpoint (20 -> 70 -> ... -> 320 ms in the first
  smoke, which used the old 20 ms start; 50 -> 100 -> ... with the current one), with the choice
  fraction staying at or above 0.996 the whole way and the mean cost tracking the window
  (0.008 -> 0.114). The schedule does keep engagement alive where the jump destroys it.
- `perf` does not move: P(choice) is 1.00 but accuracy sits at 0.48-0.53, below the 0.55 gate, so
  the window is held. **Raising the start from 20 to 50 ms did not change this** - re-smoked at
  50 ms for 80 updates (ten task checkpoints), accuracy was still 0.47-0.52 and the gate never
  opened. The ceiling went from 0.653 to 0.833, so the window is no longer the binding constraint;
  what binds is learning speed. The network comes out of 2,160 pre-training trials in which the
  pulses were zeroed and the only signal was trial history, so it has to start using LMR/HML from
  scratch, and 50 task updates (3,600 trials) is not enough - the other arms take on the order of
  5,000 trials just to form the line. Whether `perf` ever ramps inside 2,800 updates is the open
  empirical question of that arm. A stalled `perf` is still an interpretable result (a performance
  gate is strictly more conservative than a choice gate), but if you want it to move sooner the
  levers are a larger `PAGAN_STIM0` or a lower `PAGAN_RAMP_PCORRECT` for that arm alone.

Timed-out trials were 0.000 everywhere, so the `PAGAN_TIMEOUT=0` choice turns out not to matter
much in practice — with four actions and a 200-step decision epoch a trial essentially always ends
in a choice or a decline.

Unit tests (Python 2.7, no training, seconds): `tests/test_pagan_hist_tscost.py` pins the reward
structure (44 assertions: the epoch geometry and the go-cue offset at both window lengths, the
cost of declining at every position, exactly 130 charges at full length, a null in-window
left/right that is charged but does not end the trial, no refund on a late decline, and the
outcome bookkeeping) and `tests/test_pagan_hist_tscost_ramp.py <mode>` pins the schedule state
machine (one mode per process, since `PAGAN_RAMP` is read at import), including the early-stopping
guard and the `PAGAN_PRETRAIN=0` path.

## 10. Two explorer trees

`tools/measurement/build_network_explorer.py` now takes `--tree`:

```bash
# records/mechanism/activity roots are overridable by environment, so the builder runs from
# this account without editing the file (section 6 said to edit or symlink them)
export EXPLORER_RECORDS=~/records EXPLORER_MECH=~/mechanism EXPLORER_ACT=~/activity
export EXPLORER_DATA=~/pyrl/examples/work/data
python tools/measurement/build_network_explorer.py --site ~/explorer_site --tree elise
python tools/measurement/build_network_explorer.py --site ~/explorer_site --tree main
```

`--tree main` is the original tree at the site root (the seven types of the 13 Oct talk) and is
unchanged except for a link, on its front page, to the other one. `--tree elise` builds
`<site>/elise/` with its own `index.html` and `archive.html`, its own nav (plus an "All models"
link back up), and only the runs belonging to its own classes — this and the following runs are
documented there unless specified otherwise. Both trees come out of the same `ARMS` table and the
same per-run `compute()`, so adding a run to a tree is two edits: an `ARMS` entry giving its
run-name prefix a class key, and that key in `ELISE_KEYS` (or `MAIN_KEYS`) with a `CLASSES` entry
describing it. The `tscost` arms are class `e11`.

---

## 11. The stage-ladder run on the per-timestep cost (`stages4`, 16 Oct 2026)

The third run of the engagement line, and the one the original requirements deferred. Where the
two `tscost` runs of section 9 hand off from the nose-poke pre-training straight to the full task
and compare *how the presentation window is grown*, this one hands off to the rats' 10-stage
TaskSwitch ladder. It is the arm that asks whether the graded stage structure — one feature first,
then both in blocks, then an uninformative distractor, then an informative one — protects
engagement where the straight hand-off does not. Local, because 0071 was still saturated.

**Model: `examples/models/pagan_hist_stages4.py`.** The ladder of `pagan_hist_stages.py` rebuilt
on `pagan_hist_tscost` instead of `pagan_hist_effort2`. Everything about the cost is inherited and
untouched: four actions (so `Nout = 4`), 0.55/130 per engaged timestep, `R_ABORTED = 0`, the
`PRESTIM`/`STIM0`/`RAMP` machinery, `bout` at the pyrl default.

It is a port of the share's **version 2** ladder, not the **version 1** still committed in this
repo as `pagan_hist_stages.py`. v2 adds the lab's side anti-bias in stages 1-6, the quadrant
anti-bias in stage 10 of every arm (so `short` and `full` differ only in whether stages 7-9
exist), and a block-length cap at twice the rats' median for the stage — the protocol relies on a
trainer to end a block a stuck rat never ends, and three networks of the first probe sat in one
task for 30,000-90,000 trials. `pagan_hist_stages.py` is deliberately **not** updated: it is the
committed record of the `st55` effort2 runs, and changing it would change what those reproduce.
That v1/v2 divergence between this repo and the share is a known, separate thing.

### Why this file exists: four hooks, and the `stages3` bug class

`pyrl/model.py` binds **one** function per hook name out of the model file's namespace, and a
`def` shadows whatever the star import brought in — the harness does no chaining. The abandoned
`pagan_hist_stages3.py` was a three-line copy of the ladder onto `pagan_hist_fixcost` that adapted
none of this. `tscost` defines all four hooks, so each had to be handled explicitly:

| hook | what `stages4` does | what breaks otherwise |
|---|---|---|
| `set_validation` | kept as-is; it already chains through a `hasattr` guard | nothing |
| `record_outcome` | calls `_eff.record_outcome` **first and unconditionally**, before its own `if _st['validating']: return` | the cost model's diagnostics go silent — every checkpoint prints `no validation trials recorded`. The two filters are *complementary*: `tscost` summarises the validation batch and returns early when not validating, the ladder tracks training-trial accuracy and returns early when validating. Both must see every trial. |
| `update` | calls `_eff.update(perf)` exactly once; never `_curr.update` or `_base.update` | the nose-poke phases pass twice as fast, or the duration schedule gets a second independent clock — exactly how `stages3` ended up with the ladder and the window on unrelated clocks |
| `terminate` | `stage() == 10 and _eff.terminate(perf)`, not `_curr.terminate` | `tscost`'s guard against stopping at criterion while the schedule is still running is silently dropped |

`tests/test_pagan_hist_stages4.py` pins all four, and the `record_outcome` test asserts the cost
model's counter is non-zero — the exact thing `stages3` got wrong.

### The duration schedule runs in Stage 1 only, once in all of training

No new machinery: `tscost.update` already pins the window to `PAGAN_PRESTIM` (20 ms) through the
pre-training, sets it to `PAGAN_STIM0` (50 ms) at a one-shot hand-off, and grants one +50 ms
opportunity per checkpoint while P(choice) ≥ 0.70. At the hand-off `stage()` is 1, so the schedule
runs in Stage 1 by construction. Confining it there is one guard in `update`: **no stage may be
left while `_base.stimulus < _base.stimulus_max`.** Gating on the window rather than on
`stage() == 1` is what makes it correct rather than merely convenient — the schedule cannot outlive
Stage 1, and once the window is full `tscost`'s own `stimulus >= stimulus_max` branch makes it a
permanent no-op, so "once" needs no latch. With `PAGAN_RAMP=off` the window is full from the
hand-off and the guard never fires.

Expect Stage 1 to hold until about update 155 of 2,800 (30 updates of pre-training, then 25 growth
opportunities). If the schedule ever stalls the seed stays in Stage 1 for the whole run — accepted
deliberately, and interpretable.

### One bug the smoke caught, and the one departure from v2

The first `promote` smoke deadlocked in stage 3. On promotion the network engaged on LOC trials and
declined essentially every FRQ one (validation showed ~0.39 of trials reaching a choice, almost
exactly 0.5 x the LOC rate — engagement turned out to be context-specific, which is itself worth
knowing). v2 increments the block clock only on *attempted* trials, so the FRQ block never reached
its cap, never ended, and training sat in it for 15,000 trials with the attempt counter frozen at
60 and no reward signal at all. Under `effort2`, where declining is a single decision at step 0 and
~75% of trials are attempted, the bug is unreachable; under a per-timestep cost with ~60% mid-window
withdrawal it is fatal.

So `stages4` counts the block clock on every trial **presented**, not only those attempted. That is
the one deliberate departure from v2: it keeps the intent ("a block cannot last longer than twice
the rats' median") and makes the cap unconditional. The performance-gated switch is unchanged and
still runs on attempted trials. `tests/test_pagan_hist_stages4.py` has a regression test that
declines its way through a whole block and asserts the switch still happens. After the fix, the
re-run smoke shows `attempted` climbing steadily in stage 3 (118 → 750 over ten checkpoints) where
it had been frozen at 60 indefinitely.

### What the smokes actually say about this design — read before interpreting the run

Three results, all from seed 1 at `NVAL=360`, and together they say where the difficulty is.

1. **The schedule works and the block holds.** Stage 1 held at criterion for 20 checkpoints while
   the window climbed 450 → 1,250 ms (accuracy on the LOC-only task reaching 0.95-1.0), then
   promoted to stage 2 and stage 3 as soon as the window was full. The `jump` control shows why the
   schedule is needed at all: with the window going straight from 20 to 1,300 ms at the hand-off,
   the fraction of trials reaching a choice goes 0.997 → 0.000 in one checkpoint, 100% mid-window
   withdrawal at a median step of ~40, and the ladder never starts.
2. **Engagement is context-specific, and stage 3 is the barrier.** Stage 3 is the first stage with
   two tasks. There the network engages on LOC trials and declines essentially every FRQ one:
   validation settles at ~0.37 of trials reaching a choice, almost exactly half the LOC-only rate,
   and the FRQ accuracy window never accumulates its 50 trials (`frq: None` at every checkpoint).
   Blocks now rotate properly, so training is productive rather than stuck — but **promotion out of
   stage 3 needs *both* tasks above 0.70**, so on this evidence the ladder may well stop there.
3. **That is the run's question, not a defect.** Whether 2,800 updates is enough for engagement to
   generalise to the second context is exactly what the arm is for, and the smoke only had 500. The
   thing to watch in the logs is `frq` turning from `None` into a number — that is the moment
   engagement crosses contexts. If it never does, the result is that the stage ladder's two-task
   stage is where a per-timestep engagement cost stalls, which is a finding worth having.

### Two things to read carefully, documented rather than fixed

- **Promotion counts engaged trials only.** A trial is attempted only if `status['choice']` is set,
  which excludes all four of the cost model's non-choice outcomes, and `hit` comes from
  `status['correct']` rather than from `reward`, so it is immune to the per-step cost and to the
  quadrant scaling. (`pagan_hist_antibias.record_outcome` uses `hit = reward > 0`, which *would* be
  corrupted by a negative engagement cost; the ladder's own side anti-bias does not.) So a network
  declining most trials is slowed toward promotion but never blocked: stage duration and
  engagement are confounded, and the stage trace must be read alongside the fraction attempted.
- **The quadrant scaling and the cost interact.** The scaling multiplies the *terminal* reward by
  4 × p(quadrant) while the cost is charged separately on earlier timesteps, so an attempted
  correct trial in stages 7-10 nets 4p − 0.55, negative for a rare quadrant (p < 0.1375). Inherent
  to combining the rats' water-volume scaling with an engagement cost; it makes the expected value
  of engaging vary by quadrant.

### Arm and launch

One arm, 8 seeds, `NVAL=360` as in the slim local run. Run folders
`pagan_hist_stages4_g06sc55full_w001..008`, explorer class `e13`. `sc` (stage ladder on the
per-timestep **c**ost) collides with none of `st55*` (the effort2 ladder), `tc55*` or `tl55*`.
~9 core-hours, about 1.2 h wall at `JOBS=8`, ~1 GB of training output.

```bash
MAX_ITER=80 bash tools/launch/smoke_stages4.sh block      # the schedule holding stage 1, ~1 min
bash tools/launch/smoke_stages4.sh promote                # the schedule completing, then a promotion
bash tools/launch/smoke_stages4.sh jump                   # the control: a straight jump kills engagement
JOBS=8 bash tools/launch/queue_stages4_local.sh           # the run
```

Deferred, each one launcher invocation away now that the model exists: the `short` ladder
(`sc55short`), the no-pre-training control (`sc55fullctrl`, which picks up `PAGAN_PRETRAIN=0` for
free), the other two ramp criteria crossed with the ladder, and a `PAGAN_RAMP=off` stage arm.
