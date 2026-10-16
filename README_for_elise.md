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
