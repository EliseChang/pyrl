#!/usr/bin/env python3
"""Build the network explorer: one static site over every reward-trained RNN we have.

Why. A local HTML site in which the fits of every network can be explored, and what is going
on in each network seen in many different ways. First draft, built with the `explorer` skill (catalogue shape). Covers
the 16 curriculum/control pilots and the 100-seed Approach A library (Elise's pyrl stack).

Inputs (all on rebel):
    records   /mnt/big/pyrl_work/psytrack/records/<run>_record.npz     (psytrack_on_network.py --extract)
              y (1 = right), session (checkpoint index), iters, X_* regressors, and the raw_*
              per-trial fields (context 0 = LOC / 1 = FRQ, e_loc, e_frq with + = right, reward)
    models    ~/pyrl_work/examples/work/data/<run>/<run>.pkl           training_history per checkpoint
              ~/pyrl_work/examples/work/data/<run>/checkpoints/iter_*.pkl   policy and baseline params
    psytrack  /mnt/big/pyrl_work/psytrack/{,farm/}<run>_psytrack.npz    where a fit exists
    compare   /mnt/big/pyrl_work/psytrack/compare/<run>.png             PsyTrack vs ridge GLM
    population figures in /mnt/big/pyrl_work/psytrack/*.png

Per network the builder computes (cached in <site>/cache/<run>.npz, so figures can be restyled
without recomputing): task accuracy per checkpoint, the ridge GLM weights and SEs per checkpoint
with the identifiability flag, P(stay|win) and P(stay|lose), psychometric curves and the
LOC x FRQ choice matrices per context per checkpoint, and weight norms and spectral radii of the
policy and the critic across checkpoints.

Pages: index.html (population table with sparklines, population figures), one page per network
(Behaviour, Strategy, Mechanism, Critic sections with checkpoint sliders), compare.html,
methods.html. Usage on rebel:
    OMP_NUM_THREADS=2 ~/rat_strategy_discovery/.venv/bin/python build_network_explorer.py \
        --site /mnt/big/pyrl_work/explorer_site [--only pagan_hist_curriculum] [--workers 10]
then rsync the site to the Mac (gatsby_talk/rl_rnn/explorer/site/).
"""
import argparse
import glob
import os
import pickle
import re
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from explorer_lib import Site, table, fig, grid, card, kv, slider, sparkline, tabs, note, caveat, Raw, esc, keynav  # noqa: E402
from glm_vs_psytrack import ridge_logistic  # noqa: E402

# The four input roots. Defaults are Marino's paths on rebel; override by environment so the
# builder runs from a different account without editing the file (see README_for_elise.md s6).
RECORDS = os.environ.get("EXPLORER_RECORDS", "/mnt/big/pyrl_work/psytrack/records")
DATA = os.environ.get("EXPLORER_DATA", os.path.expanduser("~/pyrl_work/examples/work/data"))
PSY = os.environ.get("EXPLORER_PSY", "/mnt/big/pyrl_work/psytrack")
MECH = os.environ.get("EXPLORER_MECH", "/mnt/big/pyrl_work/mechanism")      # <run>_policy_mechanism.npz from pyrl_mechanism.py
ACT = os.environ.get("EXPLORER_ACT", "/mnt/big/pyrl_work/activity")        # <run>_activity.npz from activity_summary.py
N_GRADIENT = 72
# the rats' PsyTrack colours, so network and rat figures read the same way
CLR = {"bias": "tab:blue", "rel": "tab:red", "irr": "tab:green", "histChoice": "tab:orange", "wsls": "tab:purple"}
PHASES = [("fixed", 0, 360, "#e8f0fe"), ("blocks", 360, 1440, "#e9f7ef"), ("random", 1440, 2160, "#fef3e7")]
INPUTS = ["LMR", "HML", "LOC", "FREQ", "PREV-L-WIN", "PREV-L-LOSE", "PREV-R-WIN", "PREV-R-LOSE"]
P_MAJOR_MAX = 0.95           # above this majority-choice fraction the GLM weights are not identifiable
CACHE_VERSION = 6            # bump when compute() gains fields, so stale caches are rebuilt
N_STRIP = 12                 # checkpoints sampled for the matrix and psychometric strips


# run-name prefix on disk -> (display prefix, arm label, class key). Order matters: longest first.
ARMS = [("pagan_hist_curriculum_curr_", "pretrained_", "pretrained", "base"),
        ("pagan_hist_curriculum_ctrl_", "no_pretrain_", "no_pretrain", "base"),
        ("pagan_hist_histgauss100_", "library_", "library (no_pretrain, 36k trials)", "library"),
        ("pagan_hist_antibias_abcurr_", "antibias_pretrained_", "antibias pretrained (beta 5)", "antibias"),
        ("pagan_hist_antibias_abctrl_", "antibias_no_pretrain_", "antibias no_pretrain (beta 5)", "antibias"),
        ("pagan_hist_antibias_abnegctrl_", "antibias_inverted_no_pretrain_", "inverted antibias no_pretrain (beta -5)", "antibias"),
        ("pagan_hist_antibias_abneg_", "antibias_inverted_pretrained_", "inverted antibias pretrained (beta -5)", "antibias"),
        # effort v2 (DECLINE action, lump cost): the engagement model carried forward
        ("pagan_hist_effort2_e240curr_", "effort0.40_pretrained_", "effort 0.40 pretrained", "effort"),
        ("pagan_hist_effort2_e240ctrl_", "effort0.40_no_pretrain_", "effort 0.40 no_pretrain", "effort"),
        ("pagan_hist_effort2_e255curr_", "effort0.55_pretrained_", "effort 0.55 pretrained", "effort"),
        ("pagan_hist_effort2_e255ctrl_", "effort0.55_no_pretrain_", "effort 0.55 no_pretrain", "effort"),
        ("pagan_hist_effort2_e270curr_", "effort0.70_pretrained_", "effort 0.70 pretrained", "effort"),
        ("pagan_hist_effort2_e270ctrl_", "effort0.70_no_pretrain_", "effort 0.70 no_pretrain", "effort"),
        # weak-start networks (rho 0.6, measured starting spectral radius about 0.8): the networks of the talk from 13 Oct
        ("pagan_hist_curriculum_g06curr_", "weak_pretrained_", "weak start, pretrained", "t2"),
        ("pagan_hist_curriculum_g06ctrl_", "weak_no_pretrain_", "weak start, no_pretrain", "t1"),
        ("pagan_hist_antibias_g06abcurr_", "weak_antibias_pretrained_", "weak start, antibias pretrained (beta 5)", "t4"),
        ("pagan_hist_antibias_g06abctrl_", "weak_antibias_no_pretrain_", "weak start, antibias no_pretrain (beta 5)", "weak_other"),
        ("pagan_hist_antibias_g06abnegctrl_", "weak_antibias_inverted_no_pretrain_", "weak start, inverted antibias no_pretrain (beta -5)", "weak_other"),
        ("pagan_hist_antibias_g06abneg_", "weak_antibias_inverted_pretrained_", "weak start, inverted antibias pretrained (beta -5)", "weak_other"),
        ("pagan_hist_effort2_g06e255curr_", "weak_effort0.55_pretrained_", "weak start, effort 0.55 pretrained", "t8"),
        ("pagan_hist_effort2_g06e255ctrl_", "weak_effort0.55_no_pretrain_", "weak start, effort 0.55 no_pretrain", "t7"),
        ("pagan_hist_stages_g06st55short_", "weak_stages1-6_effort0.55_", "weak start, stages 1-6 then full task (effort 0.55)", "t9"),
        ("pagan_hist_stages_g06st55full_", "weak_stages1-10_effort0.55_", "weak start, stages 1-10 with anti-bias (effort 0.55)", "t10"),
        # the cost sweep (13 Oct evening): the same effort network at 0.70 and 0.85
        ("pagan_hist_effort2_g06e270ctrl_", "weak_effort0.70_no_pretrain_", "weak start, effort 0.70 no_pretrain", "t7"),
        ("pagan_hist_effort2_g06e270curr_", "weak_effort0.70_pretrained_", "weak start, effort 0.70 pretrained", "t8"),
        ("pagan_hist_effort2_g06e285ctrl_", "weak_effort0.85_no_pretrain_", "weak start, effort 0.85 no_pretrain", "t7"),
        ("pagan_hist_effort2_g06e285curr_", "weak_effort0.85_pretrained_", "weak start, effort 0.85 pretrained", "t8"),
        # fixation-cost effort networks (version 3, 13 Oct): no decline output, every step of fixation costs, breaking fixation abandons the trial
        ("pagan_hist_fixcost_g06fc55ctrl_", "weak_fixcost_no_pretrain_", "weak start, fixation-cost effort, no_pretrain", "weak_other"),
        ("pagan_hist_fixcost_g06fc55curr_", "weak_fixcost_pretrained_", "weak start, fixation-cost effort, pretrained", "weak_other"),
        ("pagan_hist_stages3_g06fc55short_", "weak_fixcost_stages1-6_", "weak start, fixation-cost effort, stages 1-6 then full task", "weak_other"),
        ("pagan_hist_stages3_g06fc55full_", "weak_fixcost_stages1-10_", "weak start, fixation-cost effort, stages 1-10", "weak_other"),
        # per-timestep engagement cost (version 4, 16 Oct): a DECLINE output as in effort2, but the
        # cost accrues per timestep of the stimulus-presentation window, and that window's length is
        # grown from one step by one of three schedules (the arms) or not at all (the control)
        ("pagan_hist_tscost_g06tc55off_", "tscost_noramp_", "per-timestep cost 0.55, no duration schedule", "e11"),
        ("pagan_hist_tscost_g06tc55fix_", "tscost_rampfixed_", "per-timestep cost 0.55, fixed duration schedule", "e11"),
        ("pagan_hist_tscost_g06tc55cho_", "tscost_rampchoice_", "per-timestep cost 0.55, P(choice)-gated schedule", "e11"),
        ("pagan_hist_tscost_g06tc55per_", "tscost_rampperf_", "per-timestep cost 0.55, performance-gated schedule", "e11"),
        # the slimmed local run (16 Oct): three arms, 360-trial validation batch. A separate run,
        # not comparable seed-for-seed with the tc55 arms above (a different validation batch size
        # puts the shared stimulus RNG stream out of step), hence its own class.
        ("pagan_hist_tscost_g06tl55fix_", "tscostL_rampfixed_", "per-timestep cost 0.55, fixed duration schedule (slim local run)", "e12"),
        ("pagan_hist_tscost_g06tl55cho_", "tscostL_rampchoice_", "per-timestep cost 0.55, P(choice)-gated schedule (slim local run)", "e12"),
        ("pagan_hist_tscost_g06tl55per_", "tscostL_rampperf_", "per-timestep cost 0.55, performance-gated schedule (slim local run)", "e12"),
        # the stage-ladder run (16 Oct): the rats' 10-stage TaskSwitch ladder on the per-timestep
        # cost, with the duration schedule confined to stage 1
        ("pagan_hist_stages4_g06sc55full_", "stages4_full_", "per-timestep cost 0.55, full stage ladder (stages 1-10)", "e13"),
        ("pagan_hist_stages4_g06sc55short_", "stages4_short_", "per-timestep cost 0.55, stages 1-6 then the full task", "e13"),
        # the no-pre-training control (17 Oct): straight into the full task, no curriculum at
        # all, max_iter cut to 1,000 because the collapse is expected from the first checkpoint
        ("pagan_hist_tscost_g06tl55offctrl_", "tscost_noPretrain_noRamp_", "per-timestep cost 0.55, no pre-training, no duration schedule (control, max_iter 1000)", "e14"),
        # stage curriculum on the effort v2 network (EFFORT 0.55, pre-trained): the rats' TaskSwitch stages
        ("pagan_hist_stages_st55short_", "stages1-6_effort0.55_", "stages 1-6 then full task (effort 0.55)", "stages"),
        ("pagan_hist_stages_st55full_", "stages1-10_effort0.55_", "stages 1-10 with quadrant anti-bias (effort 0.55)", "stages"),
        # effort v1 (decline shared the side outputs, per-step cost): kept for the record only
        ("pagan_hist_effort_eff40curr_", "effortV1_0.40_pretrained_", "effort v1 (failed design) 0.40 pretrained", "effort_v1"),
        ("pagan_hist_effort_eff40ctrl_", "effortV1_0.40_no_pretrain_", "effort v1 (failed design) 0.40 no_pretrain", "effort_v1"),
        ("pagan_hist_effort_eff55curr_", "effortV1_0.55_pretrained_", "effort v1 (failed design) 0.55 pretrained", "effort_v1"),
        ("pagan_hist_effort_eff55ctrl_", "effortV1_0.55_no_pretrain_", "effort v1 (failed design) 0.55 no_pretrain", "effort_v1"),
        ("pagan_hist_effort_eff70curr_", "effortV1_0.70_pretrained_", "effort v1 (failed design) 0.70 pretrained", "effort_v1"),
        ("pagan_hist_effort_eff70ctrl_", "effortV1_0.70_no_pretrain_", "effort v1 (failed design) 0.70 no_pretrain", "effort_v1")]

# The classes of models, in the order of the home page. `description` says what the model is
# and what varies inside the class; `findings` is written by hand from the results and should be
# revised whenever the numbers below it change (the numbers are computed at build time).
FRONT = ("t1", "t2", "t4", "t7", "t8", "t9", "t10")     # the types shown on the front page (13 Oct 2026); every other class is on archive.html
CLASSES = [
    ("t1", "1. No pre-training", dict(
        description="""<p><b>What it gets.</b> The full task from the first trial. No anti-bias. This is the starting point (Elise's network).</p><p>Reward-trained network of 100 units (it is never told the answer, only whether it was rewarded), started with weak recurrent connections like the networks of the Nature paper. Eight seeds, up to 201,600 trials (a rat's career). Tested afterwards on the plain full task with every anti-bias switched off.</p>""",
        findings="""<p>See the numbers and figures below; they are computed from the runs at every build.</p>""")),
    ("t2", "2. Pre-trained", dict(
        description="""<p><b>What it gets.</b> First the rats' nose-poke pre-training (one side only, then alternating blocks of sides, then random sides; 2,160 trials), then the full task. No anti-bias.</p><p>Reward-trained network of 100 units (it is never told the answer, only whether it was rewarded), started with weak recurrent connections like the networks of the Nature paper. Eight seeds, up to 201,600 trials (a rat's career). Tested afterwards on the plain full task with every anti-bias switched off.</p>""",
        findings="""<p>See the numbers and figures below; they are computed from the runs at every build.</p>""")),
    ("t4", "4. Pre-trained + side anti-bias", dict(
        description="""<p><b>What it gets.</b> The pre-training, then the full task with the trainer's side anti-bias: the next correct side is more likely to be the one the network has been getting wrong.</p><p>Reward-trained network of 100 units (it is never told the answer, only whether it was rewarded), started with weak recurrent connections like the networks of the Nature paper. Eight seeds, up to 201,600 trials (a rat's career). Tested afterwards on the plain full task with every anti-bias switched off.</p>""",
        findings="""<p>See the numbers and figures below; they are computed from the runs at every build.</p>""")),
    ("t7", "7. Effort, no pre-training", dict(
        description="""<p><b>What it gets.</b> The effort network: a fourth action, decline, available at the first step of a trial (reward 0, no history on the next trial); attempting a trial costs the effort cost once, then a correct choice pays 1 and a wrong one 0. Attempting pays only when accuracy is above the cost. Costs 0.55, 0.70 and 0.85 (the sweep of 13 Oct). Scored on accuracy among attempted trials and on the fraction of trials attempted, never on attempting 99% of trials. Full task from the first trial, no anti-bias.</p><p>Reward-trained network of 100 units (it is never told the answer, only whether it was rewarded), started with weak recurrent connections like the networks of the Nature paper. Eight seeds, up to 201,600 trials (a rat's career). Tested afterwards on the plain full task with every anti-bias switched off.</p>""",
        findings="""<p>See the numbers and figures below; they are computed from the runs at every build.</p>""")),
    ("t8", "8. Effort, pre-trained", dict(
        description="""<p><b>What it gets.</b> The effort network: a fourth action, decline, available at the first step of a trial (reward 0, no history on the next trial); attempting a trial costs the effort cost once, then a correct choice pays 1 and a wrong one 0. Attempting pays only when accuracy is above the cost. Costs 0.55, 0.70 and 0.85 (the sweep of 13 Oct). Scored on accuracy among attempted trials and on the fraction of trials attempted, never on attempting 99% of trials. After the nose-poke pre-training. No anti-bias.</p><p>Reward-trained network of 100 units (it is never told the answer, only whether it was rewarded), started with weak recurrent connections like the networks of the Nature paper. Eight seeds, up to 201,600 trials (a rat's career). Tested afterwards on the plain full task with every anti-bias switched off.</p>""",
        findings="""<p>See the numbers and figures below; they are computed from the runs at every build.</p>""")),
    ("t9", "9. Effort + stages 1-6, then the full task", dict(
        description="""<p><b>What it gets.</b> The effort network (cost 0.55, as in type 7) after the pre-training, then the rats' early stages (location task alone; both tasks in blocks, each pure; both features present with the irrelevant one uninformative) with the side anti-bias, then the full task with the quadrant anti-bias.</p><p>Reward-trained network of 100 units (it is never told the answer, only whether it was rewarded), started with weak recurrent connections like the networks of the Nature paper. Eight seeds, up to 201,600 trials (a rat's career). Tested afterwards on the plain full task with every anti-bias switched off.</p>""",
        findings="""<p>See the numbers and figures below; they are computed from the runs at every build.</p>""")),
    ("t10", "10. Effort + stages 1-10", dict(
        description="""<p><b>What it gets.</b> The effort network (cost 0.55, as in type 7) after the pre-training, then the rats' whole stage ladder: stages 1-6 with the side anti-bias, stages 7-9 where the irrelevant feature becomes informative and the evidence harder, with the quadrant anti-bias, then the full task.</p><p>Reward-trained network of 100 units (it is never told the answer, only whether it was rewarded), started with weak recurrent connections like the networks of the Nature paper. Eight seeds, up to 201,600 trials (a rat's career). Tested afterwards on the plain full task with every anti-bias switched off.</p>""",
        findings="""<p>See the numbers and figures below; they are computed from the runs at every build.</p>""")),
    ("weak_other", "Other weak-start networks (anti-bias without pre-training; inverted anti-bias; the fixation-cost effort network of 13 Oct, set aside)", dict(
        description="""<p>Weak-start networks that are not among the seven types of the talk: the side anti-bias without pre-training, and the
        inverted anti-bias (which rewards the bias) with and without pre-training. Kept as controls.</p>""",
        findings="""<p>See the numbers below.</p>""")),
    ("base", "RL networks with the rats' nose-poke pre-training, or without (talk part 4)", dict(
        description="""<p><b>What it is.</b> Elise Chang's reward-trained network: a 100-unit leaky tanh RNN policy trained by REINFORCE with a
        learned value-network critic (Song, Yang and Wang 2017, her pyrl fork), on the pulse task with four one-hot previous-outcome input
        channels (<code>pagan_hist</code>), gaussian initialisation, gain 1.15, learning rate 0.004, no entropy bonus, rewards +1 / 0. The
        network is never shown the answer; it only sees whether it was rewarded, so it can learn strategies that are wrong but pay.</p>
        <p><b>What varies.</b> Two arms with identical initial weights per seed. <b>pretrained</b>: before the task, the Princeton rats' own
        nose-poke pre-training as extracted from their ProAnti3Marino sessions (fixed left for 360 trials, 20-trial side blocks for 1,080
        trials, random side for 720 trials); <b>no_pretrain</b>: the task from trial one. 2,800 updates of 72 trials = 201,600 trials, the
        order of a rat's career; checkpoints every 5 updates; early stop at 0.95 correct with 99% of trials decided. Seeds 1-16.</p>
        <p><b>Did they learn?</b> Yes, most of them, given rat-sized budgets: 13 of 16 without pre-training reach the 0.95 criterion (median
        about 79,000 trials), 8 of 16 with it (median about 97,000); the rest are at 0.5-0.85 after 201,600 trials. Nothing learns within the
        21,600 trials of the first pilot, which is why the budget was raised.</p>
        <p><b>How it is used in the talk.</b> This is the core of part 4: networks that can make mistakes get stuck like the rats. The
        pre-trained arm is the one that shows the rats' first stage (win-stay/lose-switch and a side bias) arising from the rats' own
        early training rather than from the task, and the price of it; the no_pretrain arm is the control with the same weights. The
        pair pages (same seed, two arms) are the figure.</p>""",
        findings="""<p><b>The rats' pre-training installs the win-stay/lose-switch habit and a side bias.</b> No network given the task alone
        acquires a WSLS weight; every pre-trained one does, together with a left bias, and both persist for roughly the first 100,000 trials.
        <b>The habit costs time</b>: half the pre-trained networks reach criterion within 201,600 trials against four fifths of the others, and in
        some seeds the side lock moves from one side to the other before it dissolves. The strategies the networks pass through (history, both
        features, selection) are the rats' stages, in the rats' order. <b>Mechanism</b>: a slow direction with leading eigenvalue near 1 is
        present from the first checkpoint, and the networks sit in the selection-vector corner of the triangle.</p>""")),
    ("antibias", "The trainer's anti-bias schedule: the lab's rule, and its inverse (talk part 4)", dict(
        description="""<p><b>What it is.</b> The same networks, with the correct side drawn by the lab's anti-bias rule (BControl's
        <code>recomputeAntibias</code>: an exponentially weighted hit rate per side, kernel tau 50 trials, then P(right correct) =
        softmax(-beta x [hitL, hitR]), beta 5), so the next correct side is pushed toward the side the network is getting wrong. The stimulus is
        drawn as usual and the relevant feature mirrored when the rule asks for the other side, so pulse statistics are unchanged; validation
        batches and all analysis rollouts are drawn 50/50.</p>
        <p><b>What varies.</b> The schedule (the lab's, beta 5; or inverted, beta -5, which pushes the correct side toward the side the network
        already chooses) crossed with the pre-training (with, without). Seeds 1-16 for beta 5; 1-8 or 1-16 for the inverted arms. Together with
        the base class this is a 2 x 3 design on identical initial weights.</p>
        <p><b>Did they learn?</b> With the lab's rule: 14 of 16 without pre-training (median about 68,000 trials) and 8 of 16 with. With the
        inverted rule: none, in either arm, in 201,600 trials; they stay one-sided from the first task trial to the last.</p>
        <p><b>How it is used in the talk.</b> Part 4's second half: we need the anti-bias to escape the win-stay/lose-switch and bias
        strategies. The lab's rule speeds the escape and rescues some networks; the inverted rule shows what the schedule does mechanically
        (a trainer that feeds the animal the side it already chooses removes the gradient). The slide is the population figure of this class
        (accuracy, |bias| and WSLS by arm) with the base arms as reference.</p>""",
        findings="""<p><b>The lab's rule speeds the escape and rescues some networks</b>: without pre-training 14/16 reach criterion against 13/16
        with no schedule, and the median escape moves from about 79,000 to 68,000 trials; with pre-training the count is unchanged (8/16).
        <b>The inverted rule freezes every network</b>, pre-trained or not. The rule halves the pre-trained networks' WSLS weight early on, but
        what decides escape is breaking the side lock.</p>""")),
    ("effort", "Effort (engagement) networks: attending a trial costs, declining is free (talk part 6)", dict(
        description="""<p><b>What it is.</b> Model 3. The same network with a fourth action, DECLINE, taken at the first step of a trial: declining
        ends the trial at once with reward 0 (and no history on the next trial, like a violation); accepting is charged EFFORT once, and the
        trial then runs as usual. Nothing about engagement is installed: the network learns when a trial is worth its effort, which is the
        normative version of the engagement state the evolutionary search found in the rats (part 5).</p>
        <p><b>What varies.</b> The cost (0.4, 0.55, 0.7; chance pays 0.5 per attempt, so above 0.5 a network that is not beating chance is
        better off declining) crossed with the pre-training (with, without). Seeds 1-8, initial weights identical to the base pairs (Wout
        is drawn last). 201,600 trials, continued to 403,200 on 12 Oct.</p>
        <p><b>Did they learn?</b> Slowly, and most not fully. One of 48 reached the full criterion within 201,600 trials (effort 0.55,
        pre-trained, seed 3); with the budget doubled to 403,200 trials (12 Oct) it is five of 48 (effort 0.40 no_pretrain 2, 0.55
        pretrained 1, 0.55 no_pretrain 1, 0.70 no_pretrain 1; the learners' median 240,000-340,000 trials), with median final accuracies of
        0.71-0.78 per arm and final attempt rates of 0.76-0.93. Several know the task (0.90-0.95 correct) but still decline 10-70% of
        trials, so they never satisfy the "99% decided" half of the criterion. The numbers in the table below are computed from the runs
        and are the ones to quote.</p>
        <p><b>How it is used in the talk.</b> Part 6: effort networks need the early training stages to be motivated, and the rest of the
        curriculum to stay motivated as the task gets harder. The first clause is <b>not yet shown</b> by this class: with the nose-poke
        pre-training alone, participation is the same as without it. The experiment that tests both clauses, the rats' stage curriculum
        (stages 1-6 then 7-10 with the quadrant anti-bias) on these networks, is the next thing to run. What this class already shows for
        the slide: participation follows competence, and competence and participation dissociate (networks that know the task yet decline).</p>""",
        findings="""<p><b>They learn, and they keep working.</b> Attempt rates sit at 0.75-0.8 for the first 100,000 trials and rise to 0.8-0.9 as the
        networks learn. Learning is slower than in the base networks, and the effort arms keep a side bias of 1.5-2 throughout, where the base
        networks lose theirs by 75,000 trials: the option to decline lets a side preference persist. <b>The nose-poke pre-training alone does not
        change participation</b> (attempt rate 0.77 vs 0.78 at EFFORT 0.55).</p>""")),
    ("stages", "Effort networks on the rats' stage curriculum (talk part 6)", dict(
        description="""<p><b>What it is.</b> The effort network (EFFORT 0.55, with the nose-poke pre-training) trained the way the Princeton rats
        are trained after the nose-poke stage: the TaskSwitch ladder read from the protocol code and from the pooled trial tables of 208 rats.
        Stages 1-2: the location task alone, easiest evidence, no frequency pulses. Stages 3-4: location and frequency in performance-gated
        blocks, each pure (the other feature's pulses absent). Stages 5-6: both features present, the irrelevant one uninformative (50/50 pulses).
        Stage 7: the irrelevant feature becomes informative and the quadrant anti-bias switches on (the next trial's quadrant of the
        stimulus matrix is drawn against the recent hit rate per quadrant, and the reward is scaled by 4 x p(quadrant), as the water is).
        Stages 8-9: harder irrelevant, then harder relevant evidence. Stage 10: the full task. Promotion as in the protocol: at least 150
        attempted trials in the stage and every task above its threshold (0.75 for stages 1-2, 0.7 for 3-6, 0.68 from 7); no demotion.
        Gamma to pulse strength: 4 &rarr; 0.25, 2.5 &rarr; 0.35, 1 &rarr; 0.45, 0 &rarr; 0.5.</p>
        <p><b>What varies.</b> Two arms, seeds 1-8, identical initial weights and stimulus seeds to the effort 0.55 pretrained arm, which is the
        no-stage reference (<i>none</i>): <b>stages 1-6 then full task</b> (accumulation only first, as on the Cosyne 2025 poster) and <b>stages
        1-10 with the quadrant anti-bias</b>. 201,600 trials, checkpoints every 5 updates. Every arm is measured on the same distribution: the
        rollouts force stage 10 and switch the quadrant rule off.</p>
        <p><b>Did they learn?</b> Running (launched 12 Oct 2026, 10:23, on mvm-lin-0095; the short arm relaunched at 10:33 without the
        quadrant anti-bias in its stage 10). First read at 300 updates (about 25,000 trials): in the full ladder two of eight networks had
        reached stage 10, one stage 8, one stage 7, one stage 4, three stage 3; promotions were earned (e.g. seed 8 left stage 3 after
        21,600 attempted trials at LOC 0.77 / FRQ 0.75, then passed stages 4-8 in 200-450 trials each at 0.72-0.85 on both tasks), and in
        stage 10 the training-stream accuracy falls to 0.47-0.63 as the hard strengths and the anti-bias arrive. The no-stage reference
        (effort 0.55 pretrained, same seeds) had six of eight at chance on the full task at the same point. Participation so far follows the
        seed, not the arm (seed 8 attempts about 0.55 in every arm, seed 6 0.35-0.67, the others 0.75-0.9). The ladder figure below is
        live; the full-task comparison arrives with the record rollouts.</p>
        <p><b>How it is used in the talk.</b> Part 6, both clauses: do the early stages make the effort networks engage (attempt rate) and do the
        later stages keep them engaged as the task gets harder, compared with the same networks thrown into the full task? This is the
        experiment the effort class points to.</p>""",
        findings="""<p>Nothing yet: the runs are in progress. The stage each network is in over training, and the promotion times against the
        rats' medians, will be shown here when the first records land.</p>""")),
    ("vanilla2025", "Vanilla RNNs, the Nature 2025 code (talk part 2)", dict(
        description="""<p><b>What it is.</b> The supervised networks of the Nature 2025 paper's own training code (JAX; 20 units; inputs feature 1,
        feature 2, context 1, context 2; scalar read-out; backpropagation through time with the original loss): 1,139 seeds, each with its weights
        saved at 120 checkpoints over training and its loss history (<code>learning_rats_RNNs/large_files/models20</code>). These are the networks
        of the Cosyne 2025 poster, with the same code as the paper.</p>
        <p><b>What varies.</b> Only the seed. (The N = 100 networks and the pre-training experiments of the poster are in other folders and
        can be added.)</p>
        <p><b>Did they learn?</b> By the loss: the final training loss is below 0.003 in 68% of the 1,139 seeds and the median is 0.0021
        (from 0.008-0.010 at the start). The behavioural read-out (psychometrics, performance matrix) needs the task generator ported from
        <code>flexible_decision_making_rnn/create_stim.m</code>, not done yet; the loss stands in for it on this page.</p>
        <p><b>How it is used in the talk.</b> Part 2: how vanilla RNNs learn (the line attractor forms, then the selection vector rotates between
        contexts), and the contrast with the reward-trained networks of part 4 under one slow-point method. The N = 20 variability result of the
        poster (learning speed and solution diversity) comes from this set.</p>""",
        findings="""<p>Computed at build time from the slow-point analysis of every seed (24 of the 120 checkpoints each): see the figures below.
        First reading on two seeds: the leading eigenvalue climbs to 1 early, the number of slow points grows into a line, and the triangle
        shares spread across the three mechanisms as in the paper. Biases are interpolated from 0 to their final value across checkpoints
        (only the final biases are saved), an approximation stated here.</p>""")),
    ("library", "Elise's 100-seed library (supporting; probably not shown)", dict(
        description="""<p><b>What it is.</b> Elise's own batch of 100 Approach A networks (<code>histgauss100</code>): the same stack and task as the
        base class, gaussian initialisation, no pre-training, 500 updates = 36,000 trials, checkpoints every 10 updates, seeds 1-100. Copied
        from her run folders on 10 Oct to replicate her population result.</p>
        <p><b>Did they learn?</b> Within 36,000 trials, one reaches the 0.95 criterion; 57 of 100 reach 0.75 (between 1,440 and 36,000 trials);
        43 never do. Final accuracy median 0.72.</p>
        <p><b>How it is used in the talk.</b> Supporting evidence for individual variability from the seed alone (n = 100), and the replication of
        Elise's result; probably not a slide, since the base class at 201,600 trials tells the learning story.</p>""",
        findings="""<p><b>Individual variability from the seed alone</b>: a 25-fold spread in the time to 0.75. The irrelevant-evidence weight is
        positive from the first thousands of trials: these networks become "both features" and stay there within this budget. No systematic
        WSLS without pre-training. Mechanism and critic computed for all 100.</p>""")),
    ("effort_v1", "Effort networks, first design (failed; not used)", dict(
        description="""<p><b>What it is.</b> The first effort model: declining was signalled with CHOOSE-LEFT/RIGHT at the first step (so the opt-out
        shared its output weights with the choice) and the cost was charged per stimulus step (so every FIXATE action carried a negative return).
        EFFORT 0.4 / 0.55 / 0.7, both arms, seeds 1-4 (0.4) or 1-8. 201,600 trials.</p>
        <p><b>Did they learn?</b> No network reached criterion; attempt rates collapsed to 20-30% within the first few hundred updates.</p>
        <p><b>How it is used in the talk.</b> Not used. Kept because the diagnosis (pushing toward declining was pushing toward one side; the
        per-step cost taught the network not to fixate) is what produced v2.</p>""",
        findings="""<p>Replaced by the effort class above.</p>""")),
    ("vanilla", "Vanilla supervised networks, NeuroGym version (not used in the talk)", dict(
        description="""<p><b>What it is.</b> Supervised networks trained in NeuroGym with torch (N = 20 and N = 100, 100 seeds each, the full-budget
        set of September-October), analysed with the same slow-point method as the RL networks. Mechanism only; no behaviour pages.</p>
        <p><b>How it is used in the talk.</b> Not used: the talk uses the networks trained with the Nature 2025 code (class above). This page is
        kept for the record and for the method check (the comparison figure below).</p>""",
        findings="""<p>Vanilla networks build a line attractor during training, end with selection vectors 30-50 degrees apart and spread across the
        triangle; the reward-trained networks keep about 10 slow points, have an eigenvalue at or above 1 from the first checkpoint, selection
        vectors 80-100 degrees apart, and sit in the selection-vector corner.</p>"""))]

# Classes documented in Elise's tree (see TREES below) rather than on the original front page.
CLASSES += [
    ("e11", "11. Per-timestep engagement cost and the duration schedule", dict(
        description="""<p><b>What it gets.</b> The rats' nose-poke pre-training, then the full task, with
        engagement charged by the timestep. A fourth action, decline, is available at <i>every</i> timestep and
        ends the trial at once for no reward; every timestep of the stimulus-presentation window on which the
        network does not decline costs 0.55/130, so holding the window at its full length costs 0.55 (the lump
        cost of the effort-v2 networks) and a trial abandoned part-way costs only what it used. Whatever has
        been paid is never refunded. There is no fixation requirement: a left/right action during the window is
        masked from the reward and does not end the trial, as in the base task. Initial action probabilities are
        not handcrafted.</p>
        <p><b>What varies across the arms.</b> The presentation window is one step long through the
        pre-training, and the four arms differ only in how it grows afterwards: not at all (straight to the full
        1,300 ms at the hand-off, the control); +50 ms at every checkpoint regardless of performance; +50 ms only
        when P(choice) is at least 0.70; +50 ms only when P(choice) is at least 0.70 and accuracy on the trials
        it chose is above 0.55. Eight seeds per arm, the same seeds across arms and across the effort-v2 arms,
        up to 201,600 trials. Scored on the five trial outcomes - chose, declined on the first timestep,
        declined inside the window, declined after it, timed out - as well as on accuracy among the trials
        chosen.</p>
        <p>This replaces the fixation-cost networks (<code>pagan_hist_fixcost</code>,
        <code>pagan_hist_stages3</code>) that were tried on 13 Oct and set aside: those had no decline output, so
        withdrawing and choosing shared the same two actions and the policy collapsed into declining
        everything.</p>""",
        findings="""<p>Not yet run.</p>""")),
    ("e12", "12. Per-timestep cost, slimmed local run", dict(
        description="""<p><b>What it gets.</b> The same network and the same cost as type 11 - a decline output
        available at every timestep, 0.55/130 charged per timestep it is not chosen, no fixation requirement, the
        presentation window one step through the pre-training and then grown by 50 ms per checkpoint from a 50 ms
        start. Eight seeds per arm, up to 201,600 trials.</p>
        <p><b>How it differs from type 11.</b> Trimmed to run on a laptop while the workstation was busy. The
        no-schedule control is dropped, because the smoke test had already shown what it does (jumping straight to
        the full window takes the fraction of trials reaching a choice from 0.996 to 0.000 in one checkpoint, almost
        all of it mid-window withdrawal); and the validation batch is 360 trials rather than 720, which is where the
        saving is, a checkpoint's validation batch being the larger part of the per-update cost. Three arms remain:
        the unconditional schedule, the P(choice)-gated one, and the performance-gated one.</p>
        <p class="caveat"><b>Not comparable seed-for-seed with type 11.</b> The same weight seed gives the same
        starting weights, but validation draws consume the same stimulus RNG stream as training draws, so a
        360-trial batch puts every later trial out of step with the 720-trial run. The two runs answer the same
        question on separate samples; do not pool them. The smaller batch also halves the precision of the
        curriculum gates: with about 359 decided trials the standard error of accuracy at chance is 0.026 against
        0.019, so a network exactly at chance clears the performance arm's 0.55 gate on 2.9% of checkpoints rather
        than 0.4% - roughly 16 growths' worth of noise over a full run, against the 25 that complete the schedule.
        Read that arm's schedule trace alongside its accuracy, not on its own.</p>""",
        findings="""<p>Not yet run.</p>""")),
    ("e13", "13. Per-timestep cost and the full stage ladder", dict(
        description="""<p><b>What it gets.</b> The same network and cost as types 11 and 12 - a decline output
        available at every timestep, 0.55/130 charged per timestep it is not chosen, no fixation requirement - but
        where those hand off from the nose-poke pre-training straight to the full task, this hands off to the rats'
        10-stage TaskSwitch ladder: location task only on the easiest evidence, then both tasks in
        performance-gated blocks with the other feature's pulses absent, then both features present with the
        irrelevant one uninformative, then the irrelevant one informative and increasingly hard, then the full
        task. The lab's side anti-bias is on in stages 1-6 and the quadrant anti-bias from stage 7, with the
        protocol's reward scaling by 4 x p(quadrant). Promotion needs 150 attempted trials in the stage and every
        active task above the stage's threshold; there is no demotion, so stage lengths are the network's own
        rather than the rats' medians. A block also ends at twice the rats' median length for the stage, because
        the protocol relies on a trainer to end a block a stuck animal never ends.</p>
        <p><b>The duration schedule runs in stage 1 only, and once in all of training.</b> The presentation window
        is one step through the pre-training, starts at 50 ms in stage 1 and grows by 50 ms per checkpoint while
        P(choice) is at least 0.70. No stage can be left while the window is still growing, so the schedule cannot
        outlive stage 1; once the window is full the schedule is a permanent no-op. Expect stage 1 to hold until
        about update 155 of 2,800. If the schedule stalls, the seed stays in stage 1 for the whole run - which is
        itself the result, the ladder never having started because engagement never scaled.</p>
        <p class="caveat">Two things to read carefully. A trial counts as attempted only if it ended in a choice,
        so a network that declines most trials is slowed toward promotion but never blocked - stage duration and
        engagement are confounded, and the stage trace has to be read alongside the fraction of trials attempted.
        And the quadrant reward scaling multiplies the terminal reward by 4 x p(quadrant) while the engagement cost
        is charged separately on earlier timesteps, so an attempted correct trial in stages 7-10 nets 4p - 0.55,
        negative for a rare quadrant: the expected value of engaging varies by quadrant there.</p>""",
        findings="""<p>Not yet run.</p>""")),
    ("e14", "14. No pre-training, no duration schedule (control)", dict(
        description="""<p><b>What it gets.</b> The same per-timestep cost as types 11-13 - a decline output at
        every timestep, 0.55/130 charged per engaged timestep, no fixation requirement - but with no nose-poke
        pre-training, no stage ladder and no duration schedule: the presentation window is the full 1,300 ms from
        the very first trial. This is the control deferred from the first run's requirements ("comparison with no
        pre-training ... is deferred for following runs"), and it isolates what the pre-training window and the
        duration schedule are actually buying the other arms.</p>
        <p><b>Why max_iter is cut to 1,000.</b> At initialisation P(DECLINE) is 1/4 per timestep, so
        P(reaching a potent choice) = 0.75^130 is astronomically small, and there is neither a pre-training phase
        to learn to hold a short fixation in nor a schedule to grow the window gradually - so every seed is
        expected to collapse to zero engagement from the first checkpoint and stay there. Confirmed in a 40-update
        smoke: P(choice) = 0.000 from update 0, essentially all of it mid-window withdrawal at a constant median
        step of 3. Running the full 2,800-update budget would not add information, so it is cut to 1,000 and
        n_validation is left at the model's own default (720) for a well-resolved trajectory.</p>
        <p class="caveat">A network that never engages is the expected, correct result here, not a failure of the
        model or the run.</p>""",
        findings="""<p>Not yet run.</p>"""))]

CLASS_OF_ARM = {a: c for _, _, a, c in ARMS}

# -----------------------------------------------------------------------------------------
# Documentation trees
#
# Two static trees out of one builder, one ARMS table and one per-run cache: the original (the
# seven types of the 13 Oct talk) at the site root, and Elise's, at <root>/elise, where this and
# the following runs are documented. `--tree` picks one; each gets its own index.html and
# archive.html, and only the runs of its own classes.
# -----------------------------------------------------------------------------------------

ELISE_KEYS = ("e11", "e12", "e13", "e14")
MAIN_KEYS = tuple(k for k, _, _ in CLASSES if k not in ELISE_KEYS)

TREES = {
    "main": dict(
        sub="", title="Network explorer", index_title="Models",
        keys=MAIN_KEYS, front=FRONT,
        lead_note="The seven types of reward-trained network used in the talk, all started with weak recurrent connections. Click a type for what it gets, its numbers, its figures, its mechanism, and the entry into the one-by-one view (arrow keys loop within the class).",
        archive_note='<a href="archive.html">Other and earlier networks</a>: the weak-start controls, the strong-start networks trained before 13 Oct 2026, and the supervised networks.',
        archive_blurb="Not used in the talk. The strong-start classes were trained with Elise's default starting weights, which were found on 13 Oct 2026 to give a broad slow region instead of a line and to halve the learning speed.",
        home_extra='<a href="elise/index.html">Elise\'s runs</a>: the engagement and curriculum runs from 16 Oct 2026 on, documented in their own tree.',
        nav_extra=()),
    "elise": dict(
        sub="elise", title="Elise's runs", index_title="Elise's runs",
        keys=ELISE_KEYS, front=ELISE_KEYS,
        lead_note="The engagement and curriculum runs from 16 Oct 2026 on. Click a type for what it gets, its numbers, its figures, its mechanism, and the entry into the one-by-one view (arrow keys loop within the class).",
        archive_note='<a href="archive.html">Earlier and set-aside versions</a> of these runs.',
        archive_blurb="Versions of these experiments that were superseded or set aside; kept so the reason is on the record.",
        home_extra='<a href="../index.html">All the earlier networks</a>: the seven types of the 13 Oct talk and everything on their archive page.',
        nav_extra=(("../index.html", "All models"),)),
    }

# Rebound by main() to the tree being built.
TREE = TREES["main"]


def display_name(run):
    """Run names on disk are pyrl's (model file + suffix); the site uses names that need no key.
    The arms are called pretrained and no_pretrain, not "control", which is ambiguous."""
    for pre, disp, _, _ in ARMS:
        if run.startswith(pre):
            return disp + run[len(pre):]
    return run


def arm_of(run):
    for pre, _, arm, _ in ARMS:
        if run.startswith(pre):
            return arm
    return "other"


def is_pretrained(run):
    return "curr_" in run and "ctrl" not in run


class _Tolerant(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            return type(name, (object,), {"__setstate__": lambda self, s: self.__dict__.update(s)})


def load_pkl(path):
    with open(path, "rb") as f:
        return _Tolerant(f, encoding="latin1").load()


# ------------------------------------------------------------------------------- per network

def compute(run, site):
    """Everything numeric for one network, cached as <site>/cache/<run>.npz."""
    cache = os.path.join(site, "cache", run + ".npz")
    rec = os.path.join(RECORDS, run + "_record.npz")
    if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(rec):
        z = np.load(cache, allow_pickle=True)
        if "cache_version" in z.files and int(z["cache_version"]) == CACHE_VERSION:
            return {k: z[k] for k in z.files}
    z = np.load(rec)
    y, session, iters = z["y"], z["session"], z["iters"]
    X = {k[2:]: z[k] for k in z.files if k.startswith("X_")}
    names = sorted(X.keys())
    M = np.column_stack([X[k] for k in names])
    has_raw = "raw_context" in z.files
    cks = np.unique(session)
    out = {"run": run, "names": np.array(names), "iter": np.array([iters[session == s][0] for s in cks], float)}
    W, SE, pmaj, acc, pstay_w, pstay_l, pright, n = [], [], [], [], [], [], [], []
    acc_loc, acc_frq, acc_sel = [], [], []
    if has_raw:
        # "strong" evidence = above the median |evidence| of that feature over the record; a
        # strongly incongruent trial has strong relevant and strong irrelevant evidence pointing to
        # opposite sides: the poster's feature-selection cells (perf(1,6) and perf(6,1))
        rel_all, irr_all = z["raw_rel_raw"], z["raw_irr_raw"]
        strong_rel = np.abs(rel_all) >= np.median(np.abs(rel_all[rel_all != 0]))
        strong_irr = np.abs(irr_all) >= np.median(np.abs(irr_all[irr_all != 0]))
        incong = strong_rel & strong_irr & (np.sign(rel_all) != np.sign(irr_all))
    for s in cks:
        m = session == s
        w, se = ridge_logistic(M[m], y[m], 1.0)
        W.append(w); SE.append(se)
        ym = y[m]
        pmaj.append(max(ym.mean(), 1 - ym.mean())); n.append(int(m.sum())); pright.append(ym.mean())
        hc, ws = X["histChoice"][m], X["wsls"][m]
        prev_win = hc * ws                                   # +1 previous trial rewarded, -1 not, 0 none
        stay = (2 * ym - 1) == hc
        pstay_w.append(stay[prev_win > 0].mean() if (prev_win > 0).any() else np.nan)
        pstay_l.append(stay[prev_win < 0].mean() if (prev_win < 0).any() else np.nan)
        if has_raw:
            rel = z["raw_rel_raw"][m]
            ok = rel != 0
            correct = (ym == 1) == (rel > 0)
            acc.append(correct[ok].mean() if ok.any() else np.nan)
            cm = z["raw_context"][m]
            acc_loc.append(correct[ok & (cm == 0)].mean() if (ok & (cm == 0)).any() else np.nan)
            acc_frq.append(correct[ok & (cm == 1)].mean() if (ok & (cm == 1)).any() else np.nan)
            im = incong[m]
            acc_sel.append(correct[im].mean() if im.sum() >= 10 else np.nan)
        else:
            acc.append(np.nan); acc_loc.append(np.nan); acc_frq.append(np.nan); acc_sel.append(np.nan)
    if "raw_ck_n_total" in z.files:
        # opportunities and attempts per checkpoint rollout: the attempt rate (effort networks
        # decline trials; every other arm attempts all of them)
        tot = dict(zip(z["raw_ck_iters"].astype(int), z["raw_ck_n_total"])); cho = dict(zip(z["raw_ck_iters"].astype(int), z["raw_ck_n_choice"]))
        out["attempt_rate"] = np.array([cho.get(int(i), np.nan) / max(1, tot.get(int(i), 1)) for i in out["iter"]])
    else:
        out["attempt_rate"] = np.full(len(cks), np.nan)
    out.update(W=np.array(W).T, SE=np.array(SE).T, p_major=np.array(pmaj), n_trials=np.array(n),
               task_acc=np.array(acc), p_stay_win=np.array(pstay_w), p_stay_lose=np.array(pstay_l),
               p_right=np.array(pright), xstd=M.std(0), acc_loc=np.array(acc_loc), acc_frq=np.array(acc_frq),
               acc_sel=np.array(acc_sel), cache_version=CACHE_VERSION)
    # training history: validation accuracy, mean reward, P(choice) at every checkpoint
    hist = {}
    mp = os.path.join(DATA, run, run + ".pkl")
    model = None
    if os.path.exists(mp):
        model = load_pkl(mp)
        for e in model["training_history"]:
            pf = e["perf"]
            dec = np.asarray(pf.decisions, bool); cor = np.asarray(pf.corrects, float)
            hist[int(e["iter"])] = (float(np.nanmean(cor[dec])) if dec.any() else np.nan, float(e["mean_reward"]), dec.mean())
    out["val_acc"] = np.array([hist.get(int(i), (np.nan,) * 3)[0] for i in out["iter"]])
    # criterion reached = the run stopped early (pyrl's terminate: 0.95 correct, 99% decided, on
    # the validation batch of the TASK; the pre-training phases cannot trigger it because
    # terminate() is off outside the task phase). Read from the savefile: iter < max_iter.
    # criterion reached = pyrl's terminate() held at the run's last validation (0.95 correct and 99%
    # decided), which is why it stopped; robust to runs later continued with a larger max_iter.
    ci = -1
    if hist:
        last_it = max(hist); acc_l, _, dec_l = hist[last_it]
        if np.isfinite(acc_l) and acc_l >= 0.95 and dec_l >= 0.99:
            ci = last_it
    out["criterion_iter"] = np.array([ci])
    out["mean_reward"] = np.array([hist.get(int(i), (np.nan,) * 3)[1] for i in out["iter"]])
    out["p_choice"] = np.array([hist.get(int(i), (np.nan,) * 3)[2] for i in out["iter"]])
    # psychometrics and matrices per checkpoint, from the raw evidence
    if has_raw:
        ctx, el, ef = z["raw_context"], z["raw_e_loc"], z["raw_e_frq"]
        edges_l = np.quantile(el, np.linspace(0, 1, 8)); edges_f = np.quantile(ef, np.linspace(0, 1, 8))
        edges_l[0] -= 1e-9; edges_f[0] -= 1e-9
        bl = np.clip(np.searchsorted(edges_l, el) - 1, 0, 6); bf = np.clip(np.searchsorted(edges_f, ef) - 1, 0, 6)
        # matrices: 5 x 5 quantile bins per context
        q5l = np.quantile(el, np.linspace(0, 1, 6)); q5f = np.quantile(ef, np.linspace(0, 1, 6)); q5l[0] -= 1e-9; q5f[0] -= 1e-9
        ml = np.clip(np.searchsorted(q5l, el) - 1, 0, 4); mf = np.clip(np.searchsorted(q5f, ef) - 1, 0, 4)
        psy = np.full((len(cks), 2, 2, 7), np.nan)     # ckpt, context, feature (loc, frq), bin
        mat = np.full((len(cks), 2, 5, 5), np.nan)     # ckpt, context, loc bin, frq bin
        for si, s in enumerate(cks):
            m = session == s
            for c in (0, 1):
                mc = m & (ctx == c)
                for b in range(7):
                    sel = mc & (bl == b)
                    if sel.sum() >= 5: psy[si, c, 0, b] = y[sel].mean()
                    sel = mc & (bf == b)
                    if sel.sum() >= 5: psy[si, c, 1, b] = y[sel].mean()
                for a in range(5):
                    for b in range(5):
                        sel = mc & (ml == a) & (mf == b)
                        if sel.sum() >= 3: mat[si, c, a, b] = y[sel].mean()
        out.update(psy=psy, mat=mat, psy_centres_loc=0.5 * (edges_l[1:] + edges_l[:-1]),
                   psy_centres_frq=0.5 * (edges_f[1:] + edges_f[:-1]), mat_edges_loc=q5l, mat_edges_frq=q5f)
    # weights across checkpoints: per-channel input norms, recurrent norm and spectral radius, critic too
    cps = sorted(glob.glob(os.path.join(DATA, run, "checkpoints", "iter_*.pkl")))
    rows, dws = [], []
    prev = None
    for cp in cps:
        d = load_pkl(cp)
        pp, bp = d["policy_params"], d["baseline_params"]
        Win, Wrec, Wout = np.asarray(pp["Win"]), np.asarray(pp["Wrec"]), np.asarray(pp["Wout"])
        bWrec = np.asarray(bp["Wrec"])
        rows.append([d["iter"]] + list(np.linalg.norm(Win, axis=1)) +
                    [np.linalg.norm(Wrec), np.abs(np.linalg.eigvals(Wrec)).max(), np.linalg.norm(Wout),
                     np.linalg.norm(bWrec), np.abs(np.linalg.eigvals(bWrec)).max(), np.linalg.norm(np.asarray(bp["Win"]))])
        # instantaneous recurrent weight change, as on the poster: Frobenius norm of the change
        # in Wrec since the previous checkpoint, per update, for the actor and for the critic
        if prev is not None:
            gap = max(1, d["iter"] - prev[0])
            dws.append([d["iter"], np.linalg.norm(Wrec - prev[1]) / gap, np.linalg.norm(bWrec - prev[2]) / gap])
        prev = (d["iter"], Wrec, bWrec)
    out["weights"] = np.array(rows) if rows else np.zeros((0, 15))
    out["dw"] = np.array(dws) if dws else np.zeros((0, 3))
    # PsyTrack, if a fit exists: per-checkpoint mean of each weight
    for d in (PSY, os.path.join(PSY, "farm")):
        p = os.path.join(d, run + "_psytrack.npz")
        if os.path.exists(p):
            # align by iteration: a PsyTrack fit may cover fewer checkpoints than the record
            # (the fits were made at 300 updates; the records now run to 2800)
            pz = np.load(p)
            pn = [str(nn) for nn in pz["names"]]
            ps, pit = pz["session"], pz["iters"]
            sess_of_iter = {int(pit[ps == s][0]): s for s in np.unique(ps)}
            P = np.full((len(names), len(out["iter"])), np.nan)
            for j, it in enumerate(out["iter"]):
                s = sess_of_iter.get(int(it))
                if s is not None:
                    P[:, j] = [pz["wMode"][pn.index(nm), ps == s].mean() for nm in names]
            out["psytrack"] = P
            break
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, **out)
    return out


# ------------------------------------------------------------------------------------ figures

def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcdefaults()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _shade(ax, is_curr, x_is_trials=True):
    if not is_curr:
        return
    for name, a, b, c in PHASES:
        ax.axvspan(a, b, color=c, zorder=0, lw=0)
        ax.axvline(b, color="red", ls="--", lw=0.7, zorder=1)


def _strip_idx(n):
    return sorted(set(np.linspace(0, n - 1, N_STRIP).round().astype(int).tolist()))


def _robust_lim(arrs, cap=8.0):
    v = np.concatenate([a[np.isfinite(a)] for a in arrs]) if arrs else np.array([1.0])
    m = min(cap, max(1.0, 1.1 * np.percentile(np.abs(v), 99))) if len(v) else 1.0
    return (-m, m)


def phase_legend(ax, loc="upper left", grey=False):
    """A second legend naming the shadings: the three pre-training phases (blue fixed-left,
    green side blocks, orange random side; white = the task), and grey where the GLM is not
    identifiable. Added as an extra artist so the panel's own legend survives."""
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, edgecolor="#999", label="%s (%d trials)" % (name, b - a)) for name, a, b, c in PHASES]
    handles.append(Patch(facecolor="white", edgecolor="#999", label="task"))
    if grey:
        handles.append(Patch(facecolor="#dddddd", edgecolor="#999", label=">95% one choice: weights not identifiable"))
    leg = ax.legend(handles=handles, fontsize=7, loc=loc, framealpha=0.9, title="pre-training (shading)", title_fontsize=7)
    ax.add_artist(leg)


def phase_legend_grey_only(ax):
    from matplotlib.patches import Patch
    leg = ax.legend(handles=[Patch(facecolor="#dddddd", edgecolor="#999", label=">95% one choice: weights not identifiable")],
                    fontsize=7, loc="lower left", framealpha=0.9)
    ax.add_artist(leg)


def ax_performance(ax, r, is_curr, ls="-", label_arm="", legend=True):
    """The poster's three curves: location, frequency, feature selection."""
    x = r["iter"] * N_GRADIENT
    _shade(ax, is_curr)
    if is_curr and legend:
        phase_legend(ax)
    ax.plot(x, 100 * r["acc_loc"], color="k", lw=1.8, ls=ls, label="Location" + label_arm)
    ax.plot(x, 100 * r["acc_frq"], color="tab:blue", lw=1.8, ls=ls, label="Frequency" + label_arm)
    ax.plot(x, 100 * r["acc_sel"], color="tab:green", lw=1.8, ls=ls, label="Feature selection" + label_arm)
    ar = r.get("attempt_rate", np.full(len(x), np.nan)) if hasattr(r, "get") else np.full(len(x), np.nan)
    pc = r.get("p_choice", np.full(len(x), np.nan)) if hasattr(r, "get") else np.full(len(x), np.nan)
    if np.isfinite(pc).any() and np.nanmin(pc) < 0.98:
        # trials attempted per 100 opportunities: the "trials per session" readout (RAS metric)
        ax.plot(x, 100 * pc, color="tab:purple", lw=1.8, ls=ls, label="Attempted (% of opportunities)" + label_arm)
    ax.axhline(50, color="k", ls=":", lw=0.8); ax.set_ylim(0 if np.isfinite(pc).any() and np.nanmin(pc) < 0.98 else 35, 102); ax.set_ylabel("Performance (%)")
    if legend:
        ax.legend(fontsize=8, loc="lower right", framealpha=0.9, ncol=1)


def ax_strategy(ax, r, is_curr, title=None):
    """All five GLM weights on one axis, the rats' way: colours, +/- 1 SE bands, zero line, grey
    where not identifiable, phase shading, PsyTrack dotted where it exists."""
    names = [str(n) for n in r["names"]]
    x = r["iter"] * N_GRADIENT
    bad = r["p_major"] > P_MAJOR_MAX
    Wm = r["W"].copy(); Wm[:, bad] = np.nan
    _shade(ax, is_curr)
    half = N_GRADIENT * (np.median(np.diff(r["iter"])) if len(x) > 1 else 5) / 2
    for j in np.where(bad)[0]:
        ax.axvspan(x[j] - half, x[j] + half, color="#dddddd", zorder=0, lw=0)
    order = [("rel", "relevant"), ("irr", "irrelevant"), ("bias", "bias"), ("histChoice", "prev. choice"), ("wsls", "win-stay/lose-switch")]
    for nm, lab in order:
        i = names.index(nm)
        ax.plot(x, Wm[i], color=CLR[nm], lw=1.6, label=lab)
        ax.fill_between(x, Wm[i] - r["SE"][i], Wm[i] + r["SE"][i], color=CLR[nm], alpha=0.22, lw=0)
        if "psytrack" in r:
            ax.plot(x, r["psytrack"][i], color=CLR[nm], lw=0.8, ls=":", alpha=0.9)
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_ylim(*_robust_lim([Wm])); ax.set_ylabel("Logistic weight")
    if is_curr or bad.any():
        phase_legend(ax, loc="lower left", grey=bad.any()) if is_curr else phase_legend_grey_only(ax)
    ax.legend(fontsize=7.5, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), framealpha=0.9, handlelength=1.4, columnspacing=0.9, borderaxespad=0.2)
    if title:
        ax.set_title(title, fontsize=9, loc="left", pad=22)


def ax_strips(fig_, gs_row, r, ks, kind, label_left):
    """A row pair of tiny panels (LOC context above, FRQ context below) at the sampled checkpoints.
    kind = 'mat' (choice matrices, imshow) or 'psy' (psychometric curves)."""
    import matplotlib.gridspec as gridspec
    sub = gridspec.GridSpecFromSubplotSpec(2, len(ks), subplot_spec=gs_row, wspace=0.08, hspace=0.12)
    for c in (0, 1):
        for j, k in enumerate(ks):
            ax = fig_.add_subplot(sub[c, j])
            if kind == "mat":
                ax.imshow(r["mat"][k, c].T, origin="lower", vmin=0, vmax=1, cmap="RdBu_r", aspect="auto", interpolation="nearest")
            else:
                rel_feat = 0 if c == 0 else 1
                for feat, centres in ((0, r["psy_centres_loc"]), (1, r["psy_centres_frq"])):
                    col = "tab:red" if feat == rel_feat else "tab:green"
                    ax.plot(centres, r["psy"][k, c, feat], "-", color=col, lw=1.3)
                ax.set_ylim(-0.05, 1.05); ax.axhline(0.5, color="k", ls=":", lw=0.5); ax.axvline(0, color="k", ls=":", lw=0.5)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.5)
            if c == 1:
                ax.set_xlabel("%dk" % round(r["iter"][k] * N_GRADIENT / 1000), fontsize=7, labelpad=1)
            if j == 0:
                ax.set_ylabel(["LOC ctx", "FRQ ctx"][c], fontsize=7)
                if c == 0:
                    ax.set_title(label_left, fontsize=8, loc="left")


def ax_weight_change(ax, r, is_curr, ls="-", label_arm="", legend=True):
    _shade(ax, is_curr)
    if len(r["dw"]):
        x = r["dw"][:, 0] * N_GRADIENT
        ax.plot(x, r["dw"][:, 1], color="k", lw=1.6, ls=ls, label="actor" + label_arm)
        ax.plot(x, r["dw"][:, 2], color="tab:brown", lw=1.6, ls=ls, label="critic" + label_arm)
    ax.set_ylabel("Recurrent weight change\n(|dWrec| per update)"); ax.set_ylim(bottom=0)
    if legend:
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9)


def ax_radius(ax, r, is_curr, ls="-", label_arm="", legend=True):
    _shade(ax, is_curr)
    Wt = r["weights"]
    if len(Wt):
        x = Wt[:, 0] * N_GRADIENT
        ax.plot(x, Wt[:, 10], color="k", lw=1.6, ls=ls, label="actor" + label_arm)
        ax.plot(x, Wt[:, 13], color="tab:brown", lw=1.6, ls=ls, label="critic" + label_arm)
    ax.axhline(1, color="k", ls="--", lw=0.8)
    ax.set_ylabel("Spectral radius of Wrec"); ax.set_ylim(bottom=0)
    if legend:
        ax.legend(fontsize=8, loc="lower right", framealpha=0.9)


def fig_poster(r, out, is_curr):
    """One network on one page, poster style: performance and strategy on top; strips of choice
    matrices and psychometrics at sampled checkpoints; recurrent weight change and spectral
    radius for actor and critic at the bottom. No interaction, nothing to scroll."""
    plt = _style()
    import matplotlib.gridspec as gridspec
    fig_ = plt.figure(figsize=(17, 10.2))
    gs = gridspec.GridSpec(4, 2, figure=fig_, height_ratios=[1.35, 0.95, 0.95, 1.0], hspace=0.45, wspace=0.18,
                           left=0.05, right=0.99, top=0.92, bottom=0.06)
    ax = fig_.add_subplot(gs[0, 0]); ax_performance(ax, r, is_curr); ax.set_title("Behaviour over training (location, frequency, feature selection)", fontsize=9, loc="left", pad=22)
    ax.set_xlabel("training trials")
    ax = fig_.add_subplot(gs[0, 1]); ax_strategy(ax, r, is_curr, "Strategy: ridge GLM per checkpoint, +/- 1 SE (dotted: PsyTrack; grey: not identifiable)"); ax.set_xlabel("training trials")
    ks = _strip_idx(len(r["iter"]))
    if "mat" in r:
        ax_strips(fig_, gs[1, :], r, ks, "mat", "Choice matrices: P(right) over location (x) by frequency (y) evidence, per context, at sampled checkpoints (k training trials)")
        ax_strips(fig_, gs[2, :], r, ks, "psy", "Psychometrics: P(right) against the relevant (red) and irrelevant (green) evidence, per context")
    ax = fig_.add_subplot(gs[3, 0]); ax_weight_change(ax, r, is_curr); ax.set_xlabel("training trials")
    ax = fig_.add_subplot(gs[3, 1]); ax_radius(ax, r, is_curr); ax.set_xlabel("training trials")
    fig_.suptitle("%s   (%s, seed %d)" % (display_name(str(r["run"])), arm_of(str(r["run"])), int(re.search(r"_w(\d+)", str(r["run"])).group(1))),
                  fontweight="bold", fontsize=13, x=0.05, y=0.985, ha="left")
    fig_.savefig(out, dpi=105); plt.close(fig_)


def fig_pair(ra, rb, out, la="pretrained", lb="no_pretrain"):
    """Arm A (ra, solid, shaded phases if pre-trained) and arm B (rb, dashed) with the same seed,
    the same initial weights, overlaid: the poster's bottom row, plus matrices and strategy side by side."""
    plt = _style()
    import matplotlib.gridspec as gridspec
    seed = int(re.search(r"_w(\d+)", str(ra["run"])).group(1))
    sa, sb = is_pretrained(str(ra["run"])), is_pretrained(str(rb["run"]))
    fig_ = plt.figure(figsize=(17, 10.2))
    gs = gridspec.GridSpec(5, 2, figure=fig_, height_ratios=[1.35, 0.8, 0.8, 1.2, 1.0], hspace=0.55, wspace=0.18,
                           left=0.05, right=0.99, top=0.93, bottom=0.06)
    ax = fig_.add_subplot(gs[0, 0])
    ax_performance(ax, ra, sa, "-", " (%s)" % la, legend=False)
    ax_performance(ax, rb, sb and not sa, "--", " (%s)" % lb, legend=False)
    if sa or sb: phase_legend(ax)
    ax.legend(fontsize=7.5, loc="lower right", ncol=2, framealpha=0.9); ax.set_title("Behaviour: %s (solid) vs %s (dashed), same initial weights" % (la, lb), fontsize=10, loc="left")
    ax.set_xlabel("training trials")
    ax = fig_.add_subplot(gs[0, 1]); ax_radius(ax, ra, sa, "-", " (%s)" % la, legend=False); ax_radius(ax, rb, False, "--", " (%s)" % lb, legend=False)
    ax.legend(fontsize=8, loc="lower right", ncol=2, framealpha=0.9); ax.set_xlabel("training trials")
    ks_a, ks_b = _strip_idx(len(ra["iter"])), _strip_idx(len(rb["iter"]))
    if "mat" in ra and "mat" in rb:
        ax_strips(fig_, gs[1, :], ra, ks_a, "mat", "Choice matrices, " + la)
        ax_strips(fig_, gs[2, :], rb, ks_b, "mat", "Choice matrices, " + lb)
    axa = fig_.add_subplot(gs[3, 0]); ax_strategy(axa, ra, sa, "Strategy, " + la)
    axb = fig_.add_subplot(gs[3, 1], sharey=axa); ax_strategy(axb, rb, sb, "Strategy, " + lb)
    lim = _robust_lim([ra["W"], rb["W"]]); axa.set_ylim(*lim)
    ax = fig_.add_subplot(gs[4, 0]); ax_weight_change(ax, ra, sa, "-", " (%s)" % la, legend=False); ax_weight_change(ax, rb, False, "--", " (%s)" % lb, legend=False)
    ax.legend(fontsize=8, loc="upper right", ncol=2, framealpha=0.9); ax.set_xlabel("training trials")
    ax = fig_.add_subplot(gs[4, 1])
    # the two arms' accuracy difference, to read the cost or benefit of pre-training directly
    xa, xb = ra["iter"] * N_GRADIENT, rb["iter"] * N_GRADIENT
    common = np.intersect1d(xa, xb)
    da = np.interp(common, xa, 100 * r_or_nan(ra["task_acc"])); db = np.interp(common, xb, 100 * r_or_nan(rb["task_acc"]))
    _shade(ax, sa)
    ax.plot(common, da - db, color="tab:purple", lw=1.8)
    ax.axhline(0, color="k", ls="--", lw=0.8); ax.set_ylabel("P(correct), %s\nminus %s (%%)" % (la, lb)); ax.set_xlabel("training trials")
    fig_.suptitle("seed %d: %s vs %s" % (seed, la, lb), fontweight="bold", fontsize=13, x=0.05, y=0.985, ha="left")
    fig_.savefig(out, dpi=105); plt.close(fig_)


def r_or_nan(a):
    a = np.asarray(a, float)
    return np.where(np.isfinite(a), a, np.nan)


def load_mech(run):
    p = os.path.join(MECH, run + "_policy_mechanism.npz")
    return dict(np.load(p, allow_pickle=True)) if os.path.exists(p) else None


def load_act(run):
    p = os.path.join(ACT, run + "_activity.npz")
    return dict(np.load(p, allow_pickle=True)) if os.path.exists(p) else None


def fig_mechanism(run, out, is_curr):
    """Second screen of a network page: actor dynamics and critic across training.
    Row 1: slow points per context projected on the actor's PCs at sampled checkpoints (the
    poster's bottom row). Row 2: leaky leading eigenvalue per context, selection-vector angle,
    the triangle coordinates (feature 1) over training. Row 3: the critic: value at trial start
    and end and the terminal prediction error; the value trajectory on hit versus miss trials at
    the last checkpoint; how much the critic reads the actor's evidence axis, and both networks'
    dimensionality."""
    mech, act = load_mech(run), load_act(run)
    if mech is None and act is None:
        return False
    if mech is not None:
        # a run on which no checkpoint yielded a slow-point pair has no eigenvalue columns at all
        n = len(mech["iter"])
        for k in ("eig1_leaky", "eig2_leaky", "sel_angle"):
            if k not in mech: mech[k] = np.full(n, np.nan)
        if "tri1" not in mech: mech["tri1"] = np.array([None] * n, dtype=object)
    plt = _style()
    import matplotlib.gridspec as gridspec
    fig_ = plt.figure(figsize=(17, 10.2))
    gs = gridspec.GridSpec(3, 3, figure=fig_, height_ratios=[1.0, 1.0, 1.0], hspace=0.5, wspace=0.28,
                           left=0.05, right=0.99, top=0.92, bottom=0.07)
    # --- row 1: slow points in PC space at sampled checkpoints
    sub = gridspec.GridSpecFromSubplotSpec(1, 8, subplot_spec=gs[0, :], wspace=0.15)
    if act is not None and mech is not None:
        it_act = act["iter"].astype(int); it_m = mech["iter"].astype(int)
        ks = np.unique(np.linspace(0, len(it_act) - 1, 8).round().astype(int))
        for j, k in enumerate(ks):
            ax = fig_.add_subplot(sub[0, j])
            mean, comp = act["pca_mean"][k], act["pca_comp"][k]
            traj = act["traj_pc"][k]                              # (ctx, choice, T, 3)
            t0, t1 = int(act["stim_t0"][k]), int(act["dec_t0"][k])
            for c, col in ((0, "k"), (1, "tab:blue")):
                for ch in (0, 1):
                    tr = traj[c, ch, t0:t1]
                    if np.isfinite(tr).all():
                        ax.plot(tr[:, 0], tr[:, 1], color=col, lw=0.8, alpha=0.5, ls="-" if ch else "--")
            m = int(np.argmin(np.abs(it_m - it_act[k])))
            # the settled slow manifold (small, light) and the slow points proper (larger, dark)
            for key, col, sz, al in (("settled_loc", "k", 3, 0.25), ("settled_frq", "tab:blue", 3, 0.25),
                                     ("fp_loc", "k", 10, 1.0), ("fp_frq", "tab:blue", 10, 1.0)):
                if key not in mech: continue
                F = mech[key][m]
                if F is not None and np.size(F):
                    F = np.atleast_2d(F)
                    Pj = (F - mean) @ comp[:2].T
                    ax.scatter(Pj[:, 0], Pj[:, 1], s=sz, color=col, alpha=al, zorder=3, lw=0)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("%dk" % round(it_act[k] * N_GRADIENT / 1000), fontsize=8)
            if j == 0:
                ax.set_ylabel("PC2", fontsize=8); ax.set_xlabel("PC1", fontsize=8)
        fig_.text(0.05, 0.955, "Actor: slow manifold (light dots: visited states settled under the context input) and slow points (dark) in the LOC (black) and FRQ (blue) context on the actor's PC1-PC2, with the mean trajectories by context and eventual choice (dashed L, solid R)", fontsize=9)
    # --- row 2: eigenvalue, angle, triangle
    if mech is not None:
        x = mech["iter"] * N_GRADIENT
        ax = fig_.add_subplot(gs[1, 0]); _shade(ax, is_curr)
        ax.plot(x, mech["eig1_leaky"], color="k", lw=1.6, label="LOC context"); ax.plot(x, mech["eig2_leaky"], color="tab:blue", lw=1.6, label="FRQ context")
        ax.axhline(1, color="k", ls="--", lw=0.8); ax.set_ylabel("largest eigenvalue at the slow point\n(leaky dynamics)"); ax.legend(fontsize=8); ax.set_xlabel("training trials")
        ax = fig_.add_subplot(gs[1, 1]); _shade(ax, is_curr)
        ax.plot(x, mech["sel_angle"], color="tab:green", lw=1.6); ax.set_ylim(0, 180); ax.set_ylabel("angle between selection vectors\n(LOC vs FRQ, degrees)"); ax.set_xlabel("training trials")
        ax = fig_.add_subplot(gs[1, 2]); _shade(ax, is_curr)
        tri = np.array([t if (t is not None and np.size(t) == 3) else [np.nan] * 3 for t in mech["tri1"]], float)
        ok = np.isfinite(tri).all(1)
        if ok.any():
            ax.stackplot(x[ok], tri[ok].T, labels=["selection vector", "direct input", "indirect input"], colors=["tab:purple", "tab:orange", "tab:green"], alpha=0.85)
        ax.set_ylim(0, 1); ax.set_ylabel("share of the mechanism (feature 1)"); ax.legend(fontsize=7, loc="upper right"); ax.set_xlabel("training trials")
    # --- row 3: critic
    if act is not None:
        x = act["iter"] * N_GRADIENT
        ax = fig_.add_subplot(gs[2, 0]); _shade(ax, is_curr)
        ax.plot(x, act["v_start"], color="tab:brown", lw=1.6, label="value at trial start")
        ax.plot(x, act["v_end"], color="tab:brown", lw=1.6, ls="--", label="value at the decision")
        ax.plot(x, act["p_hit"], color="k", lw=1.2, label="P(hit)")
        ax.plot(x, act["pe_terminal"], color="tab:red", lw=1.2, label="terminal prediction error")
        ax.axhline(0, color="k", ls=":", lw=0.7); ax.set_ylabel("critic"); ax.legend(fontsize=7); ax.set_xlabel("training trials")
        ax = fig_.add_subplot(gs[2, 1])
        k = len(x) - 1
        vh, vm = act["v_mean_t_hit"][k], act["v_mean_t_miss"][k]
        ax.plot(vh, color="tab:green", lw=1.6, label="hit trials"); ax.plot(vm, color="tab:red", lw=1.6, label="miss trials")
        ax.axvline(int(act["stim_t0"][k]), color="#999", ls=":", lw=0.8); ax.axvline(int(act["dec_t0"][k]), color="#999", ls=":", lw=0.8)
        ax.set_xlabel("time step within the trial (last checkpoint)"); ax.set_ylabel("critic value"); ax.legend(fontsize=8)
        ax = fig_.add_subplot(gs[2, 2]); _shade(ax, is_curr)
        ax.plot(x, act["critic_reads_evidence"], color="tab:brown", lw=1.6, label="|corr(value, actor evidence axis)|")
        ax2 = ax.twinx(); ax2.plot(x, act["actor_pr"], color="k", lw=1.2, ls="--", label="actor dimensionality (PR)"); ax2.plot(x, act["critic_pr"], color="tab:brown", lw=1.2, ls="--", label="critic dimensionality (PR)")
        ax2.set_ylabel("participation ratio"); ax2.spines["right"].set_visible(True)
        ax.set_ylim(0, 1.05); ax.set_ylabel("critic reads the evidence"); ax.set_xlabel("training trials")
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    fig_.suptitle("%s: mechanism (actor) and critic across training" % display_name(run), fontweight="bold", fontsize=13, x=0.05, y=0.985, ha="left")
    fig_.savefig(out, dpi=105); plt.close(fig_)
    return True


def ck_subset(r, dense):
    n = len(r["iter"])
    return list(range(n)) if dense else sorted(set(list(range(0, n, max(1, n // 12))) + [n - 1]))


def work(args):
    """One network: compute, draw everything. Runs in a worker process."""
    run, site, dense = args
    try:
        r = compute(run, site)
        img = os.path.join(site, "img", run); os.makedirs(img, exist_ok=True)
        is_curr = is_pretrained(run)
        fig_poster(r, os.path.join(img, "poster.png"), is_curr)
        fig_mechanism(run, os.path.join(img, "mechanism.png"), is_curr)
        return run, None
    except Exception as e:
        import traceback
        return run, traceback.format_exc()


# -------------------------------------------------------------------------------------- pages

ARM_STYLE = {   # colour and line style per arm for the population figure
    "pretrained": ("tab:red", "-"), "no_pretrain": ("k", "-"),
    "antibias pretrained (beta 5)": ("tab:red", "--"), "antibias no_pretrain (beta 5)": ("k", "--"),
    "inverted antibias pretrained (beta -5)": ("tab:red", ":"), "inverted antibias no_pretrain (beta -5)": ("k", ":"),
    "effort 0.40 pretrained": ("tab:orange", "-"), "effort 0.40 no_pretrain": ("tab:gray", "-"),
    "effort 0.55 pretrained": ("tab:orange", "--"), "effort 0.55 no_pretrain": ("tab:gray", "--"),
    "effort 0.70 pretrained": ("tab:orange", ":"), "effort 0.70 no_pretrain": ("tab:gray", ":"),
    "weak start, pretrained": ("tab:red", "-"), "weak start, no_pretrain": ("k", "-"),
    "weak start, antibias pretrained (beta 5)": ("tab:red", "--"), "weak start, antibias no_pretrain (beta 5)": ("k", "--"),
    "weak start, inverted antibias pretrained (beta -5)": ("tab:red", ":"), "weak start, inverted antibias no_pretrain (beta -5)": ("k", ":"),
    "weak start, effort 0.55 pretrained": ("tab:orange", "--"), "weak start, effort 0.55 no_pretrain": ("tab:gray", "--"),
    "weak start, fixation-cost effort, no_pretrain": ("tab:gray", "-"), "weak start, fixation-cost effort, pretrained": ("tab:orange", "-"),
    "weak start, fixation-cost effort, stages 1-6 then full task": ("tab:green", "-"), "weak start, fixation-cost effort, stages 1-10": ("darkgreen", "--"),
    "weak start, effort 0.70 no_pretrain": ("tab:gray", "--"), "weak start, effort 0.70 pretrained": ("tab:orange", "--"),
    "weak start, effort 0.85 no_pretrain": ("tab:gray", ":"), "weak start, effort 0.85 pretrained": ("tab:orange", ":"),
    "weak start, stages 1-6 then full task (effort 0.55)": ("tab:green", "-"), "weak start, stages 1-10 with anti-bias (effort 0.55)": ("darkgreen", "--"),
    "stages 1-6 then full task (effort 0.55)": ("tab:green", "-"), "stages 1-10 with quadrant anti-bias (effort 0.55)": ("darkgreen", "--"),
    "library (no_pretrain, 36k trials)": ("tab:blue", "-")}

STAGE_LOGS = os.path.expanduser("~/pyrl_work/logs/stages")   # training logs of the stage probe (copied from 0095 by collect_and_build.sh)
RAT_STAGES = "/mnt/big/rat_behavior/derived/stage_statistics.csv"   # pooled trials per stage over 208 rats (read-only)


def parse_stage_log(run):
    """The stage a stage-probe network was in at every checkpoint, read from its training log.
    pagan_hist_stages.update prints "[ pagan_hist_stages ] arm X stage S: ..." just before pyrl
    prints "After N updates", so the last stage line seen is the stage at that checkpoint.
    Returns (updates, stages) arrays, or None if the log is not there."""
    suf = run.replace("pagan_hist_stages_", "")
    path = os.path.join(STAGE_LOGS, suf + ".log")
    if not os.path.exists(path):
        return None
    ups, sts, last = [], [], None
    with open(path) as f:
        for line in f:
            m = re.match(r"\[ pagan_hist_stages \] arm \w+ stage (\d+):", line)
            if m:
                last = int(m.group(1)); continue
            m = re.match(r"After (\d+) updates", line)
            if m and last is not None:
                ups.append(int(m.group(1))); sts.append(last)
    return (np.array(ups), np.array(sts)) if ups else None


def rat_stage_trials():
    """Trials per stage of the pooled rats: n_trials / n_rats per stage, as a staircase reference
    (an average per rat that was in the stage, not a median over rats)."""
    try:
        import csv
        rows = list(csv.DictReader(open(RAT_STAGES)))
        return {int(r["stage"]): float(r["n_trials"]) / max(1.0, float(r["n_rats"])) for r in rows if r.get("n_trials")}
    except Exception:
        return {}


def fig_stage_ladder(runs, out):
    """Stage against training trials for every stage-probe network (one line per seed, one panel
    per arm), with the rats' average staircase in grey. Runs are run names; the stage comes from
    the training logs, so this figure exists before the records do."""
    plt = _style()
    arms = {}
    for run in runs:
        arms.setdefault(arm_of(run), []).append(run)
    if not arms:
        return False
    fig_, axes = plt.subplots(1, len(arms), figsize=(6.5 * len(arms), 3.6), sharey=True, squeeze=False)
    rat = rat_stage_trials()
    pre = sum(b - a for _, a, b, _ in PHASES)             # nose-poke trials before stage 1
    any_data = False
    for ax, (a, rs) in zip(axes[0], sorted(arms.items())):
        col, ls = ARM_STYLE.get(a, ("tab:green", "-"))
        for run in sorted(rs):
            d = parse_stage_log(run)
            if d is None: continue
            any_data = True
            ax.step(d[0] * N_GRADIENT, d[1], where="post", color=col, ls=ls, lw=1.2, alpha=0.8)
            ax.text(d[0][-1] * N_GRADIENT, d[1][-1], re.search(r"_w(\d+)", run).group(1).lstrip("0"), fontsize=7, color=col, va="bottom")
        if rat:
            order = [1, 2, 3, 4, 5, 6, 10] if "1-6" in a else list(range(1, 11))
            # staircase: the value after each step is the NEXT stage (where="post"), the last stage holds
            x, y, t = [pre], [order[0]], pre
            for i, st in enumerate(order[:-1]):
                t += rat.get(st, 0); x.append(t); y.append(order[i + 1])
            ax.step(x, y, where="post", color="0.5", lw=2.0, alpha=0.6, label="rats (pooled average per stage)")
        _shade(ax, True)
        ax.set_title(a, fontsize=10); ax.set_xlabel("training trials"); ax.set_yticks(range(1, 11))
        ax.set_xlim(0, 2800 * N_GRADIENT); ax.set_ylim(0.5, 10.5); ax.legend(fontsize=7, loc="lower right")
    axes[0][0].set_ylabel("stage")
    fig_.suptitle("Where each network is on the rats' stage ladder (promotion: 150 attempted trials and the stage's accuracy threshold)", fontsize=10, fontweight="bold")
    fig_.tight_layout(); fig_.savefig(out, dpi=110); plt.close(fig_)
    return any_data


def _median_on_grid(runs, key, grid, transform=None):
    M = np.full((len(runs), len(grid)), np.nan)
    for i, r in enumerate(runs):
        x = r["iter"] * N_GRADIENT
        y = np.asarray(r[key], float) if key in r else np.full(len(x), np.nan)
        if transform: y = transform(r, y)
        ok = np.isfinite(y)
        if ok.sum() < 2: continue
        # hold the last value after a run stops (criterion reached): the network stays learned
        M[i] = np.interp(grid, x[ok], y[ok], right=y[ok][-1])
        M[i, grid < x[ok][0]] = np.nan
    with np.errstate(all="ignore"):
        return np.nanmedian(M, 0), np.isfinite(M).sum(0)


def fig_population_by_arm(by_run, out, exclude=("library (no_pretrain, 36k trials)",), only=None):
    exclude = tuple(exclude) + tuple(a for _, _, a, _ in ARMS if "v1" in a)
    """Median over seeds, per arm, of accuracy, attempt rate, |bias|, WSLS and relevant weight
    against training trials; all arms on one axis so the three experiments can be compared."""
    plt = _style()
    arms = {}
    for run, r in by_run.items():
        a = arm_of(run)
        if a in exclude or (only is not None and a not in only): continue
        arms.setdefault(a, []).append(r)
    grid = np.arange(0, 2800 * N_GRADIENT + 1, 720)
    names = None
    rows = [("task_acc", "P(correct) on the task", None, (0.4, 1.0)),
            ("p_choice", "trials attempted (fraction)", None, (0, 1.02)),
            ("bias", "|bias| weight", lambda r, y: np.where(r["p_major"] > P_MAJOR_MAX, np.nan, np.abs(y)), None),
            ("wsls", "win-stay/lose-switch weight", lambda r, y: np.where(r["p_major"] > P_MAJOR_MAX, np.nan, y), None),
            ("rel", "relevant-evidence weight", lambda r, y: np.where(r["p_major"] > P_MAJOR_MAX, np.nan, y), None)]
    fig_, axes = plt.subplots(len(rows), 1, figsize=(12, 2.4 * len(rows)), sharex=True)
    for ax, (key, lab, tf, ylim) in zip(axes, rows):
        for a, runs in sorted(arms.items()):
            col, ls = ARM_STYLE.get(a, ("tab:purple", "-"))
            if key in ("bias", "wsls", "rel"):
                nm = [str(n) for n in runs[0]["names"]]
                i = nm.index(key)
                series = [dict(r, **{"_w": r["W"][i]}) for r in runs]
                med, n = _median_on_grid(series, "_w", grid, tf)
            else:
                med, n = _median_on_grid(runs, key, grid, tf)
            ax.plot(grid, med, color=col, ls=ls, lw=1.8, label="%s (n=%d)" % (a, len(runs)))
        ax.axhline(0.5 if key == "task_acc" else 0, color="k", ls=":", lw=0.7)
        if ylim: ax.set_ylim(*ylim)
        ax.set_ylabel(lab, fontsize=9)
    axes[0].legend(fontsize=7, ncol=3, loc="lower right", framealpha=0.9)
    axes[-1].set_xlabel("training trials")
    fig_.suptitle("Population by arm: median over seeds (runs that reached criterion hold their last value)", fontsize=11, fontweight="bold")
    fig_.tight_layout(); fig_.savefig(out, dpi=110); plt.close(fig_)


def fig_mechanism_population(by_run, out, arms=("pretrained", "no_pretrain", "antibias pretrained (beta 5)", "antibias no_pretrain (beta 5)", "inverted antibias pretrained (beta -5)")):
    """Medians per arm of the mechanism and critic measures against training trials."""
    plt = _style()
    grid = np.arange(0, 2800 * N_GRADIENT + 1, 1440)
    series = {}
    for run in by_run:
        a = arm_of(run)
        if a not in arms: continue
        m, act = load_mech(run), load_act(run)
        if m is not None:
            x = m["iter"] * N_GRADIENT
            tri = np.array([t if (t is not None and np.size(t) == 3) else [np.nan] * 3 for t in m["tri1"]], float)
            for key, y in (("eig_loc", m["eig1_leaky"]), ("eig_frq", m["eig2_leaky"]), ("sel_angle", m["sel_angle"]),
                           ("tri_sel", tri[:, 0]), ("tri_direct", tri[:, 1]), ("tri_indirect", tri[:, 2])):
                series.setdefault((a, key), []).append((x, np.asarray(y, float)))
        if act is not None:
            x = act["iter"] * N_GRADIENT
            for key in ("v_start", "pe_terminal", "critic_reads_evidence", "actor_pr", "critic_pr", "angle_evidence_ctx"):
                series.setdefault((a, key), []).append((x, np.asarray(act[key], float)))
    rows = [("eig_loc", "leading eigenvalue, LOC (leaky)", (0.9, 1.1)), ("eig_frq", "leading eigenvalue, FRQ (leaky)", (0.9, 1.1)),
            ("sel_angle", "selection-vector angle LOC vs FRQ (deg)", (0, 180)), ("tri_sel", "triangle: selection-vector share", (0, 1)),
            ("tri_indirect", "triangle: indirect-input share", (0, 1)), ("angle_evidence_ctx", "evidence-axis angle LOC vs FRQ (deg)", (0, 180)),
            ("v_start", "critic value at trial start", (-0.2, 1.05)), ("pe_terminal", "terminal prediction error", (-1, 1)),
            ("critic_reads_evidence", "|corr(value, evidence axis)|", (0, 1)), ("actor_pr", "actor dimensionality (PR)", None), ("critic_pr", "critic dimensionality (PR)", None)]
    fig_, axes = plt.subplots(len(rows), 1, figsize=(11, 1.9 * len(rows)), sharex=True)
    for ax, (key, lab, ylim) in zip(axes, rows):
        for a in arms:
            runs = series.get((a, key), [])
            if not runs: continue
            M = np.full((len(runs), len(grid)), np.nan)
            for i, (x, y) in enumerate(runs):
                ok = np.isfinite(y)
                if ok.sum() >= 2:
                    M[i] = np.interp(grid, x[ok], y[ok], right=np.nan); M[i, grid < x[ok][0]] = np.nan
            col, ls = ARM_STYLE.get(a, ("tab:purple", "-"))
            with np.errstate(all="ignore"):
                ax.plot(grid, np.nanmedian(M, 0), color=col, ls=ls, lw=1.8, label="%s (n=%d)" % (a, len(runs)))
        if key.startswith("eig"): ax.axhline(1, color="k", ls="--", lw=0.7)
        if ylim: ax.set_ylim(*ylim)
        ax.set_ylabel(lab, fontsize=8)
    axes[0].legend(fontsize=7, ncol=2, loc="lower right", framealpha=0.9); axes[-1].set_xlabel("training trials")
    fig_.suptitle("Mechanism and critic by arm: median over seeds", fontsize=11, fontweight="bold")
    fig_.tight_layout(); fig_.savefig(out, dpi=110); plt.close(fig_)


def fig_vanilla_vs_rl(by_run, out):
    """One method, two kinds of network: medians of the slow-point measures against the fraction
    of training, for the vanilla supervised networks (N20, N100; sup_*_vanilla_mechanism.npz)
    and the reward-trained arms. The x axis is normalised because the two are trained on
    different units (gradient steps versus trials)."""
    plt = _style()
    groups = {}
    for f in glob.glob(os.path.join(MECH, "sup_N*_vanilla_mechanism.npz")):
        g = "vanilla N20 (supervised)" if "_N20_" in f else "vanilla N100 (supervised)"
        groups.setdefault(g, []).append(dict(np.load(f, allow_pickle=True)))
    for run in by_run:
        a = arm_of(run)
        if a in ("pretrained", "no_pretrain", "antibias pretrained (beta 5)", "antibias no_pretrain (beta 5)"):
            m = load_mech(run)
            if m is not None: groups.setdefault("RL " + a, []).append(m)
    style = {"vanilla N20 (supervised)": ("tab:cyan", "-"), "vanilla N100 (supervised)": ("tab:blue", "-"),
             "RL pretrained": ("tab:red", "-"), "RL no_pretrain": ("k", "-"),
             "RL antibias pretrained (beta 5)": ("tab:red", "--"), "RL antibias no_pretrain (beta 5)": ("k", "--")}
    grid = np.linspace(0, 1, 31)
    rows = [("eig1_leaky", "leading eigenvalue, LOC context", (0.85, 1.1)), ("eig2_leaky", "leading eigenvalue, FRQ context", (0.85, 1.1)),
            ("sel_angle", "selection-vector angle LOC vs FRQ (deg)", (0, 180)),
            ("tri0", "triangle: selection-vector share", (0, 1)), ("tri1", "triangle: direct-input share", (0, 1)), ("tri2", "triangle: indirect-input share", (0, 1)),
            ("n_fp_loc", "number of slow points, LOC context", None)]
    fig_, axes = plt.subplots(len(rows), 1, figsize=(10, 2.0 * len(rows)), sharex=True)
    for ax, (key, lab, ylim) in zip(axes, rows):
        for g, ms in groups.items():
            M = np.full((len(ms), len(grid)), np.nan)
            for i, m in enumerate(ms):
                it = np.asarray(m["iter"], float); x = (it - it.min()) / max(1e-9, it.max() - it.min())
                if key.startswith("tri"):
                    tri = np.array([t if (t is not None and np.size(t) == 3) else [np.nan] * 3 for t in m["tri1"]], float)
                    y = tri[:, int(key[3])]
                else:
                    y = np.asarray(m[key], float)
                ok = np.isfinite(y)
                if ok.sum() >= 2: M[i] = np.interp(grid, x[ok], y[ok])
            col, ls = style.get(g, ("tab:purple", "-"))
            with np.errstate(all="ignore"):
                ax.plot(grid, np.nanmedian(M, 0), color=col, ls=ls, lw=1.8, label="%s (n=%d)" % (g, len(ms)))
        if key.startswith("eig"): ax.axhline(1, color="k", ls="--", lw=0.7)
        if ylim: ax.set_ylim(*ylim)
        ax.set_ylabel(lab, fontsize=8)
    axes[0].legend(fontsize=7, ncol=2, loc="lower right", framealpha=0.9); axes[-1].set_xlabel("fraction of training")
    fig_.suptitle("Vanilla supervised versus reward-trained networks under one slow-point method (medians)", fontsize=11, fontweight="bold")
    fig_.tight_layout(); fig_.savefig(out, dpi=110); plt.close(fig_)


def fig_vanilla2025(by_run, out):
    """The 2025-code networks: loss and slow-point measures against the checkpoint (120 per seed),
    one light line per seed and the median; the final triangle of every seed; and the RL arms'
    medians against the fraction of training for comparison. Returns the number of seeds."""
    files = sorted(glob.glob(os.path.join(MECH, "vanilla2025_s*_mechanism.npz")))
    if not files:
        return 0
    plt = _style()
    ms = [dict(np.load(f, allow_pickle=True)) for f in files]
    grid = np.linspace(0, 1, 25)
    def series(m, key):
        it = np.asarray(m["iter"], float); x = it / max(1, it.max())
        if key.startswith("tri"):
            tri = np.array([t if (t is not None and np.size(t) == 3) else [np.nan] * 3 for t in m["tri1"]], float); y = tri[:, int(key[3])]
        else:
            y = np.asarray(m[key], float)
        ok = np.isfinite(y)
        return (np.interp(grid, x[ok], y[ok]) if ok.sum() >= 2 else np.full(len(grid), np.nan))
    rows = [("loss", "training loss", None), ("eig1", "leading eigenvalue, context 1", (0.5, 1.1)), ("n_fp_loc", "number of slow points, context 1", None),
            ("sel_angle", "selection-vector angle (deg)", (0, 180)), ("tri0", "triangle: selection vector", (0, 1)), ("tri1", "triangle: direct input", (0, 1)), ("tri2", "triangle: indirect input", (0, 1))]
    # RL arms for comparison (medians over seeds, fraction of training)
    rl = {}
    for run in by_run:
        a = arm_of(run)
        if a in ("pretrained", "no_pretrain"):
            m = load_mech(run)
            if m is not None: rl.setdefault(a, []).append(m)
    fig_, axes = plt.subplots(len(rows), 1, figsize=(10, 2.0 * len(rows)), sharex=True)
    for ax, (key, lab, ylim) in zip(axes, rows):
        M = np.array([series(m, key) for m in ms])
        sub = M[np.random.default_rng(0).choice(len(M), size=min(60, len(M)), replace=False)]
        for y in sub: ax.plot(grid, y, color="tab:cyan", lw=0.5, alpha=0.25)
        with np.errstate(all="ignore"):
            ax.plot(grid, np.nanmedian(M, 0), color="tab:blue", lw=2.2, label="vanilla, 2025 code (n=%d, median)" % len(M))
        if key != "loss":
            for a, col in (("no_pretrain", "k"), ("pretrained", "tab:red")):
                if a in rl:
                    R = []
                    for m in rl[a]:
                        it = np.asarray(m["iter"], float); x = (it - it.min()) / max(1e-9, it.max() - it.min())
                        if key.startswith("tri"):
                            tri = np.array([t if (t is not None and np.size(t) == 3) else [np.nan] * 3 for t in m["tri1"]], float); y = tri[:, int(key[3])]
                        else:
                            kk = {"eig1": "eig1_leaky"}.get(key, key); y = np.asarray(m[kk], float)
                        ok = np.isfinite(y)
                        if ok.sum() >= 2: R.append(np.interp(grid, x[ok], y[ok]))
                    if R:
                        with np.errstate(all="ignore"):
                            ax.plot(grid, np.nanmedian(np.array(R), 0), color=col, lw=1.6, ls="--", label="RL %s (median)" % a)
        if key.startswith("eig"): ax.axhline(1, color="k", ls="--", lw=0.7)
        if ylim: ax.set_ylim(*ylim)
        if key == "loss": ax.set_yscale("log")
        ax.set_ylabel(lab, fontsize=8)
    axes[1].legend(fontsize=7, loc="lower right", framealpha=0.9); axes[-1].set_xlabel("fraction of training")
    fig_.suptitle("Vanilla RNNs, Nature 2025 code: learning and mechanism across training, with the RL base arms for comparison", fontsize=10, fontweight="bold")
    fig_.tight_layout(); fig_.savefig(out, dpi=110); plt.close(fig_)
    # final triangle of every seed
    tri = []
    for m in ms:
        t = [x for x in m["tri1"] if (x is not None and np.size(x) == 3)]
        if t: tri.append(np.asarray(t[-1], float))
    tri = np.array(tri)
    fig_, ax = plt.subplots(figsize=(5.5, 5))
    if len(tri):
        A, B, C = np.array([0, 0]), np.array([1, 0]), np.array([0.5, np.sqrt(3) / 2])
        P = tri[:, 0:1] * A + tri[:, 1:2] * B + tri[:, 2:3] * C
        ax.scatter(P[:, 0], P[:, 1], s=8, alpha=0.5, color="tab:blue", label="2025 code, final (n=%d)" % len(tri))
        for a, col in (("no_pretrain", "k"), ("pretrained", "tab:red")):
            Q = []
            for m in rl.get(a, []):
                t = [x for x in m["tri1"] if (x is not None and np.size(x) == 3)]
                if t: Q.append(np.asarray(t[-1], float))
            if Q:
                Q = np.array(Q); PQ = Q[:, 0:1] * A + Q[:, 1:2] * B + Q[:, 2:3] * C
                ax.scatter(PQ[:, 0], PQ[:, 1], s=28, color=col, marker="x", label="RL %s, final (n=%d)" % (a, len(Q)))
        ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3) / 2, 0], color="k", lw=1)
        ax.text(0, -0.05, "selection vector", ha="center", fontsize=9); ax.text(1, -0.05, "direct input", ha="center", fontsize=9); ax.text(0.5, np.sqrt(3) / 2 + 0.03, "indirect input", ha="center", fontsize=9)
    ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.12, 1.0); ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=8, loc="upper right")
    for sp in ax.spines.values(): sp.set_visible(False)
    fig_.tight_layout(); fig_.savefig(out.replace(".png", "_triangle.png"), dpi=120); plt.close(fig_)
    return len(ms)


def summary_row(r):
    names = [str(n) for n in r["names"]]
    acc = r["task_acc"] if np.isfinite(r["task_acc"]).any() else r["val_acc"]
    hit = np.where(acc >= 0.75)[0]
    good = ~(r["p_major"] > P_MAJOR_MAX)
    last = np.where(good)[0][-1] if good.any() else len(acc) - 1
    run = str(r["run"])
    arm = arm_of(run)
    ci = int(np.asarray(r.get("criterion_iter", [-1])).ravel()[0])
    row = {"run": run, "name": display_name(run), "arm": arm, "cls": CLASS_OF_ARM.get(arm, "other"), "seed": int(re.search(r"_w(\d+)", run).group(1)),
           "to_095": int(ci * N_GRADIENT) if ci > 0 else None,
           "final_attempt": float(np.nanmean(r["attempt_rate"][-3:])) if np.isfinite(r["attempt_rate"]).any() else None,
           "checkpoints": len(acc), "trials": int(r["iter"][-1] * N_GRADIENT), "final_acc": float(acc[-1]),
           "to_075": int(r["iter"][hit[0]] * N_GRADIENT) if len(hit) else None,
           "n_deg": int((~good).sum()), "psytrack": "yes" if "psytrack" in r else "",
           "acc_spark": sparkline(acc, baseline=0.5, lo=0.4, hi=1.0),
           "wsls_spark": sparkline(np.where(good, r["W"][names.index("wsls")], np.nan), color="tab:purple", baseline=0.0),
           "rel_spark": sparkline(np.where(good, r["W"][names.index("rel")], np.nan), color="tab:red", baseline=0.0)}
    for nm in ("rel", "irr", "wsls", "histChoice", "bias"):
        row["w_" + nm] = float(r["W"][names.index(nm), last])
    return row


FIT_CSS = '<style>main{max-width:none;padding:8px 16px} figure{margin:4px 0} figure img{max-height:calc(100vh - 120px);width:auto;max-width:100%} dl.kv{grid-template-columns:repeat(8,max-content 1fr);font-size:12.5px} h1{font-size:20px;margin:2px 0}</style>'


def network_page(site, r, dense, neighbours=None):
    """One network, one screen: a one-line header and the poster figure, sized to the viewport.
    neighbours = (prev_run, next_run) within the same group, wrapped around, for the arrow keys."""
    run = str(r["run"]); short = display_name(run)
    img = "../img/%s/" % run
    root = os.path.join(site.root, "networks")
    row = summary_row(r)
    seed = row["seed"]
    pair = ""
    for name, pa, pb, la, lb in PAIRINGS:
        if run.startswith(pa) or run.startswith(pb):
            pair += ' &middot; <a href="../pairs/%s_seed_%03d.html">%s</a>' % (name, seed, esc(name + " pair"))
    body = kv({"arm": row["arm"], "seed": seed, "training trials": row["trials"], "final P(correct)": row["final_acc"],
               "first >= 0.75 at": row["to_075"], "non-identifiable ckpts": row["n_deg"], "PsyTrack": row["psytrack"] or "no",
               "run": Raw("<code>%s</code>%s" % (esc(run), pair))})
    if neighbours:
        pv, nx = neighbours
        ctitle = dict((k, t) for k, t, _ in CLASSES).get(CLASS_OF_ARM.get(row["arm"], ""), row["arm"])
        body = keynav("%s.html" % pv, "%s.html" % nx, display_name(pv), display_name(nx), ctitle) + body
    body = '<p class="note">Class: <a href="../classes/%s.html">%s</a></p>' % (CLASS_OF_ARM.get(row["arm"], "other"), esc(dict((k, t) for k, t, _ in CLASSES).get(CLASS_OF_ARM.get(row["arm"], ""), row["arm"]))) + body
    body += fig(img + "poster.png", "", root=root)
    if os.path.exists(os.path.join(site.root, "img", run, "mechanism.png")):
        body += '<h2 id="mechanism">Mechanism and critic</h2>' + fig(img + "mechanism.png", "", root=root)
    else:
        body += note("Mechanism and critic: not computed yet for this network.")
    if os.path.exists(os.path.join(site.root, "movies", run + ".mp4")):
        body += '<h2 id="slowpoints">Slow points over training</h2>' + note("Slow points under each context cue at 40 points in training, in a fixed view (blue: location context, red: frequency context); then coloured by speed, by the network's choice at that state, and by how strongly it holds fixation.") + '<video controls loop muted playsinline style="max-width:100%%" src="../movies/%s.mp4"></video>' % run
    site.page("networks/%s.html" % run, short, body, extra_head=FIT_CSS)


# Comparisons drawn as pairs with the same seed: (name, prefix of arm A, prefix of arm B, label A, label B)
PAIRINGS = [("pretrain", "pagan_hist_curriculum_curr_", "pagan_hist_curriculum_ctrl_", "pretrained", "no_pretrain"),
            ("antibias_pretrained", "pagan_hist_antibias_abcurr_", "pagan_hist_curriculum_curr_", "antibias pretrained", "pretrained"),
            ("antibias_no_pretrain", "pagan_hist_antibias_abctrl_", "pagan_hist_curriculum_ctrl_", "antibias no_pretrain", "no_pretrain"),
            ("antibias_inverted", "pagan_hist_antibias_abneg_", "pagan_hist_curriculum_curr_", "inverted antibias pretrained", "pretrained"),
            ("antibias_inverted_no_pretrain", "pagan_hist_antibias_abnegctrl_", "pagan_hist_curriculum_ctrl_", "inverted antibias no_pretrain", "no_pretrain"),
            ("effort0.40", "pagan_hist_effort2_e240curr_", "pagan_hist_effort2_e240ctrl_", "effort 0.40 pretrained", "effort 0.40 no_pretrain"),
            ("effort0.55", "pagan_hist_effort2_e255curr_", "pagan_hist_effort2_e255ctrl_", "effort 0.55 pretrained", "effort 0.55 no_pretrain"),
            ("effort0.70", "pagan_hist_effort2_e270curr_", "pagan_hist_effort2_e270ctrl_", "effort 0.70 pretrained", "effort 0.70 no_pretrain"),
            ("weak_pretrain", "pagan_hist_curriculum_g06curr_", "pagan_hist_curriculum_g06ctrl_", "weak pretrained", "weak no_pretrain"),
            ("weak_antibias_pretrained", "pagan_hist_antibias_g06abcurr_", "pagan_hist_curriculum_g06curr_", "weak antibias pretrained", "weak pretrained"),
            ("weak_antibias_no_pretrain", "pagan_hist_antibias_g06abctrl_", "pagan_hist_curriculum_g06ctrl_", "weak antibias no_pretrain", "weak no_pretrain"),
            ("weak_fixcost", "pagan_hist_fixcost_g06fc55curr_", "pagan_hist_fixcost_g06fc55ctrl_", "fixation-cost effort, pretrained", "fixation-cost effort, no_pretrain"),
            ("weak_fixcost_vs_base", "pagan_hist_fixcost_g06fc55curr_", "pagan_hist_curriculum_g06curr_", "fixation-cost effort, pretrained", "weak pretrained"),
            ("weak_fixcost_stages1-10_vs_none", "pagan_hist_stages3_g06fc55full_", "pagan_hist_fixcost_g06fc55curr_", "fixation-cost stages 1-10", "fixation-cost, no stages"),
            ("weak_fixcost_stages1-6_vs_none", "pagan_hist_stages3_g06fc55short_", "pagan_hist_fixcost_g06fc55curr_", "fixation-cost stages 1-6", "fixation-cost, no stages"),
            ("weak_effort0.55", "pagan_hist_effort2_g06e255curr_", "pagan_hist_effort2_g06e255ctrl_", "weak effort 0.55 pretrained", "weak effort 0.55 no_pretrain"),
            ("weak_effort_vs_base", "pagan_hist_effort2_g06e255curr_", "pagan_hist_curriculum_g06curr_", "weak effort 0.55 pretrained", "weak pretrained"),
            ("weak_stages1-10_vs_none", "pagan_hist_stages_g06st55full_", "pagan_hist_effort2_g06e255curr_", "weak stages 1-10", "weak no stages"),
            ("weak_stages1-6_vs_none", "pagan_hist_stages_g06st55short_", "pagan_hist_effort2_g06e255curr_", "weak stages 1-6", "weak no stages"),
            ("stages1-6_vs_none", "pagan_hist_stages_st55short_", "pagan_hist_effort2_e255curr_", "stages 1-6 then full task", "no stages (effort 0.55 pretrained)"),
            ("stages1-10_vs_none", "pagan_hist_stages_st55full_", "pagan_hist_effort2_e255curr_", "stages 1-10 with quadrant anti-bias", "no stages (effort 0.55 pretrained)"),
            ("stages1-10_vs_stages1-6", "pagan_hist_stages_st55full_", "pagan_hist_stages_st55short_", "stages 1-10 with quadrant anti-bias", "stages 1-6 then full task"),
            ("effort0.55_vs_base_pretrained", "pagan_hist_effort2_e255curr_", "pagan_hist_curriculum_curr_", "effort 0.55 pretrained", "pretrained"),
            ("effort0.55_vs_base_no_pretrain", "pagan_hist_effort2_e255ctrl_", "pagan_hist_curriculum_ctrl_", "effort 0.55 no_pretrain", "no_pretrain")]


def pair_pages(site, by_run):
    """One page per seed and pairing, both arms overlaid (fig_pair); solid = arm A, dashed = arm B."""
    all_links = {}
    for name, pa, pb, la, lb in PAIRINGS:
        seeds = sorted({int(re.search(r"_w(\d+)", r).group(1)) for r in by_run if r.startswith(pa)})
        seeds = [sd for sd in seeds if (pb + "w%03d" % sd) in by_run]
        links = []
        for si, seed in enumerate(seeds):
            ra, rb = by_run[pa + "w%03d" % seed], by_run[pb + "w%03d" % seed]
            out = os.path.join(site.root, "img", "pairs"); os.makedirs(out, exist_ok=True)
            png = os.path.join(out, "%s_seed_%03d.png" % (name, seed))
            if not os.path.exists(png):
                fig_pair(ra, rb, png, la, lb)
            body = kv({"seed": seed, la: Raw('<a href="../networks/%s.html">%s</a>' % (ra["run"], display_name(str(ra["run"])))),
                       lb: Raw('<a href="../networks/%s.html">%s</a>' % (rb["run"], display_name(str(rb["run"])))),
                       "same initial weights": "yes (same weight seed; Win, bin, Wrec identical at iteration 0)"})
            pv, nx = seeds[si - 1], seeds[(si + 1) % len(seeds)]
            body = keynav("%s_seed_%03d.html" % (name, pv), "%s_seed_%03d.html" % (name, nx), "seed %d" % pv, "seed %d" % nx, name) + body
            body += fig("../img/pairs/%s_seed_%03d.png" % (name, seed), "", root=os.path.join(site.root, "pairs"))
            site.page("pairs/%s_seed_%03d.html" % (name, seed), "seed %d: %s vs %s" % (seed, la, lb), body, extra_head=FIT_CSS)
            links.append((seed, summary_row(ra), summary_row(rb)))
        all_links[name] = (la, lb, links)
    body = note("Each pair starts from the same weight seed (Win, bin and Wrec identical at iteration 0) and only the arm differs. Solid = first arm, dashed = second arm in the overlays.")
    for name, pa, pb, la, lb in PAIRINGS:
        la_, lb_, links = all_links.get(name, (la, lb, []))
        if not links: continue
        rows = [{"seed": sd, "pair": Raw('<a href="pairs/%s_seed_%03d.html">seed %d</a>' % (name, sd, sd)),
                 "a_acc": a_["final_acc"], "b_acc": b_["final_acc"], "a_075": a_["to_075"], "b_075": b_["to_075"],
                 "a_wsls": a_["w_wsls"], "b_wsls": b_["w_wsls"], "a_spark": a_["acc_spark"], "b_spark": b_["acc_spark"]} for sd, a_, b_ in links]
        body += '<h2 id="%s">%s vs %s</h2>' % (name, esc(la), esc(lb))
        body += table(rows, [("pair", "pair"), ("a_acc", "final acc, " + la), ("b_acc", "final acc, " + lb), ("a_075", "to 0.75, " + la), ("b_075", "to 0.75, " + lb),
                             ("a_wsls", "WSLS, " + la), ("b_wsls", "WSLS, " + lb), ("a_spark", "accuracy, " + la), ("b_spark", "accuracy, " + lb)],
                      tid="pairs_" + name, numeric=["a_acc", "b_acc", "a_075", "b_075", "a_wsls", "b_wsls"])
    site.page("pairs.html", "Pairs: the same seed under two arms", body)
    return



def population_pages(site, rows):
    cols = [("name", "network"), ("cls", "class"), ("arm", "arm"), ("seed", "seed"), ("trials", "training trials"), ("final_acc", "final P(correct)"),
            ("to_075", "trials to 0.75"), ("acc_spark", "accuracy"), ("rel_spark", "relevant weight"), ("wsls_spark", "WSLS weight"),
            ("w_rel", "rel"), ("w_irr", "irr"), ("w_wsls", "wsls"), ("w_histChoice", "prev"), ("w_bias", "bias"),
            ("n_deg", "non-ident. ckpts"), ("psytrack", "PsyTrack")]
    numeric = ["seed", "trials", "final_acc", "to_075", "w_rel", "w_irr", "w_wsls", "w_histChoice", "w_bias", "n_deg"]
    links = {"name": lambda r: "networks/%s.html" % r["run"]}
    by_run = population_pages.by_run
    # ---- home: the classes of models
    home = note(TREE["lead_note"])
    cards = []
    archive_cards = []
    for key, title, info in CLASSES:
        n = sum(1 for run in by_run if CLASS_OF_ARM.get(arm_of(run)) == key)
        arms = sorted({arm_of(run) for run in by_run if CLASS_OF_ARM.get(arm_of(run)) == key}, key=lambda a: [x[2] for x in ARMS].index(a))
        desc = re.sub(r"<[^>]+>", "", info["description"]).strip()
        short = desc[:420] + ("..." if len(desc) > 420 else "")
        if key == "vanilla2025":
            nv = len(glob.glob(os.path.join(MECH, "vanilla2025_s*_mechanism.npz")))
            meta = "%d seeds, mechanism and loss (behaviour pages to come)" % nv if nv else "analysis running"
        else:
            meta = ("%d networks, %d arms" % (n, len(arms))) if n else ("runs in progress, nothing collected yet" if key == "stages" else "mechanism only")
        (cards if key in FRONT else archive_cards).append('<h3><a href="classes/%s.html">%s</a></h3><p class="lead">%s</p><p>%s</p>' % (key, esc(title), esc(meta), esc(short)))
    home += grid(cards)
    home += note(TREE["archive_note"])
    site.page("archive.html", "Other and earlier networks", note(TREE["archive_blurb"]) + grid(archive_cards))
    home += note(TREE["home_extra"])
    home += note('<a href="networks.html">All networks in one sortable table</a> &middot; <a href="pairs.html">all same-seed pairs</a> &middot; <a href="compare.html">PsyTrack vs GLM</a> &middot; <a href="methods.html">methods</a>')
    site.page("index.html", TREE["index_title"], home, lead="%d networks in %d classes. Start from a class." % (len(rows), sum(1 for k, _, _ in CLASSES if any(CLASS_OF_ARM.get(arm_of(r)) == k for r in by_run))))
    # ---- class pages
    for key, title, info in CLASSES:
        runs_c = [run for run in by_run if CLASS_OF_ARM.get(arm_of(run)) == key]
        body = '<h2>What it is</h2>' + info["description"]
        body += '<h2>What we found</h2>' + info["findings"]
        if key in FRONT:
            lf = os.path.join(site.root, "img", "line_formation_%s.png" % key)
            if os.path.exists(lf):
                body += "<h2>Mechanism over training</h2>" + fig("../img/line_formation_%s.png" % key, "This type (colour) against type 1, no pre-training (black): accuracy, then the slow points under each context cue: how long the slow structure is, how close to a single line, the contraction along it (1 = a line attractor), how slow it is, and whether the choice is ordered along it. Median over seeds; thin lines are single networks; pink is the nose-poke phase.", root=os.path.join(site.root, "classes"))
            else:
                body += "<h2>Mechanism over training</h2>" + note("Not computed yet for this type.")
        if key == "stages":
            # the stage per checkpoint comes from the training logs, so this is shown as soon as the logs are collected
            log_runs = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(STAGE_LOGS, "st55*.log")))
            log_runs = ["pagan_hist_stages_" + r for r in log_runs]
            try:
                if fig_stage_ladder(log_runs, os.path.join(site.root, "img", "stage_ladder.png")):
                    body += "<h2>Stage ladder (from the training logs, %d runs)</h2>" % len(log_runs)
                    body += fig("../img/stage_ladder.png", "Stage against training trials, one line per seed, labelled with the seed at its end; grey: the pooled rats' average trials per stage, as a staircase. Shading: the nose-poke phases.", root=os.path.join(site.root, "classes"))
            except Exception as e:
                body += caveat("stage ladder failed: %s" % esc(e))
        if runs_c:
            arms_c = sorted({arm_of(r) for r in runs_c}, key=lambda a: [x[2] for x in ARMS].index(a))
            srows = []
            for a in arms_c:
                rs = [summary_row(by_run[r]) for r in runs_c if arm_of(r) == a]
                crit = [x["to_095"] for x in rs if x["to_095"] is not None]
                att = [x["final_attempt"] for x in rs if x["final_attempt"] is not None]
                srows.append({"arm": a, "n": len(rs), "criterion": "%d/%d" % (len(crit), len(rs)),
                              "median_to_095": int(np.median(crit)) if crit else None,
                              "final_acc_median": float(np.median([x["final_acc"] for x in rs])),
                              "attempt_median": float(np.median(att)) if att else None,
                              "wsls_median": float(np.median([x["w_wsls"] for x in rs])), "bias_median": float(np.median([abs(x["w_bias"]) for x in rs])),
                              "rel_median": float(np.median([x["w_rel"] for x in rs])), "irr_median": float(np.median([x["w_irr"] for x in rs]))})
            body += "<h2>Numbers (computed from the runs)</h2>"
            body += table(srows, [("arm", "arm"), ("n", "n"), ("criterion", "reached 0.95"), ("median_to_095", "median trials to 0.95 (learners)"),
                                   ("final_acc_median", "final P(correct), median"), ("attempt_median", "final attempt rate, median"),
                                   ("wsls_median", "final WSLS, median"), ("bias_median", "final |bias|, median"), ("rel_median", "final rel, median"), ("irr_median", "final irr, median")],
                          tid="sum_" + key, filterable=False, numeric=["n", "median_to_095", "final_acc_median", "attempt_median", "wsls_median", "bias_median", "rel_median", "irr_median"])
            ref = ["pretrained", "no_pretrain"] if key in ("antibias", "effort") else (["effort 0.55 pretrained"] if key == "stages" else [])
            if key in FRONT and key != "t1":
                ref = ["weak start, no_pretrain"]          # every type is drawn against type 1
            try:
                fig_population_by_arm(by_run, os.path.join(site.root, "img", "population_%s.png" % key), exclude=(), only=set(arms_c + ref))
                body += "<h2>Population figures</h2>" + fig("../img/population_%s.png" % key, "Median over seeds per arm (base arms as reference where applicable): task accuracy, fraction of trials attempted, |bias|, WSLS and relevant-evidence weights.", root=os.path.join(site.root, "classes"))
            except Exception as e:
                body += caveat("population figure failed: %s" % esc(e))
            try:
                fig_mechanism_population(by_run, os.path.join(site.root, "img", "mechanism_%s.png" % key), arms=tuple(arms_c + ref))
                body += fig("../img/mechanism_%s.png" % key, "Mechanism and critic by arm, median over seeds.", root=os.path.join(site.root, "classes"))
            except Exception as e:
                body += caveat("mechanism figure failed: %s" % esc(e))
            links_html = []
            for name, pa, pb, la, lb in PAIRINGS:
                if any((pa + "w%03d" % sd) in runs_c or (pb + "w%03d" % sd) in runs_c for sd in range(1, 17)):
                    links_html.append('<a href="../pairs.html#%s">%s vs %s</a>' % (name, esc(la), esc(lb)))
            if links_html:
                body += "<h2>Same-seed pairs</h2>" + note(" &middot; ".join(links_html))
            first = sorted(runs_c, key=lambda x: ([x2[2] for x2 in ARMS].index(arm_of(x)), int(re.search(r"_w(\d+)", x).group(1))))[0]
            body += '<h2>One by one</h2>' + note('<a href="../networks/%s.html">Open the first network of this class</a>; the arrow keys then loop through all %d (arms in order, then seed).' % (first, len(runs_c)))
        elif key == "vanilla":
            body += "<h2>Vanilla versus reward-trained, one method</h2>" + fig("../img/vanilla_vs_rl.png", "", root=os.path.join(site.root, "classes"))
        elif key == "vanilla2025":
            try:
                nv = fig_vanilla2025(by_run, os.path.join(site.root, "img", "vanilla2025.png"))
                if nv:
                    body += "<h2>Learning and mechanism across training (%d seeds)</h2>" % nv
                    body += fig("../img/vanilla2025.png", "One light line per seed (60 shown), the median in blue; the RL base arms' medians dashed, against the fraction of training. Loss on a log scale.", root=os.path.join(site.root, "classes"))
                    body += "<h2>Where every seed ends in the triangle</h2>"
                    body += fig("../img/vanilla2025_triangle.png", "Final triangle coordinates (feature 1) of every 2025-code seed (blue) and of the RL base networks (crosses).", root=os.path.join(site.root, "classes"))
                else:
                    body += caveat("The slow-point analysis of the 2025-code seeds has not been collected yet.")
            except Exception as e:
                body += caveat("vanilla2025 figures failed: %s" % esc(e))
        site.page("classes/%s.html" % key, title, body)
    # ---- all networks table
    body = note("One row per network. Click a header to sort, type to filter (class, arm, seed), click a network to open it.")
    body += table(rows, cols, tid="nets", numeric=numeric, links=links)
    body += "<h2>Population figures</h2>"
    try:
        fig_population_by_arm(population_pages.by_run, os.path.join(site.root, "img", "population_by_arm.png"))
        body += fig("img/population_by_arm.png", "Median over seeds per arm: task accuracy, fraction of trials attempted, |bias|, WSLS and relevant-evidence weights (non-identifiable checkpoints excluded) against training trials. Runs that reached criterion hold their last value.", root=site.root)
    except Exception as e:
        body += caveat("population-by-arm figure failed: %s" % esc(e))
    try:
        fig_mechanism_population(population_pages.by_run, os.path.join(site.root, "img", "mechanism_by_arm.png"))
        body += fig("img/mechanism_by_arm.png", "Mechanism and critic by arm, median over seeds: leading eigenvalue at the slow point per context, selection-vector angle, triangle shares, evidence-axis angle, critic value at trial start, terminal prediction error, how much the critic reads the actor's evidence axis, dimensionality.", root=site.root)
    except Exception as e:
        body += caveat("mechanism-by-arm figure failed: %s" % esc(e))
    try:
        fig_vanilla_vs_rl(population_pages.by_run, os.path.join(site.root, "img", "vanilla_vs_rl.png"))
        body += fig("img/vanilla_vs_rl.png", "Vanilla supervised networks (Cosyne 2025 poster, N20 and N100) and the reward-trained arms under the same slow-point analysis: leading eigenvalue per context, selection-vector angle, triangle shares, number of slow points, against the fraction of training.", root=site.root)
    except Exception as e:
        body += caveat("vanilla-vs-RL figure failed: %s" % esc(e))
    cards = []
    for f, cap in (("curriculum_ridge.png", "Curriculum versus control, 8 seeds each: accuracy and the five ridge-GLM weights across training (thin = seed, thick = median)."),
                   ("histgauss100_ridge.png", "Approach A library, 100 seeds: the same from the ridge GLM."),
                   ("curriculum_psytrack.png", "Curriculum versus control from the PsyTrack fits (8 per arm)."),
                   ("histgauss100_psytrack.png", "Library seeds with a PsyTrack fit (63 of 100 when the farm was stopped).")):
        src = os.path.join(PSY, f)
        if os.path.exists(src):
            import shutil; shutil.copy(src, os.path.join(site.root, "img", f))
            cards.append(fig("img/" + f, cap, root=site.root))
    body += "".join(cards)
    site.page("networks.html", "All networks", body, lead="%d networks. Built from the per-checkpoint rollouts and the saved checkpoints." % len(rows))
    # compare page: the existing PsyTrack-vs-GLM index, copied in
    cmp_dir = os.path.join(PSY, "compare")
    if os.path.isdir(cmp_dir):
        import shutil
        dst = os.path.join(site.root, "compare"); os.makedirs(dst, exist_ok=True)
        for f in os.listdir(cmp_dir):
            if f.endswith(".png") or f.endswith(".txt"):
                shutil.copy(os.path.join(cmp_dir, f), dst)
        body = note("For every network with a PsyTrack fit: PsyTrack (line) against the per-checkpoint ridge GLM (dots, +/- 1 SE); grey where one choice exceeds 95% of trials.")
        for f in sorted(os.listdir(dst)):
            if f.endswith(".png"):
                run = f[:-4]
                txt = os.path.join(dst, run + ".txt")
                stats = "".join(l for l in open(txt) if "corr" in l) if os.path.exists(txt) else ""
                body += '<h3 id="%s"><a href="networks/%s.html">%s</a></h3><pre style="font-size:12px;color:#555">%s</pre>%s' % (run, run, display_name(run), esc(stats), fig("compare/" + f, root=site.root))
        site.page("compare.html", "PsyTrack versus ridge GLM", body)


def methods(site):
    S = [("Networks", "<p>Elise Chang's fork of Song, Yang and Wang (2017) pyrl: a 100-unit leaky tanh RNN policy trained by REINFORCE with a learned value-network baseline (the critic), on the pulse task with four one-hot previous-outcome input channels (pagan_hist). <b>Curriculum</b> networks first go through the rats' nose-poke pre-training (fixed left 360 trials, 20-trial side blocks 1080 trials, random side 720 trials) then the task; <b>control</b> networks start on the task with identical seeds; the <b>library</b> is Elise's 100-seed Approach A set (36,000 trials each, no pre-training).</p>"),
         ("Checkpoints and rollouts", "<p>A checkpoint is saved every 5 updates (curriculum, control) or 10 (library); one update is 72 trials. At every checkpoint the frozen network is rolled out on 1,800 condition-balanced trials of the real task, sequentially, so the previous-outcome input is real. Everything on the Behaviour and Strategy tabs is computed from those rollouts except the validation accuracy, mean reward and P(choice), which come from the training log.</p>"),
         ("Location, frequency and feature-selection performance", "<p>As on the Cosyne 2025 poster. Location performance: P(correct) on LOC-context rollout trials; frequency performance: P(correct) on FRQ-context trials. Feature selection: P(correct) on strongly incongruent trials, where the relevant and the irrelevant evidence are both strong (above the median |evidence| of that feature) and point to opposite sides, averaged over contexts; 50% for a network that integrates all pulses regardless of context, 100% for one that selects the relevant feature (the poster's perf(1,6) and perf(6,1) cells).</p>"),
         ("Recurrent weight change and spectral radius", "<p>Recurrent weight change: Frobenius norm of the change in the recurrent matrix between consecutive checkpoints, divided by the number of updates between them, for the actor (policy) and the critic (value network). Spectral radius: largest |eigenvalue| of the recurrent matrix itself (not yet the Jacobian at a fixed point, which is the poster's largest eigenvalue; planned).</p>"),
         ("P(correct)", "<p>Fraction of rollout trials with a choice whose choice matched the side with more relevant evidence; zero-evidence trials excluded. Validation accuracy is pyrl's own, on 720 trials drawn from whatever the network is being trained on at that checkpoint (so during pre-training it measures pre-training performance).</p>"),
         ("Ridge GLM", "<p>Logistic regression of choice (1 = right) on five regressors, fitted independently at every checkpoint on its 1,800 trials, with a Gaussian prior of precision 1 on every weight (Newton's method). Regressors: relevant evidence and irrelevant evidence (z-scored over the whole record, so the weights are per SD), previous choice (+1 right, -1 left, 0 none), win-stay/lose-switch (previous choice times +1 if rewarded, -1 if not), bias. +/- 1 SE from the inverse Hessian. On a checkpoint where one choice exceeds 95% of trials the split of the habit between bias, previous choice and win-stay is not identifiable; those checkpoints are greyed and excluded from summaries.</p>"),
         ("PsyTrack", "<p>Roy et al. 2021, as used on the rats: a per-trial random walk on each weight with a per-session jump, the checkpoints as sessions, hyperparameters by evidence optimisation. One to five CPU-hours per network; across 62 networks it agrees with the ridge GLM to r 0.99-1.00 per weight wherever the weights are identifiable (compare page), so the GLM is the default here and PsyTrack is shown dotted where a fit exists.</p>"),
         ("Selection index and strategy share", "<p>Selection index: (rel - irr) / (|rel| + |irr|) from the GLM weights; 1 = only the relevant feature, 0 = both equally, negative = the irrelevant one more. Strategy share: |w| x SD(x) of each regressor (|w| for the bias), normalised to sum to one at each checkpoint; the fraction of the decision variable each source accounts for, independent of the overall scale of the weights.</p>"),
         ("Psychometrics and matrices", "<p>Evidence is the signed pulse-count difference per feature (+ = right; for frequency, + = more high-pitched pulses). Psychometric curves: P(right) in seven quantile bins of one feature, separately per context; matrices: P(right) in 5 x 5 quantile bins of location x frequency evidence, per context, cells with fewer than 3 trials blank.</p>"),
         ("Weights", "<p>From the saved parameters at every checkpoint: L2 norms of the input weight rows by channel group (evidence LMR/HML, context LOC/FREQ, the four history channels), of the output weights, and the spectral radius (largest |eigenvalue|) of the recurrent matrix of the policy and of the critic.</p>")]
    from explorer_lib import methods_page
    methods_page(site, "methods.html", "Methods", S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="/mnt/big/pyrl_work/explorer_site")
    ap.add_argument("--only", default=None, help="substring filter on run names")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--pages-only", action="store_true", help="skip figure generation (uses the cache)")
    ap.add_argument("--tree", default="main", choices=sorted(TREES),
                    help="which documentation tree to build: main (site root) or elise (<site>/elise)")
    a = ap.parse_args()

    # The tree selects the classes, the front page, the site root and the nav; everything below
    # reads CLASSES/FRONT/TREE as globals, so rebinding them here is the whole switch.
    global CLASSES, FRONT, TREE
    TREE = TREES[a.tree]
    CLASSES = [c for c in CLASSES if c[0] in TREE["keys"]]
    FRONT = tuple(k for k in TREE["front"] if k in TREE["keys"])
    root = os.path.join(a.site, TREE["sub"]) if TREE["sub"] else a.site

    runs = sorted(os.path.basename(p).replace("_record.npz", "") for p in glob.glob(os.path.join(RECORDS, "*_record.npz")))
    runs = [r for r in runs if arm_of(r) != "other"
            and CLASS_OF_ARM.get(arm_of(r)) in TREE["keys"]
            and (a.only is None or a.only in r)]
    print("tree %s -> %s: %d networks" % (a.tree, root, len(runs)))
    os.makedirs(os.path.join(root, "img"), exist_ok=True)
    jobs = [(r, root, True) for r in runs]
    if not a.pages_only:
        with Pool(a.workers) as pool:
            for i, (run, err) in enumerate(pool.imap_unordered(work, jobs)):
                print("%3d/%d %s %s" % (i + 1, len(jobs), run, "" if err is None else "FAILED\n" + err), flush=True)
    site = Site(root, TREE["title"],
                nav=[("index.html", TREE["index_title"]), ("networks.html", "All networks"), ("pairs.html", "Pairs"), ("compare.html", "PsyTrack vs GLM"), ("methods.html", "Methods")] + list(TREE["nav_extra"]),
                wiki_html="file:///Users/mpagan/Documents/GitHub/_master/html/wiki/areas/conferences_and_talks.html")
    rows, by_run = [], {}
    for run in runs:
        try:
            r = compute(run, root)
        except Exception as e:
            print("skip", run, e); continue
        rows.append(summary_row(r)); by_run[run] = r
    # arrow-key neighbours: within each CLASS of models, arms in ARMS order then seed, looping
    arm_order = {a: i for i, (_, _, a, _) in enumerate(ARMS)}
    groups = {}
    for run in by_run:
        groups.setdefault(CLASS_OF_ARM.get(arm_of(run), "other"), []).append(run)
    for g in groups.values():
        g.sort(key=lambda x: (arm_order.get(arm_of(x), 99), int(re.search(r"_w(\d+)", x).group(1))))
        for i, run in enumerate(g):
            network_page(site, by_run[run], "curriculum" in run, (g[i - 1], g[(i + 1) % len(g)]))
    population_pages.by_run = by_run
    population_pages(site, rows)
    pair_pages(site, by_run)
    methods(site)
    site.finish()


if __name__ == "__main__":
    main()
