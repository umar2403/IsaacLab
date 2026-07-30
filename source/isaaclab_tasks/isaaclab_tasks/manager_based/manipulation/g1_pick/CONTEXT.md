# g1_pick — working context (as of 2026-07-29)

Narrative of how the grasp-selection work got to its current state, what was learned,
and what is still open. Companion to `OPTIMIZER_IMPLEMENTATION_SPEC.md` (the algorithm)
and `grasp_selection/README.md` (the code).

**End goal:** train a teacher policy in sim → distill to a student → deploy on the real
G1 for the pick task. Hardware is the destination, which is why grasp *quality* matters
and not just task success.

---

## 1. Where we started

A working RL pick policy with UltraDexGrasp/BODex shaping already integrated: a 32-grasp
library, `sample_grasp_goal` picking one entry per episode, and two reward terms
(`grasp_goal_palm`, `grasp_goal_hand`) pulling the hand toward it. The grasp index was
**chosen by hand** (`fixed_grasp_idx = 23`).

**Goal of this work:** replace the human choice with an optimizer.

## 2. The optimizer

Built `grasp_selection/` from `OPTIMIZER_IMPLEMENTATION_SPEC.md` (Version 2,
sphere-contact-gated FSWO). CPU-only, no Isaac Lab import. `g1_pick_env_cfg.py` resolves
`_OPTIMAL_GRASP_IDX = get_optimal_grasp_idx()` at import from a cached `scores.json`.

Validated three independent ways — verbatim spec code, the spec's `pytorch_kinematics`
FK path, and this implementation — all returning identical rankings.

**Three deviations from the spec, all deliberate and commented in the code:**

1. **Score from the NNLS residual `‖Lα‖²`, not `αᵀQα`.** Identical in exact arithmetic,
   but on near-degenerate contacts NNLS returns `|α| ~ 1e11` and the quadratic form
   cancels catastrophically — one grasp scored **+3.5e7**, impossible for a quantity
   defined as ≤ 0, which corrupted the tie-break ordering.
2. **Added `MAX_TIP_PENETRATION` (3 mm).** The spec's gate is `gap <= contact_dist` with
   `gap` allowed to be arbitrarily negative, so fingers buried inside the cube count as
   contacts *and* inflate `n_spheres`, which drives the ranking — a bias toward
   interpenetrating grasps. It picked one with a fingertip 7.6 mm inside the cube, caught
   by eye in the 3D viewer. Disable with `--max-penetration -1`.
3. **numpy FK instead of `pytorch-kinematics`**, to avoid a new dependency in the conda
   env. Agrees with `pk` to 1.9e-7 m.

## 3. Grasp indices are NOT portable — the biggest source of confusion

An index only means something *relative to one library file*. Four libraries exist, and
the same number refers to a different physical grasp in each. This caused a real mix-up
mid-project (a run believed to be on #12 was actually on #16).

| file | md5 | what it is |
|---|---|---|
| `cube_5cm_grasps_valid.npz` | `053bc74c` | **ACTIVE** — Shahid's synthesis run, optimizer picks **#12** |
| `...valid.shahid_import.npz` | `053bc74c` | same file, kept as the named copy |
| `...valid.npz.clone_backup` | `da9008a3` | the top-down 32-lib; optimizer picks **#23** |
| `...valid.no_topdown_rebuild.npz` | `33d0ef0e` | 300-pool rebuild, no top-down; picks **#16** (pool #127) |
| `...valid.optimizer_rebuilt.npz` | `1100644a` | 300-pool rebuild, top-down kept; picks **#7** (pool #79) |

**Ground truth for what a run actually trained on is
`logs/rsl_rl/g1_pick/<run>/params/env.yaml` → `fixed_grasp_idx`.** Kit swallows `print()`
during AppLauncher startup, so console output is not reliable. Always quote an index
*together with its library*.

Shahid's library and this repo's 300-pool share only **1 of 32** grasps. This repo's pool
contains **zero** five-finger wraps, which is why #12 (5 fingers) cannot be reproduced
from it at any gate setting.

## 4. Grasp selection history

| pick | library | why it was rejected / kept |
|---|---|---|
| #17 | top-down 32 | first automatic pick; fingertip 5 mm inside the cube |
| #7 (pool #79) | 300-pool top-down rebuild | 10 spheres but middle fingertip **7.6 mm inside**; rejected on sight → motivated the penetration gate |
| #23 | top-down 32 | what `model_7998` trained on; only 4 fingers, one fingertip 2.2 mm in, a collision sphere **9.2 mm** in, and part of its grip just rests on the cube's top face |
| **#12** | **Shahid (active)** | **5 fingers, every fingertip outside (min +6.2 mm), deepest sphere −0.8 mm, grips two opposing faces** |

#18 (top-down 32, = pool #3) is the **fallback**: top-down, zero fingertip penetration
(min +0.5 mm), 4 fingers. Preferred over #23 for hardware.

## 5. The top-down filter and `posture_rew` — what they were for

Two separate mechanisms that are easy to conflate:

- **`posture_rew`** (inside `compute_task_reward`) pulls the wrist to a hardcoded point
  **8 cm straight above the cube**. It predates the UltraDex work; its job is pre-grasp
  staging. It knows nothing about the grasp library, so it *only* agrees with a
  near-top-down goal and fights a side grasp horizontally.
- **The top-down filter** (`build_goal_library.py`, `TOPDOWN_MAX_DEG = 50`) keeps only
  near-vertical approaches, added in commit `5c4735f424` so the library "only contains
  palm-down approaches the G1 arm can actually reach". The reachability half of that
  claim is the author's empirical finding and has **never been IK-verified**.

Knock-on effect: `grasp_goal_hand` is *gated* on palm-to-goal distance, so if the palm
never reaches the goal the finger imitation silently pays ≈ 0.

Note `posture_rew` tracks `right_wrist_yaw_link` while `grasp_goal_palm` tracks
`R_hand_base_link` — different bodies, a few cm apart.

## 6. The no-top-down experiment (branch `umar/g1-pick-no-topdown`)

Removed the top-down filter (`--topdown-deg 180`), set `use_posture=False`, raised
`grasp_goal_palm` weight 1.0 → 2.0. Trained `model_7998` → `model_10997` (3000 iters) on
the no-top-down rebuild, `fixed_grasp_idx: 16`.

| metric | start | end |
|---|---|---|
| `Episode_Termination/target_lifted` | 0.431 | **0.940** |
| `Episode_Reward/task_reward` | 1.761 | 3.674 |
| `Episode_Reward/grasp_goal_palm` | 0.029 | **0.029** |
| `Episode_Reward/grasp_goal_hand` | 0.001 | **0.001** |

**Result: the pick got excellent (94%), the UltraDex mimicry did not happen at all.**

**Diagnosis:** `success_bonus = is_lifted × is_grasped × 1000` dwarfs any additive
shaping (≤ 2/step over ~37-step episodes), and worse, the two objectives *compete* — the
policy would have to risk a 94%-reliable grasp to chase pose-matching. So it correctly
ignored the goal and optimised its own grasp. **The top-down filter was never what
blocked mimicry; the reward scale was.**

Also verified along the way: the Inspire hand's slave joints **are** driven now
(`right_hand` actuator matches `R_.*_joint` at stiffness 100, and
`InspireMimicAction.apply_actions` sets their targets via the transmission ratios), so
finger-pose mimicry is mechanically achievable. Earlier notes claiming the mimic joints
were inert are outdated. Finger action noise std is ~3.9 vs ~0.36 for the arm — the
policy has no incentive to hit a precise finger pose, not an inability.

## 7. Current state — pose-gated success bonus

Because the destination is hardware, grasp quality wins over training convenience, so the
target is **#12** (Shahid library) despite being a 65° side approach.

Structural fix instead of reweighting — the bonus itself is now conditional:

```python
pose_match    = (1 - tanh(pos_err / 0.10)) * (1 - tanh(ang_err / 0.8))
success_scale = success_floor + (1 - success_floor) * pose_match     # floor 0.2
success_bonus = is_lifted * is_grasped * success_scale * 1000.0
```

Measured on `R_hand_base_link` against the live goal (which tracks the cube), so it asks
*"are you holding it the UltraDex way at the moment you lift it?"* A wrong-pose pick still
pays 200 — the floor keeps the pick signal alive if the pose proves hard — and a matched
pose pays the full 1000.

Verified in a 3-iteration smoke run: `target_lifted` 0.94 → 0.96 (pick intact) while
`task_reward` fell 3.674 → 0.749. That ratio is ≈ 0.2, i.e. the policy is collecting
**exactly the floor** — `pose_match ≈ 0` for its current grasp, so the gate is biting.

**This run has since completed — see section 8 for the (negative) result.**
The command used was:

```bash
cd /home/umar/IsaacLab
source /home/umar/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
source /home/umar/IsaacLab/_isaac_sim/setup_conda_env.sh
PYTHONPATH=/home/umar/IsaacLab/source/isaaclab_tasks:$PYTHONPATH \
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-G1-Pick-v0 --headless --num_envs 1024 \
  --resume --load_run 2026-07-27_01-36-40_no_topdown_warm --checkpoint model_10997.pt \
  --max_iterations 3000 --run_name grasp12_posegated
```

**The metric that decides the experiment is `grasp_goal_palm`.** It has been pinned at
0.029 across 3000 iterations. If it climbs, the gate worked. If it stays flat while
`target_lifted` sinks, the 65° pose is out of the arm's reach — fall back to #18.
Knobs if too harsh: raise `success_floor` toward 0.4, or loosen `pose_pos_std` /
`pose_ang_std`.

## 8. Result of the pose-gated run — it did NOT work

`2026-07-28_01-29-33_grasp12_posegated`, 3000 iterations, `model_10997` → `model_13996`,
`fixed_grasp_idx: 12`, `pose_gated_success: true`, `success_floor: 0.2`.

| metric | start | 50% | end | max |
|---|---|---|---|---|
| `Episode_Termination/target_lifted` | 0.506 | 0.942 | **0.933** | 0.954 |
| `Episode_Reward/task_reward` | 0.511 | 0.989 | 1.075 | 1.111 |
| **`Episode_Reward/grasp_goal_palm`** | 0.019 | 0.028 | **0.028** | 0.030 |
| **`Episode_Reward/grasp_goal_hand`** | 0.002 | 0.002 | **0.002** | 0.002 |
| `Episode_Termination/distractor_dropped` | 0.016 | 0.054 | 0.062 | 0.075 |
| `Train/mean_episode_length` | 16.5 | 15.6 | 15.7 | 21.3 |

**The pose terms are still flat.** `grasp_goal_palm` sat at the same ~0.028–0.030 plateau
as the previous run; `grasp_goal_hand` never moved off 0.002. The policy simply got its
pick rate back up (0.51 → 0.93) and collected the 200-point floor, exactly as before.
`task_reward` rising 0.511 → 1.075 is that recovering pick rate, **not** improving pose
match — if `pose_match` had improved, `grasp_goal_palm` would have moved too, since both
read the same pose error.

So a 5× incentive (200 vs 1000) applied to the terminal bonus was **not** enough.

### Why — the current best explanation

The decisive observation is that **`grasp_goal_palm` is dense, ungated, and at weight 2.0,
and it still produced zero movement in 6000 total iterations across two runs.** That rules
out "the shaping is too small or too gated" as the whole story. Two candidates remain:

1. **Exploration / sparse conjunction.** The extra 800 points only materialise when the
   policy matches the pose *and* still completes a grasp-and-lift. Random exploration
   essentially never produces that conjunction, so the gradient toward it is never
   sampled. The episode is also only ~16 steps (~0.5 s at 30 Hz) — the policy dives
   straight in, leaving little time to align a wrist pose en route.
2. **The 65° pose is not reachable** for this arm at the tray, so `pose_match` cannot rise
   and settling for the floor is genuinely optimal.

These have not been distinguished yet. **Cheap test:** `play_with_goal_markers.py` prints
the live palm→goal distance and colours the goal sphere green on reach. Run it on
`model_13996` — if the palm settles ~10–15 cm from the goal, the pose is nearly reachable
and this is an exploration problem (1). If the distance stays large or the arm visibly
strains/hits limits, it is reachability (2).

### What to try next, roughly in order of expected value

- **Curriculum / phase split.** Train reaching the goal *pose* first with the lift reward
  disabled, then re-enable lifting. This removes the conjunction that (1) says is the
  blocker — the policy learns the pose while it is the only thing being rewarded.
- **Shape the approach, not just the endpoint.** Reward pose match *along the trajectory*
  with real weight, rather than only at the moment of lift.
- **Slow the policy down.** ~16-step episodes mean a ballistic dive. An action-rate or
  approach-speed constraint would give the wrist time to align.
- **Try #18** (top-down, zero fingertip penetration) to remove reachability as a variable.
  If the pose terms move for #18 but not #12, that settles (1) vs (2) directly.
- Reconsider whether pose imitation is the right mechanism at all — an alternative for the
  hardware goal is to keep the policy's own grasp and validate it with the optimizer as an
  offline scoring tool, which is what `grasp_selection/` already does well.

## 9. Open questions

- **Arm reachability at 65° is unverified.** No IK check has ever been run for a
  side-approach wrist pose at the tray. This is the main risk for #12.
- **The grasp has never been tested for stability.** `target_object_lifted` is a
  single-frame height check with no dwell requirement, and episodes average ~37 steps
  (~1.2 s at 30 Hz), so nothing has ever asked the policy to *hold*. Making it require
  N consecutive steps above the height is a two-line change and would make the success
  criterion honest. Not done — deliberately deferred.
- **`#12`'s approach corridor passes 5.7 cm from distractor #8** (cube half-edge 2.5 cm,
  hand ~10 cm wide). Comparable to #23's 7.0 cm, so probably not disqualifying, but worth
  watching `Episode_Termination/distractor_dropped`.

## 10. Running things

`./isaaclab.sh -p` picks the wrong Python in a non-interactive shell. Use:

```bash
source /home/umar/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
source /home/umar/IsaacLab/_isaac_sim/setup_conda_env.sh
PYTHONPATH=/home/umar/IsaacLab/source/isaaclab_tasks:$PYTHONPATH python <script>
```

`isaacsim` is not pip-installed in the env — it comes from the bundled `_isaac_sim/`.
`isaaclab_tasks` in site-packages is a broken stub, hence the `PYTHONPATH`.

Inspect a grasp visually (CPU-only, writes a self-contained three.js page):

```bash
python grasp_sampler/visualize_grasp_offline.py --mode contacts   # green = contacting spheres
```
