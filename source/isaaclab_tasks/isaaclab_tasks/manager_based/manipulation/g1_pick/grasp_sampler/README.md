# UltraDexGrasp → Inspire Hand: Goal-Grasp Pipeline for G1 Pick

This document explains, from the ground up, how the [UltraDexGrasp](https://github.com/InternRobotics/UltraDexGrasp)
grasp-synthesis pipeline was adapted to the **Inspire Hand** on the **Unitree G1**
robot, and how its output is used to improve the reinforcement-learning policy in
the `g1_pick` task. It is written so that someone with no prior knowledge of the
project can follow it end to end.

---

## 1. The problem we are solving

The `g1_pick` environment trains a G1 humanoid (right arm + Inspire dexterous
hand) to pick a 5 cm cube off a tray using reinforcement learning (RL). The
original reward measures *distance from the fingers to the cube center* — it says
"get close to the cube", but never says **what a good grasp looks like**: where
the palm should be, from which direction to approach, how the fingers should be
shaped. The policy has to discover all of that by trial and error, which is slow
and produces grasps that don't generalize.

A **grasp synthesizer** solves the complementary problem: given an object's
shape, it computes physically stable grasp configurations — a 6-DoF palm pose
plus finger joint angles that achieve *force closure* (the finger contact forces
can resist gravity and disturbances). If we give the RL policy these grasps as
**goals** and reward it for approaching them, it no longer has to discover grasp
geometry from scratch.

```
                       OFFLINE (once)                          ONLINE (RL training)
     ┌────────────────────────────────────────────┐   ┌──────────────────────────────────┐
     │  cube mesh ─► BODex grasp optimizer         │   │  each episode reset:             │
     │              (UltraDexGrasp pipeline)       │   │    pick nearest goal grasp G*    │
     │        │                                    │   │        │                         │
     │        ▼                                    │   │        ▼                         │
     │  100 candidate grasps                       │   │  reward the policy for moving    │
     │        │                                    │   │  palm → G* pose and fingers →    │
     │        ▼                                    │   │  G* joint angles (small shaping  │
     │  calibrate + re-center + filter in Isaac    │   │  bonus on top of the original    │
     │  Lab  ─►  32-grasp goal library (.npz)     ─┼──►│  task reward)                    │
     └────────────────────────────────────────────┘   └──────────────────────────────────┘
```

### What UltraDexGrasp actually is

UltraDexGrasp (ICRA 2026, Shanghai AI Lab) is **not a pretrained network** — it is
a *data-generation pipeline*. Internally it delegates grasp computation to
**BODex**, an optimization engine that:

1. samples candidate palm poses on an inflated hull around the object,
2. runs a GPU-parallel gradient optimization that pulls fingertips onto the
   object surface while enforcing force closure, collision avoidance, and joint
   limits,
3. outputs, per grasp, three stages: **pregrasp** (fingers 1.3 cm off the
   surface), **grasp** (in contact), and **squeeze** (pressed in).

UltraDexGrasp ships with configs for the *XHand* and *LEAP* hands. Nothing knows
about the Inspire Hand — that is the porting work described below.

---

## 2. What had to be built

### 2.1 A software environment (the `ultradex` conda env)

BODex needs PyTorch with CUDA extensions (pytorch3d, torch-scatter, a C++
collision wrapper built on `coal`). Two machine-specific gotchas:

- The GPU here is an **RTX 5060 (Blackwell, sm_120)**. The torch version in the
  UltraDexGrasp README (2.4.1 + CUDA 11.8) has no Blackwell kernels, so we use
  **torch 2.11 + cu128** and compile everything against a **CUDA 12.8 toolkit
  installed inside the conda env** (the system CUDA 13.3 is rejected by torch's
  build checks). Builds need:
  `CUDA_HOME=$CONDA_PREFIX` and `CPATH=$CONDA_PREFIX/targets/x86_64-linux/include`.
- warp-lang's torch interop moved (`wp.torch.*` → `wp.*`); one line in BODex's
  `world_mesh.py` was patched, and the coal wrapper needed `-std=c++17`.

Everything lives in `grasp_sampler/ultradex_repo/` (the cloned repo) and
`ultradex_repo/third_party/` (BODex_api, pytorch3d).

### 2.2 Teaching BODex the Inspire Hand

BODex describes a hand with four pieces, all created under
`ultradex_repo/third_party/BODex_api/src/bodex/content/`:

| Piece | File | What it contains |
|---|---|---|
| Hand model | `assets/robot/inspire_hand/inspire_hand_right.urdf` | links, joints, meshes (taken from the public `dex-urdf` project, then heavily modified — see §3) |
| Robot config | `configs/robot/inspire_right.yml` | joint list, fingertip links, **collision spheres**, self-collision pairs |
| Palm frame | `configs/robot/hand_pose_transfer/inspire.yml` | how BODex's canonical grasp frame maps onto the hand's root frame |
| Task config | `configs/manip/sim_inspire_sim2real/fc_right.yml` | contact points (the 5 fingertips), seed pose, force-closure parameters |

Details worth understanding:

- **Collision spheres.** BODex represents the hand as ~40 spheres for fast
  GPU collision checking. These were generated automatically
  (`gen_spheres.py` script) by slicing each link's collision mesh along its long
  axis and fitting a sphere per slice; the palm's collision boxes were converted
  analytically. The **first sphere of each fingertip link is placed exactly at
  the fingertip** because BODex's contact strategy refers to contacts as
  `link_name/sphere_index` (e.g. `thumb_distal/0`).

- **The canonical grasp frame.** When BODex seeds palm poses around the object
  it works in a canonical frame: **+x points at the object (approach), +z along
  the fingers, +y from pinky toward index**. This was reverse-engineered by
  computing forward kinematics for the XHand (whose transfer matrix is known)
  and matching physical directions. For the Inspire URDF the root frame already
  matches this convention, so the rotation is identity; the translation
  `t = [0, 0, -0.09]` pulls the root back 9 cm so the **palm center** (not the
  wrist) lands on the sampled approach point — the Inspire hand is much longer
  from root to fingertip (21 cm) than the XHand (6 cm).

- **Joint order.** The 6 actuated joints are listed in the config in exactly the
  same order the `g1_pick` policy uses:
  `[thumb_yaw, thumb_pitch, index, middle, ring, pinky]` (proximal joints).
  This makes the mapping between BODex output and the policy's action space the
  identity (`bodex_2_sim_q_idx = [0..5]` in `util/bodex_util.py`, where an
  `'inspire'` hand type was added).

- **The cube asset.** BODex expects each object as a directory
  (`mesh/simplified.obj`, `urdf/coacd.urdf`, `info/simplified.json`). A unit
  cube was created at `ultradex_repo/asset/object_mesh/cube/` and scaled to
  0.05 m at synthesis time.

### 2.3 The synthesis driver

`synthesize_inspire_grasps.py` instantiates UltraDexGrasp's `GraspSynthesizer`
with `hand_type='inspire'`, runs it on the cube, and saves
`grasp_dataset/cube_5cm_grasps.npz`:

```
grasp_pose : float32 (N, 1, 3, 13)
             N grasps × 1 hand × 3 stages (pregrasp/grasp/squeeze) ×
             [x y z  qw qx qy qz  thumb_yaw thumb_pitch index middle ring pinky]
```

Poses are the **hand root pose expressed in the object's frame** (object centered
at the origin, resting on a virtual table). Synthesis takes about a minute for
100 grasps on the RTX 5060.

We deliberately **skip UltraDexGrasp's later stages** (cuRobo arm planning +
SAPIEN execution): those generate UR5e arm trajectories which are useless for a
G1 — our RL policy is the "arm planner".

---

## 3. The hard part: making the model match the simulator

This is where most of the engineering time went, and the findings matter beyond
this pipeline.

### 3.1 Discovery 1: the USD hand's coupled joints are broken

The Inspire Hand has 6 motors driving 12 joints — each finger's intermediate
(and the thumb's distal) segments are mechanically **coupled** to the proximal
joint. In `g1_with_hands_final.usd` this coupling is modeled with
`PhysxMimicJointAPI` (gearings −1.6 / −2.4 / −1.0), and the slave joints have
**no position drives at all**.

Consequences, verified by probing the live `Isaac-G1-Pick-v0` env
(`probe_env_hand.py`) and a standalone scene (`measure_coupling.py`):

- The `InspireMimicAction` class in `g1_pick_env_cfg.py` writes position targets
  for the slave joints using hardware ratios (0.8024 / 0.9487 / 1.0843). **These
  targets do nothing** — a joint without a drive ignores position targets. The
  code comment claiming the software path "overrides" the USD gearing is wrong.
- The PhysX mimic constraint is nearly undamped (dampingRatio 0.005) and its
  negative gearings fight the joint limits. Measured behavior in the training
  env: thumb intermediate/distal settle at **−0.16 / −0.24 rad** (slightly
  bent backward!) regardless of command; finger intermediates drift anywhere in
  **0.0–1.3 rad** depending on motion history — effectively uncontrolled.
- The previously-trained policy succeeded anyway: it grasps mostly with the
  proximal finger segments and the palm. But this is a **sim-to-real risk** —
  the real hand's distal segments behave completely differently.

**How the pipeline handles it:** the synthesis URDF's slave joints were
**frozen (converted to fixed joints)** at the empirically measured postures
(thumb −0.16/−0.24, fingers 1.15 rad), so BODex plans with the hand shape the
simulator actually produces.

### 3.2 Discovery 2: two different Inspire Hands

The public `dex-urdf` Inspire model and the USD in this repo are different
revisions of the hand. A rigid-body calibration (Kabsch fit over 10 link
origins, done in `validate_grasps_isaaclab.py`) shows the **link origins match
to < 1.5 mm** — but the distal geometry differs by **1–3 cm at the fingertips**.
A grasp planned to touch the cube in the URDF model misses it by centimeters in
the USD.

**How the pipeline handles it — empirical grip re-centering:** for every grasp,
the validator closes the *simulated* hand to the grasp's joint configuration
(with no cube present), reads where the sim's own fingertips actually are, and
**moves the grasp's object-relative position so the cube sits at the sim hand's
real pinch center** (thumb tip vs index/middle midpoint). The palm pose and
joint angles stay as BODex planned them; only the object offset is corrected
(median correction ≈ 3.3 cm).

### 3.3 Discovery 3: frame calibration between URDF and USD

BODex outputs poses relative to the URDF's `base` link; the simulator reports
the hand as `R_hand_base_link`. These frames differ by a 90° axis permutation.
Rather than guessing, the validator computes it: it puts both models in the same
joint configuration, collects 10 corresponding link-origin positions, and solves
the least-squares rigid transform (Kabsch/SVD). Residual: **0.3 mm mean**. The
resulting `T_usdbase_urdfbase` is stored inside the goal library and applied by
the reward code.

### 3.4 Why "does it hold in mid-air?" was abandoned as the filter

The classic validation (place object in the grasp, close fingers, turn on
gravity, check it holds) was implemented — and taught us the discoveries above —
but ultimately **cannot work with this USD**: the distal finger segments are
stochastic (§3.1), so whether a pinch holds in mid-air is essentially a coin
flip of constraint dynamics, not a property of the grasp. Since the grasps are
used as *reward shaping goals* (the policy closes the final centimeter itself,
exactly as it already learned to do), the filter used instead is:

1. **Geometric soundness** in the synthesis model: thumb + ≥2 fingers within
   2 cm of the cube surface at the grasp configuration.
2. **Tray compatibility**: no fingertip and no palm below the cube's underside
   (a cube on a tray cannot be grasped from below).

`build_goal_library.py` applies both and keeps the best 32 grasps →
**`grasp_dataset/cube_5cm_grasps_valid.npz`** — the goal library.

---

## 4. How the goals plug into the RL policy

All integration code is in [`mdp/grasp_goal.py`](../mdp/grasp_goal.py), wired
into [`g1_pick_env_cfg.py`](../g1_pick_env_cfg.py). Three pieces:

### 4.1 Goal assignment at every episode reset (`sample_grasp_goal`)

An event term that runs after the cube is placed on the tray:

1. Load the 32-grasp library once (converting each grasp's root pose into the
   USD hand frame using the stored calibration `T`).
2. Transform all 32 candidate palm poses from the cube's frame into world
   coordinates using the cube's freshly randomized pose.
3. Score every candidate with **reach cost + clutter risk** and pick the argmin:
   - *reach cost* = distance from the robot's current palm to the candidate pose;
   - *clutter risk* = number of distractors sitting within 7 cm of the
     candidate's **approach corridor** (the segment from 12 cm behind the palm
     goal, along the palm normal, down to the cube center). Each blocking
     distractor costs as much as 0.5 m of extra reach, so a slightly farther
     grasp with a clean corridor beats a near grasp that would plough through
     clutter. If *everything* is blocked, the least-blocked grasp is still
     chosen — the policy then clears it, exactly as the pre-UltraDexGrasp
     policy learned to do (the distractor penalties that taught that behavior
     are unchanged).
4. Store per-env goal buffers: `goal_pos_w` (3), `goal_quat_w` (4),
   `goal_hand_q` (6) plus the pregrasp/squeeze finger stages and the chosen
   grasp index.

**CURRENT MODE — single fixed grasp.** The selection above turned out to have a
fundamental flaw for a policy that cannot observe its goal (see problem 17 in
§6): with the goal varying per episode, the reward is a function of a hidden
variable, and the best the policy can do is optimize the *average* goal —
hovering near the cube at the mean offset, never committing to any one grasp.
The config therefore now sets `fixed_grasp_idx: 15` in the `sample_grasp_goal`
event: **every env, every episode trains against the same library grasp**
(index 15 — the grasp most frequently chosen by nearest-palm selection, i.e.
empirically the most reachable). This makes the goal a deterministic function
of the observed cube position — fully learnable with no observation change.
In this mode the scoring machinery above (including the distractor-corridor
logic) is bypassed entirely; the original distractor penalties, which taught
the pre-UltraDexGrasp policy to clear clutter, carry that concern alone.
Set `fixed_grasp_idx: -1` to restore per-episode selection — but only do that
after making the policy goal-conditioned (§8, item 5).

### 4.1b Goals are LIVE, not frozen (`update_live_goals`)

The grasps are stored relative to the cube, and the cube gets pushed around
during an episode. The reward functions therefore call
`update_live_goals()` every step, which re-attaches each env's *chosen* grasp
(by its stored index) to the cube's **current** pose. Without this the goal
would stay where the cube *spawned*; after any contact moved the cube, the
policy would be rewarded for hovering in empty space (this was an actual bug,
spotted during play visualization — see problem 14 in §6).

Note the split: the expensive BODex optimization stays offline (a solve takes
~1 min, a control step is 33 ms), but the cheap object-frame → world transform
runs online every step. That combination gives "live" goals at zero cost.

### 4.2 Two shaping rewards

Added to `RewardsCfg` alongside the six original terms (which are unchanged):

**Palm reward** (weight 1.0) — pulls the palm toward the goal pose:

$$r_{palm} = 1 - \tanh\left(\frac{\lVert p_{palm} - p^*\rVert}{0.15}\right)$$

**Hand-configuration reward** (weight 0.3) — pulls the six proximal joints
toward the goal grasp's joint angles, **gated** so it only pays once the palm is
near the goal (otherwise the policy would curl its fingers from across the room):

$$r_{hand} = \underbrace{\left(1-\tanh\frac{\lVert p_{palm}-p^*\rVert}{0.20}\right)}_{\text{gate}} \cdot \left(1 - \tanh\frac{\lVert q - q^*\rVert}{0.5}\right)$$

History: the first training run used weight 0.5 and gate 0.10 m; the policy
plateaued hovering ~10 cm from the goal — exactly at the gate edge, where the
palm gradient was weak and the finger reward had not switched on yet. The
current values (weight 1.0, gate 0.20 m) keep the reward pulling through that
zone. The weights remain small relative to the task reward (posture+reach+grasp
≈ 4/step, success bonus 1000): the goals *guide*, they do not dominate. If the
policy finds a better grasp than the library's, the task reward still wins.

### 4.3 What did NOT change

Observations (still 96-dim), actions (13-dim), terminations, curriculum, PPO
hyperparameters — all untouched. A trained checkpoint from before remains
loadable.

---

## 5. How training works (for someone new to the project)

The policy is trained with **PPO** (Proximal Policy Optimization) via RSL-RL:

- **1024 parallel simulations** of the same scene run on the GPU. Each episode:
  cube spawns at a random spot on the tray, the robot must lift it 29 cm.
- Every control step (30 Hz) the policy network (MLP, 512-256-128) reads the
  96-dim observation (joint states, cube position, fingertip positions, …) and
  outputs 13 joint-position targets (7 arm + 6 hand).
- The reward each step = original task reward (approach + grasp + lift +
  success − penalties) **+ the two new goal-grasp terms**.
- Every 32 steps, the collected experience (32 × 1024 = 32,768 transitions) is
  used to update the network. 5000 such iterations ≈ 160 M steps ≈ 10 hours on
  the RTX 5060.

### Commands

```bash
# terminal 1 — training (tmux recommended so it survives the terminal closing)
tmux new -s g1_training
cd /home/umar/IsaacLab
conda activate env_isaaclab
source _isaac_sim/setup_conda_env.sh          # required: binary Isaac Sim install
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-G1-Pick-v0 --headless --num_envs 1024 --max_iterations 5000
# detach: Ctrl+B then D    reattach: tmux attach -t g1_training

# terminal 2 — monitoring
conda activate env_isaaclab
tensorboard --logdir /home/umar/IsaacLab/logs/rsl_rl/g1_pick
# open http://localhost:6006
```

### What to watch in TensorBoard

| Curve | Expected behavior |
|---|---|
| `Episode_Reward/grasp_goal_palm` | rises from the very first iterations (reaching is easy) |
| `Episode_Reward/grasp_goal_hand` | ~0 until palms reliably reach goals, then rises; if still flat at iter ~1000, the 10 cm gate may be too tight |
| `Episode_Reward/task_reward` | **the key comparison**: overlay a pre-goal-shaping run and check whether this curve takes off earlier (earlier take-off = faster grasp acquisition = the shaping worked) |
| `Train/mean_reward` | totals look *worse* early vs old runs (penalties + time spent chasing goals) — judge by `task_reward` |

### Resuming a crashed/stopped run

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-G1-Pick-v0 --headless --num_envs 1024 \
  --resume --load_run <timestamp_folder> --checkpoint model_<N>.pt
```

---

## 6. Problems we hit and how we solved them

A chronological, complete list. The model-mismatch discoveries (§3) are
summarized here too so this section can be read standalone.

### A. Getting the software to build at all

| # | Problem | Symptom | Root cause | Fix |
|---|---|---|---|---|
| 1 | torch from the UltraDexGrasp README won't run | CUDA errors on any GPU op | RTX 5060 is Blackwell (sm_120); torch 2.4.1+cu118 has no Blackwell kernels | torch 2.11 + cu128 in a fresh `ultradex` conda env |
| 2 | pytorch3d / BODex CUDA extensions fail to build | `CUDA version (13.3) mismatches ... PyTorch (12.8)`, then `cuda_runtime.h: No such file` | system nvcc is 13.3; conda toolkit headers live in `targets/x86_64-linux/` | install CUDA 12.8 toolkit **inside the env**; build with `CUDA_HOME=$CONDA_PREFIX`, `CPATH=$CONDA_PREFIX/targets/x86_64-linux/include` |
| 3 | BODex crashes at import of its collision world | `module 'warp' has no attribute 'torch'` | warp-lang 1.14 moved the torch interop to top level | patched `wp.torch.*` → `wp.*` in `geom/sdf/world_mesh.py` |
| 4 | coal C++ wrapper fails to compile | `use of deleted function coal::Contact::operator=` | conda-forge coal needs C++17; setup.py said `-std=c++11` | changed to `-std=c++17` |
| 5 | BODex refuses the Inspire URDF | `mimic joint can go out of it's upper limit...` | BODex validates mimic limits/velocities strictly (e.g. 0.6×1.334=0.8004 > 0.8) | widened mimic joint limits & velocities in our URDF copy |
| 6 | Isaac Lab scripts can't import `isaacsim` | `ModuleNotFoundError: isaacsim` under `isaaclab.sh -p` | this machine uses the **binary** Isaac Sim install; the conda env alone doesn't see it | always `conda activate env_isaaclab && source _isaac_sim/setup_conda_env.sh` |
| 7 | headless scripts "hang forever" after finishing | a validation run sat 4 h at 100 % CPU with all output stuck in buffers | `simulation_app.close()` can deadlock in headless mode | scripts save results first, then `os._exit(0)`; prints use `flush=True` |

### B. Making the grasp model match the simulator (details in §3)

| # | Problem | Symptom | Root cause | Fix |
|---|---|---|---|---|
| 8 | Every validated grasp failed (0/100), fingers wouldn't track | slave joints pegged at limits (2.04 rad); commanded flexion produced hyper-extension in a bare scene | the USD's slave joints have **no drives**; only a near-undamped `PhysxMimicJointAPI` with negative gearings moves them; `InspireMimicAction`'s software ratios are inert | measured the *actual* slave behavior inside the real env (`probe_env_hand.py`) and **froze the synthesis URDF's slave joints at those empirical postures** (thumb −0.16/−0.24 rad, fingers ≈1.15 rad) |
| 9a | fingers missed the cube by 1–3 cm even for geometrically perfect grasps | tip-to-surface gaps stayed 1–5 cm however hard we squeezed | the public dex-urdf Inspire model is a **different hand revision** than the USD (link origins match <1.5 mm, distal geometry doesn't) | **empirical grip re-centering**: close the sim hand to each grasp's joint config, read its real pinch center, move the cube-relative grasp position there (median 3.3 cm correction) |
| 9b | BODex poses are in the wrong frame for the sim | goals placed 90° off | URDF `base` vs USD `R_hand_base_link` differ by an axis permutation | Kabsch calibration over 10 link origins (0.3 mm residual), transform stored in the goal library |
| 10 | cube free-fell during validation before fingers closed | `moved ≈ 23 m` = exact free-fall distance | BODex assumes the object rests on a table; our validator dropped it mid-air, and fingers take ~0.3 s to close | **pin the cube** while the fingers close, release for the gravity-hold test |
| 11 | validation dynamics didn't match the training env | slaves blew to limits in the standalone scene but settled in the real env | writing joint **states** every step re-excites the underdamped mimic constraint | hold posture with **PD targets only**; never write states mid-episode |
| 12 | even then, mid-air hold was 0/100 | finger intermediates drift 0–1.3 rad between runs | the distal segments are effectively **stochastic** (see 8) — mid-air hold is a coin flip of constraint dynamics, not grasp quality | dropped mid-air hold as the criterion; filter is now geometric soundness + tray-compatible approach (§3.4) |

### C. Training-time problems

| # | Problem | Symptom | Root cause | Fix |
|---|---|---|---|---|
| 13 | first training run (5000 iters) never picked anything | `target_lifted = 0.000` for the whole run; 100 % timeouts; `task_reward` flat from iteration ~35 | hover local optimum: palm parked ~10 cm from goal, exactly at the finger-reward gate edge; heavy smoothness penalty discourages contact; full 10-cube clutter from step 1 (no curriculum in the active cfg) | widened gate 0.10→0.20 m, palm weight 0.5→1.0 (§4.2); longer training / staged clutter as follow-ups; BC-from-demos pipeline built as plan B (`collect_demos.py`) |
| 14 | **stale goals** — robot visibly parked where the cube *used to be* (user-spotted during play) | in some envs the palm hovers far from the red cube | goals were computed once at reset and frozen; any contact that slid the cube left the goal — and the shaping reward — pointing at empty space, *fighting* the task reward | goals made **live**: the chosen grasp is re-attached to the cube's current pose every step (`update_live_goals`, §4.1b) |
| 15 | goal selection could steer into clutter | shaping pulled the hand on a collision course through distractors, then penalties punished it | nearest-palm selection ignored distractors | **clutter-aware selection**: approach-corridor blocking count added to the selection score (§4.1) |
| 16 | training process silently died ~15 min in | no crash message; process gone; `oom_reaper: reaped process (python)` in kernel log | `--video --enable_cameras` on a 14 GB-RAM machine: offscreen rendering + frame buffers exhausted system memory | train **without video** on this machine (checkpoints every 50 iters — Ctrl+C, `play.py` the latest, `--resume`); or reduce to ≤768 envs with rare, short clips |
| 17 | **the hidden-goal ceiling** — even after fixes 14/15, the second run plateaued: palm reward pinned at 0.55–0.58 (~7 cm) for hundreds of iterations, finger reward decaying, zero lifts; when resumed to 5000 iters the policy *abandoned* the goal rewards (palm 0.58→0.38) while task reward rose (1.14→1.9+) — it stopped chasing the goal entirely | the goal **varies per episode but is not in the observation**: identical states earn different rewards depending on a variable the policy cannot see, so it can only optimize the average over goals → hover at the centroid of candidate poses. A structural ceiling that no weight/gate tuning can remove | **single fixed grasp** (`fixed_grasp_idx: 15`): with one goal, `goal = f(observed cube pose)` — deterministic and learnable. Long-term fix: goal-conditioned policy (goal appended to the observation), after which per-episode selection over the full library becomes sound again |

### D. Training-campaign log (chronological)

| Run | Setup | Outcome |
|---|---|---|
| `2026-07-06_18-14-06` (5000 it) | multi-grasp, nearest-palm, frozen goals, palm w=0.5 / gate 0.10 | 0 lifts; palm plateau 0.22 (≈9.5 cm) from iteration ~35; diagnosis → problems 13–14 |
| `2026-07-08_04-16-44` | + video recording | OOM-killed at ~iter 100 (problem 16) |
| `2026-07-08_04-52-57` (2000 it) | + live goals, clutter-aware selection, palm w=1.0 / gate 0.20 | 0 lifts; palm peaked 0.63 @ iter ~490 then decayed — hidden-goal ceiling (problem 17); task reward still rising at end |
| `2026-07-08_08-59-27` (resumed → 5000 it) | same | 0 lifts; policy traded goal rewards away for task reward (task 1.9+, clearly above baseline's 1.14 — closer cube engagement, but no grasp) |
| current | **single fixed grasp #15**, fresh, 3000 it | in progress — success criteria: palm > 0.7, finger reward rising, first nonzero `target_lifted` |

If the single-grasp run also fails to lift: next levers, in order — goal-conditioned
observation (96→109), then BC pretraining from scripted demonstrations
(`collect_demos.py`, already written) with PPO fine-tuning.

### The meta-lesson

Almost every problem came from one of two sources: (a) **silent divergence
between models of the same hand** (URDF vs USD, software mimic vs physics
mimic, synthesis frame vs sim frame) — countered by *measuring the simulator
instead of trusting documentation* (probe scripts, calibration, empirical
re-centering); and (b) **reward shaping pointing at something other than the
live simulation state** (frozen goals, gate edges) — countered by making every
quantity the reward touches recomputed from the current sim state each step.

---

## 7. Folder & file structure — what is where and how it is implemented

### 7.1 Annotated tree

```
g1_pick/
├── g1_pick_env_cfg.py               ← MODIFIED: +1 event term, +2 reward terms (§7.4)
├── mdp/
│   ├── __init__.py                  ← MODIFIED: re-exports grasp_goal module
│   └── grasp_goal.py                ← NEW: goal sampler + reward functions (§7.3)
└── grasp_sampler/                   ← NEW: everything below
    ├── README.md                    ← this file
    │
    │  ── the 3-stage offline pipeline ──
    ├── synthesize_inspire_grasps.py ← STAGE 1: BODex synthesis
    ├── validate_grasps_isaaclab.py  ← STAGE 2: calibration + grip re-centering
    ├── check_grasps_offline.py      ← helper: FK-based geometric scoring
    ├── build_goal_library.py        ← STAGE 3: filtering → goal library
    │
    │  ── one-off diagnostic scripts (kept for reference) ──
    ├── measure_coupling.py          ← slave-joint behavior in a bare scene
    ├── probe_env_hand.py            ← slave-joint behavior inside the real env
    │
    ├── grasp_dataset/               ← pipeline outputs
    │   ├── cube_5cm_grasps.npz            ← stage 1: 100 raw grasps
    │   ├── cube_5cm_grasps_recentered.npz ← stage 2: re-centered + calibration T
    │   ├── cube_5cm_grasps_valid.npz      ← stage 3: 32-grasp GOAL LIBRARY ★
    │   ├── geom_rank.npy / geom_ok.npy    ← offline scoring cache
    │
    └── ultradex_repo/               ← git clone of UltraDexGrasp
        ├── rollout.py               ← upstream demo entry point (unused — needs UR5e/SAPIEN)
        ├── util/bodex_util.py       ← MODIFIED: 'inspire' hand type branch
        ├── env/config/env.yaml      ← upstream env config (unused)
        ├── asset/object_mesh/
        │   ├── bowl/                ← upstream example object
        │   └── cube/                ← NEW: our 5 cm cube asset
        │       ├── mesh/simplified.obj    (unit cube, scaled ×0.05 at synthesis)
        │       ├── urdf/coacd.urdf        (collision decomposition stub)
        │       └── info/simplified.json   (center of mass, bounding box)
        └── third_party/
            ├── pytorch3d/           ← built from source (torch 2.11 / cu128 / sm_120)
            └── BODex_api/           ← the grasp optimizer
                └── src/bodex/
                    ├── geom/sdf/world_mesh.py       ← PATCHED: warp 1.14 API
                    ├── geom/cpp/setup.py            ← PATCHED: -std=c++17
                    └── content/
                        ├── assets/robot/inspire_hand/       ← NEW
                        │   ├── inspire_hand_right.urdf      ← heavily modified (§7.2)
                        │   └── meshes/{visual,collision}/   ← from dex-urdf
                        └── configs/
                            ├── robot/inspire_right.yml               ← NEW
                            ├── robot/hand_pose_transfer/inspire.yml  ← NEW
                            └── manip/sim_inspire_sim2real/fc_right.yml ← NEW
```

★ = the one file training actually reads.

### 7.2 The pipeline files, one by one

**`synthesize_inspire_grasps.py`** (stage 1, runs in the `ultradex` env)
Thin driver, ~50 lines. Adds `ultradex_repo` to `sys.path`, builds
`GraspSynthesizer(hand=1, hand_type='inspire', dof=6, num_grasp=100)`, calls
`synthesize_grasp(<cube asset dir>, [0,0,0,1,0,0,0], 0.05)` and saves the
returned `(100, 1, 3, 13)` array plus metadata (`joint_order`, `stages`) to
`grasp_dataset/cube_5cm_grasps.npz`. All the real work happens inside BODex,
driven by the three Inspire config files.

**`validate_grasps_isaaclab.py`** (stage 2, runs in `env_isaaclab`)
The largest script. Boots a headless Isaac Lab scene with **one environment per
grasp** (100 G1 robots + 100 cubes in a grid) and does, in order:

1. *Frame calibration* — writes an identical joint configuration into both the
   simulated USD hand and a pure-numpy URDF forward-kinematics model
   (`urdf_fk()` inside the file), collects 10 corresponding link-origin
   positions, and solves the rigid transform between the two hand base frames
   with the Kabsch/SVD algorithm (`solve_rigid_transform()`). Result:
   `T_usdbase_urdfbase`, residual ≈ 0.3 mm.
2. *Pass A: empirical grip centers* — parks the cube far away, PD-closes each
   hand to its grasp's joint configuration, computes the true fingertip points
   (distal body pose + fixed tip offset), and records the pinch center
   (thumb tip vs index/middle midpoint) per grasp.
3. *Pass B* — re-opens the fingers, pins the cube at the pinch center, closes
   pregrasp → grasp → deep squeeze with ramped targets, releases, and logs how
   far the cube moves (kept for diagnostics; not used as the filter, see §3.4).
4. Re-expresses every grasp's root pose relative to the re-centered cube and
   saves everything to `cube_5cm_grasps_recentered.npz`.

Two implementation details are load-bearing: the body is held by **PD targets
only** (writing joint *states* every step re-excites the underdamped PhysX
mimic constraint and ruins the hand), and the script ends with `os._exit(0)`
because `simulation_app.close()` can hang for hours in headless mode.

**`check_grasps_offline.py`** (helper, no simulator)
Contains the reusable pieces: a minimal URDF parser (`parse()`), forward
kinematics to the five fingertip frames (`fk_tips()`), and a signed
point-to-cube-surface distance (`cube_surface_dist()`). Run standalone it
scores all raw grasps and writes `geom_rank.npy` (used by the validator's
per-grasp diagnostics).

**`build_goal_library.py`** (stage 3)
Imports the helpers above, loads the *recentered* grasps, and keeps a grasp iff
(a) thumb + at least two fingers are within 2 cm of the cube surface in the
synthesis model and (b) nothing reaches below the cube's underside (tray
compatibility). Keeps the best 32 by contact quality and writes
`cube_5cm_grasps_valid.npz` — same array format, plus `T_usdbase_urdfbase`
carried through, plus `source_indices` for traceability.

**`measure_coupling.py` / `probe_env_hand.py`** (diagnostics)
The scripts that produced Discovery 1 (§3.1). `measure_coupling.py` sweeps the
proximal joints in a bare `InteractiveScene` and prints where the slave joints
settle; `probe_env_hand.py` does the same *through the real
`Isaac-G1-Pick-v0` env's action pipeline* (`gym.make` + `env.step`), which is
how the two regimes were shown to differ. Keep them — they're the first thing
to re-run after anyone edits the USD hand.

### 7.3 `mdp/grasp_goal.py` — the training-side integration

Three public symbols, consumed by the env config:

- **`sample_grasp_goal`** — a `ManagerTermBase` subclass used as a reset event.
  Its `__init__` runs once when the env is built: it loads the goal library,
  converts every grasp's root pose from the synthesis URDF frame into the USD
  `R_hand_base_link` frame using the stored calibration `T`, moves everything
  to the GPU, and allocates per-env goal buffers. Its `__call__(env, env_ids)`
  runs on every episode reset for the resetting envs: it transforms all 32
  candidate palm poses into world coordinates with the cube's new pose
  (batched quaternion math, no Python loops), picks the candidate nearest the
  robot's current palm, and writes `goal_pos_w / goal_quat_w / goal_hand_q`.
- **`grasp_goal_palm_reward`** — reads the goal buffers back (the term instance
  is found through the event manager and cached on the env as
  `env._grasp_goal_term`) and returns `1 − tanh(‖p_palm − p*‖ / 0.15)`.
- **`grasp_goal_hand_config_reward`** — same lookup; returns the palm-proximity
  gate times `1 − tanh(‖q − q*‖ / 0.5)` over the six proximal joints.

The module hard-codes the goal-library path (relative to the package) and the
hand joint names in policy order; both are constants at the top of the file.

### 7.4 Changes to existing files (all additive)

- **`g1_pick_env_cfg.py`** — two blocks:
  in `EventCfg`, `sample_grasp_goal = EventTerm(func=mdp.sample_grasp_goal,
  mode="reset", ...)` placed **after** `reset_target_object` (event terms run in
  declaration order, and the sampler must see the cube's new pose); in
  `RewardsCfg`, `grasp_goal_palm` (w=0.5) and `grasp_goal_hand` (w=0.3).
  Deleting these three blocks restores the exact previous behavior.
- **`mdp/__init__.py`** — one line: `from .grasp_goal import *`.
- **`ultradex_repo/util/bodex_util.py`** — an `elif hand_type == 'inspire':`
  branch setting `bodex_2_sim_q_idx = [0..5]` (identity, because the BODex
  cspace order was chosen to match the policy's action order).

### 7.5 The Inspire URDF: what was changed and why

`assets/robot/inspire_hand/inspire_hand_right.urdf` started as the public
dex-urdf model and went through three rounds of surgery (§3):

1. mimic-joint limits/velocities widened so BODex's strict parser accepts them;
2. mimic multipliers rewritten twice (hardware ratios, then USD gearing) while
   we chased the sim's true coupling;
3. **final state:** the six slave joints are `type="fixed"` with the measured
   sim postures baked into their origin rotations (thumb −0.16 / −0.24 rad,
   finger intermediates 1.15 rad). The hand BODex optimizes is therefore
   6-DoF, matching what the policy can actually control.

If the USD hand is ever fixed (drives added to slave joints), this URDF should
be regenerated with real mimic joints at the corrected ratios.

### Regenerating for a new object

1. Create an object directory under `ultradex_repo/asset/object_mesh/<name>/`
   (see how the cube one is built at the end of this README's history, or copy
   the `bowl` example layout).
2. Point `CUBE_ASSET` / `CUBE_SIZE` in `synthesize_inspire_grasps.py` at it and run it
   (`ultradex` env, `LD_LIBRARY_PATH=$CONDA_PREFIX/lib`).
3. Run `validate_grasps_isaaclab.py --grasp_file <new npz>` (env_isaaclab +
   `source _isaac_sim/setup_conda_env.sh`).
4. Run `build_goal_library.py` (adjust the cube-specific surface-distance
   function for the new shape).
5. Point `grasp_file` in `mdp/grasp_goal.py` (or the event's `params`) at the
   new library.

---

## 8. Known limitations & future work

1. **The USD hand should be fixed** (add drives to the slave joints or correct
   the mimic gearing signs). Until then the sim's distal finger segments are
   stochastic, and any sim-trained behavior involving fingertips transfers
   poorly to the real hand (real coupling: 0.8024/0.9487/1.0843).
2. **Model fidelity**: the ideal synthesis model would be a URDF extracted
   directly from `g1_with_hands_final.usd` (same revision, same geometry),
   eliminating the grip re-centering workaround.
3. **Single object**: the library covers only the 5 cm cube. The pipeline
   generalizes — DexGraspNet meshes can be fed through stages 1–3 for varied
   objects (the ClutterDexGrasp-style student stage will want this).
4. **Only one grasp is trained right now** (`fixed_grasp_idx: 15`, §4.1) — a
   deliberate simplification to defeat the hidden-goal ceiling (problem 17).
   The 32-grasp library and its selection machinery (reach cost + optional
   clutter-corridor scoring, per-grasp UCB bandit, critic-based
   $\arg\max_g V(o|g)$) are the multi-grasp upgrade path — valid only after (5).
5. **The policy does not observe its goal** — the goal acts only through the
   reward. This is THE structural limitation (it caused problem 17). Appending
   the goal (palm pose in robot frame + finger angles, ~13 dims) to the
   observation makes goal variation legal again: per-episode random selection
   over all 32 grasps then becomes the *best* training scheme (standard
   goal-conditioned RL). Requires an obs-dim migration (96→109;
   `tools/pad_checkpoint.py` workflow) and a fresh or padded checkpoint.
6. **Mid-air physical validation** can be revisited once (1) is fixed.
7. **If RL keeps stalling** (see §6, problem 13): the imitation route is
   prepared — `collect_demos.py` scripts an IK oracle through the env to
   record policy-compatible (obs, action) demonstrations for BC pretraining,
   followed by PPO fine-tuning from the BC checkpoint.
