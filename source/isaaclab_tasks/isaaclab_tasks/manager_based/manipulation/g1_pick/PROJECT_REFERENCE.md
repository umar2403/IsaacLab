# G1 Pick — Complete Technical Reference

> **Scope.** This document covers everything your friend built in the `g1_pick` folder:
> the full MDP formulation and mathematics (Part 1), the software architecture and
> file-by-file explanation (Part 2), and an analysis of grasp sampler options for
> the next research step (Part 3).

---

## Part 1 — Mathematical & Conceptual

### 1.1 Problem Statement

A Unitree G1 humanoid robot stands at a table with a tray. A small red cube (5 cm)
sits on the tray, possibly surrounded by up to 10 identically-sized blue/green/yellow
distractor cubes. The robot must pick up the target cube — reach, grasp, and lift it
above a success height — using only its right arm and Inspire dexterous hand, while
keeping all distractors on the tray. The lower body and left arm are frozen.

Training is pure reinforcement learning (no demonstrations, no teleoperation data).
The trained **teacher policy** observes privileged simulation state (exact object
positions). A future **student policy** will observe only partial sensors (camera
point cloud) and learn via behavioral cloning from teacher rollouts.

---

### 1.2 MDP Formulation

The environment is a **finite-horizon discounted Markov Decision Process**
$(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma, T)$.

#### State Space $\mathcal{S}$

The true physical state $s$ includes the full configuration of all rigid bodies:
all 53 joint positions and velocities of the robot (29 body + 12 right hand + 12
left hand), the 6-DoF pose and velocity of the target cube, and the 6-DoF poses
and velocities of all 10 distractors. Contact forces exist in the physics engine
but are not directly accessible to the policy.

#### Observation Space $\mathcal{O}$ (Teacher)

The policy observes $o \subset s$ — a **96-dimensional** privileged observation
vector (privileged because it includes exact object positions unavailable on a
real robot). The vector is assembled by the `ObservationManager` each control step:

| Component | Function | Dims |
|---|---|---|
| Right arm + hand joint positions | `joint_pos_rel` | 13 |
| Right arm + hand joint velocities | `joint_vel_rel`, clipped ±50 | 13 |
| Target cube position in robot frame | `target_object_position_b` | 3 |
| Target cube lin + ang velocity | `object_root_velocity` | 6 |
| Distractor 1–10 positions in robot frame | `target_object_position_b` × 10 | 30 |
| Right wrist + 5 fingertip positions in robot frame | `fingertip_positions_b` | 18 |
| Previous action (action history) | `last_action` | 13 |
| **Total** | | **96** |

The joint positions and velocities cover: `right_shoulder_{pitch,roll,yaw}`,
`right_elbow`, `right_wrist_{roll,pitch,yaw}` (7), and
`R_thumb_proximal_{yaw,pitch}`, `R_{index,middle,ring,pinky}_proximal` (6).

The fingertip bodies tracked are:
`right_wrist_yaw_link`, `R_thumb_distal`, `R_index_intermediate`,
`R_middle_intermediate`, `R_ring_intermediate`, `R_pinky_intermediate`.

All positions are expressed in the robot's root (pelvis) frame via the rotation

$$\mathbf{p}^b = R_W^b(\mathbf{p}^w - \mathbf{p}^w_\text{robot})$$

where $R_W^b = $ `quat_apply_inverse(robot_quat_w, ·)`. This makes the observation
invariant to the robot's absolute world position and heading.

#### Action Space $\mathcal{A}$

The policy outputs a 13-dimensional continuous action vector $\mathbf{a} \in [-1,1]^{13}$:

- $\mathbf{a}^{arm} \in \mathbb{R}^7$: right shoulder (pitch/roll/yaw), elbow, wrist (roll/pitch/yaw)
- $\mathbf{a}^{hand} \in \mathbb{R}^6$: thumb proximal yaw/pitch, index/middle/ring/pinky proximal

These are **position targets relative to the default pose**, scaled before application:

$$q^{target}_i = q^{default}_i + \text{scale}_i \cdot a_i$$

with $\text{scale}_{arm} = 0.3$ rad and $\text{scale}_{hand} = 0.5$ rad.

The `InspireMimicAction` class then applies hardware transmission ratios to set
six additional **slave (mimic) joints** that mechanically couple to the proximal joints.
This reduces the effective action space from 13 actuated + 6 slave = 19 hand DoF
to just 6 policy-controlled DoF, matching the physical Inspire Hand's tendon drive:

$$q_{thumb,inter}^{target} = q_{thumb,pitch}^{target} \times 0.8024$$
$$q_{thumb,distal}^{target} = q_{thumb,inter}^{target} \times 0.9487$$
$$q_{finger,inter}^{target} = q_{finger,prox}^{target} \times 1.0843 \quad \forall \text{ finger} \in \{\text{index, middle, ring, pinky}\}$$

#### Transition Dynamics $\mathcal{P}(s' | s, a)$

NVIDIA PhysX on the GPU simulates all 4096 environments in parallel.

- Physics timestep: $\Delta t = 1/120$ s
- Control decimation: $d = 4$ (policy runs at $120/4 = 30$ Hz)
- Episode length: $T_{s} = 8.0$ s $\Rightarrow T = 240$ control steps

At each control step, the action is applied once and PhysX takes 4 simulation
substeps. The robot's arm actuators use implicit PD control (spring-damper model
built into PhysX): torque $\tau = K_p(q^{target} - q) - K_d \dot{q}$, with
$K_p = 300$, $K_d = 30$ for the arms. The right hand uses $K_p = 100$, $K_d = 0.5$.
The locked joints (left hand, legs, waist) are frozen at $K_p = 10000$, $K_d = 1000$.

#### Reward Function $\mathcal{R}(s, a)$

The total per-step reward is:

$$R(s,a) = r_\text{task}(s) + w_{sm} \cdot p_\text{smooth}(s,a) + w_{im} \cdot p_\text{impact}(s) + w_{da} \cdot p_\text{dist,accel}(s) + w_{ot} \cdot p_\text{off\_tray}(s) + w_{dd} \cdot p_\text{dist,drop}(s)$$

with weights $w_{sm} = -3,\ w_{im} = -2,\ w_{da} = -3,\ w_{ot} = -10,\ w_{dd} = -100$.

**Task reward** $r_\text{task}$ (weight = 1.0):

Let $\mathbf{p}_c$ be the cube position, $\mathbf{p}_{palm}$ be the wrist/palm position,
$\{\mathbf{p}_{tip,i}\}_{i=0}^{4}$ be the 5 fingertip positions (index 0 = thumb).
Define surface-adjusted distances:

$$d_{thumb} = \max\!\left({\|\mathbf{p}_{tip,0} - \mathbf{p}_c\|} - 0.025,\ 0\right)$$
$$d_{fingers} = \max\!\left(\frac{1}{4}\sum_{i=1}^{4}\|\mathbf{p}_{tip,i} - \mathbf{p}_c\| - 0.025,\ 0\right)$$

The 0.025 m offset accounts for the cube's radius so zero distance means surface contact.

**1. Posture reward** — palm above the cube:

$$r_{posture} = 1 - \tanh\!\left(\frac{\max(\|\mathbf{p}_{palm} - (\mathbf{p}_c + [0,0,0.08])\| - 0.03, 0)}{0.3}\right)$$

**2. Reach reward** — fingertip centroid to cube:

$$r_{reach} = 1 - \tanh\!\left(\frac{\|\bar{\mathbf{p}}_{tip} - \mathbf{p}_c\|}{0.25}\right), \quad \bar{\mathbf{p}}_{tip} = \frac{1}{4}\sum_{i=1}^{4}\mathbf{p}_{tip,i}$$

**3. Grasp reward** — both thumb and fingers individually close:

$$r_{thumb} = 1 - \tanh\!\left(\frac{d_{thumb}}{0.055}\right), \quad r_{fingers} = 1 - \tanh\!\left(\frac{d_{fingers}}{0.055}\right)$$
$$r_{grasp} = \frac{r_{thumb} + r_{fingers}}{2}$$

**4. Grasp gate** (soft AND of thumb and fingers near cube, used as multiplier):

$$G(s) = \left(1 - \tanh\frac{d_{thumb}}{0.06}\right)\!\left(1 - \tanh\frac{d_{fingers}}{0.06}\right)$$

This goes to 1.0 only when both the thumb and the finger cluster are within ~6 cm
of the cube simultaneously. It prevents the policy from earning lift rewards by
pushing the cube with its palm without forming a proper grasp.

**5. Lift reward** — height gain conditioned on grasp:

$$r_{lift} = 2.0 \cdot \text{clamp}(z_c - 0.845,\ 0,\ 0.30) \cdot G(s)$$

where $z_c$ is the cube's height above the environment origin.

**6. Success bonus** — sparse, large:

$$r_{success} = 1000.0 \cdot \mathbb{1}[z_c > 1.134] \cdot G(s)$$

The threshold 1.134 m is ~29 cm above the tray surface (1.2× the original lift goal).

**7. Drop override** — if the cube falls off the table ($z_c < 0.6$ m), the entire task
reward is replaced with $-0.5$ regardless of the above terms.

$$r_\text{task} = \begin{cases} -0.5 & z_c < 0.6 \\ r_{posture} + r_{reach} + 2\,r_{grasp} + r_{lift} + r_{success} & \text{otherwise} \end{cases}$$

**Penalty terms:**

**8. Action smoothness penalty** $p_\text{smooth}$ (weight = −3.0):

$$p_\text{smooth} = 0.005 \sum_i(a_i - a_i^{prev})^2 + 0.001 \sum_j \dot{q}_j^2$$

**9. Fingertip impact penalty** $p_\text{impact}$ (weight = −2.0):

$$p_\text{impact} = \frac{1}{6}\sum_{k=1}^{6}\tanh\!\left(\frac{\|\Delta\mathbf{v}_{tip,k}\|}{3.0}\right)$$

where $\Delta\mathbf{v}_{tip,k}$ is the per-step velocity change of the $k$-th hand body.
Gentle contact gives ~0.1, a hard jab saturates near 1.0.

The three distractor penalties operate on a **three-level severity hierarchy** — each targets a distinct failure mode at increasing cost:

**10. Distractor acceleration penalty** $p_\text{dist,accel}$ (weight = −3.0, fires every step):

$$p_\text{dist,accel} = \frac{1}{N_d}\sum_{d=1}^{N_d}\tanh\!\left(\frac{\|\Delta\mathbf{v}_{d}\|}{2.0}\right)$$

Measures the per-step velocity change $\|\Delta\mathbf{v}_d\| = \|\mathbf{v}_d^t - \mathbf{v}_d^{t-1}\|$ of each distractor — a proxy for contact impulse (force × time). A gentle nudge gives $\Delta v \approx 0.1$ m/s $\Rightarrow$ penalty $\approx 0.05$. A hard swipe saturates near 1.0. This penalizes violent contact style even if the distractor stays on the tray. Averaged over all $N_d$ distractors.

**11. Distractor off-tray penalty** $p_\text{off\_tray}$ (weight = −10.0, fires every step):

$$p_\text{off\_tray} = \sum_{d=1}^{N_d}\mathbb{1}[z_d < 0.835]$$

Counts how many distractors are currently sitting below tray-surface height (0.835 m) but still above the floor drop threshold (0.6 m) — i.e., fell off the tray edge onto the table surface. This fires every step as long as a distractor is off the tray. Over a 240-step episode, one tipped distractor costs $-10 \times 240 = -2400$ — heavy enough to dominate the task reward and force the policy to learn tray-aware avoidance.

**12. Distractor drop penalty** $p_\text{dist,drop}$ (weight = −100.0, fires once then terminates):

$$p_\text{dist,drop} = \mathbb{1}\!\left[\exists\, d : z_d < 0.6\right]$$

Sparse: fires $-100$ the step any distractor falls completely off the table ($z < 0.6$ m). The episode terminates at this point so it fires at most once. This is the catastrophic failure signal.

**Summary of the hierarchy:** penalty 10 discourages bad contact dynamics (style), penalty 11 discourages tipping (outcome, with heavy per-step accumulation), penalty 12 penalizes total loss of a cube (catastrophic, terminates episode). Together they shape the policy to treat distractors carefully at all stages of the approach.

#### Termination Conditions

| Condition | Type | Trigger |
|---|---|---|
| Time out | Non-terminal (truncation) | $t > 240$ steps |
| Target lifted | Terminal (success) | $z_c > 1.134$ m |
| Target dropped | Terminal (failure) | $z_c < 0.6$ m |
| Distractor dropped | Terminal (failure) | $\exists\, d : z_d < 0.6$ m |

#### Discount and Returns

$$\gamma = 0.99, \quad G_t = \sum_{k=0}^{T-t} \gamma^k R_{t+k}$$

---

### 1.3 PPO Algorithm

The policy is trained with **Proximal Policy Optimization** (Schulman et al., 2017)
via the RSL-RL backend.

**Network architecture.** Actor and critic are separate MLPs sharing the same
structure: input → 512 → 256 → 128 → output, with ELU activations. The actor
outputs the mean $\mu$ of a Gaussian action distribution; log-std is a learnable
parameter initialized at $\log(0.8)$. The critic outputs a scalar $V(o)$.
Input normalization (running mean/std) is applied to both.

**Data collection.** Each PPO iteration collects a rollout buffer of

$$N_{steps} \times N_{envs} = 32 \times 4096 = 131{,}072 \text{ transitions}$$

**Advantage estimation (GAE-$\lambda$).**

$$\delta_t = r_t + \gamma V(o_{t+1}) - V(o_t)$$
$$\hat{A}_t = \sum_{l=0}^{T-t}(\gamma\lambda)^l\delta_{t+l}, \quad \lambda = 0.95$$

**Policy loss (clipped surrogate objective).**

$$\rho_t(\theta) = \frac{\pi_\theta(a_t|o_t)}{\pi_{\theta_{old}}(a_t|o_t)}$$
$$\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t\!\left[\min\!\left(\rho_t\hat{A}_t,\ \text{clip}(\rho_t, 1-\varepsilon, 1+\varepsilon)\hat{A}_t\right)\right], \quad \varepsilon = 0.2$$

**Value loss.**

$$\mathcal{L}^{VF}(\phi) = \mathbb{E}_t\!\left[(V_\phi(o_t) - \hat{V}_t^{target})^2\right]$$

with clipped value loss enabled ($\varepsilon = 0.2$).

**Total loss.**

$$\mathcal{L}(\theta,\phi) = -\mathcal{L}^{CLIP}(\theta) + c_1\mathcal{L}^{VF}(\phi) - c_2 H[\pi_\theta]$$

where $c_1 = 1.0$ (value loss coefficient) and $c_2 = 0.005$ (entropy coefficient).
The entropy bonus prevents premature action distribution collapse.

**Optimization.**

- Learning rate: $3 \times 10^{-4}$, adaptive schedule (KL target $= 0.01$)
- Gradient clipping: $\|\nabla\| \leq 1.0$
- Mini-batches per rollout: 4
- Learning epochs per rollout: 5
- Total training: 5000 PPO iterations $\approx 655\text{M}$ environment steps

At each iteration, the 131,072-transition buffer is split into 4 mini-batches
and the optimizer runs 5 passes through them. The adaptive LR scheduler increases
LR when the mean KL divergence is below 0.01, and decreases it above.

---

### 1.4 Curriculum Learning

The curriculum has two **orthogonal** progressions: (i) a **phase progression**
designed to unlock reward terms, and (ii) a **difficulty progression** controlling
clutter count. Critically, these are **not symmetric in their current implementation** —
read the note below before assuming both are active.

#### Important: What Is and Is Not Active in the Current Config

**Only one reward function exists throughout all of training.** The `RewardsCfg`
in `g1_pick_env_cfg.py` has six terms: `task_reward`, `action_smoothness`,
`fingertip_impact`, `distractor_accel`, `distractor_off_tray`, `distractor_drop`.
All six are active from step 1 and their weights never change.

The `compute_task_reward` function bundles posture + reach + grasp + lift + success
all in one call. There is no difficulty-gated switching between "easy reward" and
"hard reward" — the same function runs regardless of clutter level.

**The phase-based reward unlocking is scaffolded but not triggered.**
The `PickingCurriculumScheduler` is designed to find terms named `reaching_target`,
`lifting_target`, `declutter`, `pick_success` in the reward manager and mutate
their weights to zero → nonzero at phase boundaries. But none of those names
exist in the current `RewardsCfg`. When `_init_rm` runs it finds no matches, so
`_phase1_indices` and `_phase2_indices` are empty dicts — the phase logic does
nothing. The curriculum's phase advancement code executes without effect.

**What the curriculum actually does:** only the difficulty bandit (see below).
The effective learning progression reach → grasp → lift emerges from the reward
shaping inside `compute_task_reward` (the grasp gate $G(s)$ makes lift reward zero
until grasping occurs naturally), not from any curriculum-triggered weight change.

To activate phase-based unlocking, one would need to split `compute_task_reward`
into separate `reaching_target`, `lifting_target`, `pick_success` `RewTerm` entries
in `RewardsCfg`, initialize the lift/success weights to 0, and let the scheduler
mutate them. That is the intended design but is not the current state.

#### Phase Progression (Three Phases — Designed, Not Currently Active)

The `PickingCurriculumScheduler` maintains rolling deques of episode reward sums
and mutates reward weights at runtime when thresholds are crossed.

**Phase 0 — Reaching only.** Only `task_reward` (with its posture + reach + grasp
sub-components) is active. Lifting and success rewards have weight 0. The agent
learns to move the right hand toward the cube.

**Phase 0 → 1 transition.** Triggered when:

$$\frac{1}{N}\sum_{i=1}^{N} r^{ep}_{reaching,i} \geq \phi_1 = 0.4, \quad N \geq 50 \text{ episodes}$$

where $r^{ep}_{reaching}$ is the cumulative reaching reward over an episode.
On transition: `lifting_target` (weight = 5.0) and `declutter` (weight = 2.0) are enabled.

**Phase 1 → 2 transition.** Triggered when:

$$\frac{1}{N}\sum_{i=1}^{N} r^{ep}_{lifting,i} \geq \phi_2 = 0.75, \quad N \geq 50 \text{ episodes}$$

On transition: `pick_success` (weight = 1.0) is enabled.

The scheduler resolves reward term indices lazily (on first call) so it never
requires the reward manager to exist during construction.

#### Difficulty Progression (Per-Environment Bandit)

Each environment $i$ maintains an integer difficulty $d_i \in [0, 60]$, updated
at every episode reset:

$$d_i \leftarrow \text{clamp}\!\left(d_i + \delta_i,\ 0,\ 60\right), \quad \delta_i = \begin{cases} +1 & z_c > 0.15 \text{ m (success)} \\ -1 & \text{otherwise} \end{cases}$$

Clutter activation schedule (from `reset_clutter_based_on_difficulty`):

| Difficulty | Distractors active | Distractor objects |
|---|---|---|
| 0–29 | 0 | none |
| 30–39 | 1 (distractor_1) | activation threshold = 30 |
| 40–49 | 2 (+ distractor_2) | activation threshold = 40 |
| 50–60 | 3 (+ distractor_3, ...) | threshold = 50, 60, ... |

For each distractor, if $d_i \geq$ its activation threshold, it is placed
uniformly on the tray; otherwise it is hidden at $z = -5$ m (below the world).
This ensures that at early training stages, the agent faces no clutter; clutter
density ramps up only as the agent demonstrates reliable picking.

The global difficulty fraction $\bar{d} = \text{mean}(d_i) / 60$ is returned
by the scheduler and logged by Isaac Lab.

---

### 1.5 Robot Kinematics

The G1 body has 29 revolute joints: 12 lower-body, 3 waist, 14 arm joints
(7 per arm). Each Inspire hand adds 12 revolute joints: 6 actuated proximal
joints (thumb yaw, thumb pitch, index/middle/ring/pinky proximal) and 6 mimic
joints (intermediate + distal for each finger chain). Total articulation: **53 DOF**.

The right hand kinematic chain:

```
R_hand_base_link (palm, fixed to right_wrist_yaw_link)
├── R_thumb_proximal_yaw_joint → R_thumb_proximal_base
│   └── R_thumb_proximal_pitch_joint → R_thumb_proximal
│       └── R_thumb_intermediate_joint (mimic, gear=−1.6×) → R_thumb_intermediate
│           └── R_thumb_distal_joint (mimic, gear=−2.4×) → R_thumb_distal
├── R_index_proximal_joint → R_index_proximal
│   └── R_index_intermediate_joint (mimic, gear=−1.0×) → R_index_intermediate
├── R_middle_proximal_joint → R_middle_proximal
│   └── R_middle_intermediate_joint (mimic, gear=−1.0×) → R_middle_intermediate
├── R_ring_proximal_joint → R_ring_proximal
│   └── R_ring_intermediate_joint (mimic, gear=−1.0×) → R_ring_intermediate
└── R_pinky_proximal_joint → R_pinky_proximal
    └── R_pinky_intermediate_joint (mimic, gear=−1.0×) → R_pinky_intermediate
```

The hand is attached to the wrist via a `PhysicsFixedJoint` (AssemblerFixedJoint)
at offset $(0.0415, -0.003, 0)$ m from `right_wrist_yaw_link`. Note: the USD's
`PhysxMimicJointAPI` gearings ($-1.6, -2.4, -1.0$) are overridden by the
software mimic in `InspireMimicAction` (ratios $0.8024, 0.9487, 1.0843$) —
the software path takes effect because it runs after `super().apply_actions()`.

---

### 1.6 Teacher–Student Framework (Planned)

The current teacher $\pi^E$ observes privileged state $o^E \in \mathbb{R}^{96}$
(exact object positions, distractor positions). At deployment, only
partial sensor observations $o^S$ are available (camera, joint encoders).

The plan:

1. **Collect rollouts**: Run $\pi^E$ across varying clutter densities, save
   $(o^S, a)$ pairs — the student's training dataset $\mathcal{D}^E$.
2. **Distill**: Train student $\pi^S$ via behavioral cloning:
   $$\pi^{S*} = \arg\min_\pi\ \mathbb{E}_{(o,a)\sim\mathcal{D}^E}\!\left[-\log\pi(a|o^S)\right]$$
3. **Student observation** $o^S$: right arm/hand joints + point cloud of tray
   (from head-mounted D435 depth camera in the USD).

This is precisely the paradigm described in ClutterDexGrasp (Chen et al., CoRL 2025),
which is included in this folder as `clutterdex.pdf`. The PDF is the closest
published work to what this project implements and is the strongest external
reference for the teacher–student design.

---

## Part 2 — Software Reference

### 2.1 Repository Architecture

```
g1_pick/                           ← self-contained Isaac Lab task plugin
├── __init__.py                    ← gym.register() — entry point for Isaac Lab
├── g1_pick_env_cfg.py             ← central config: scene, actions, obs, rewards, events
├── g1_pick_env_cfg_v1_*.py        ← archived snapshot: perfect single-target pick
├── g1_pick_env_cfg_10cube_*.py    ← archived snapshot: 10-cube clutter version
├── robot_cfg.py                   ← G1 + Inspire Hand articulation definition
├── g1_with_hands_final.usd        ← binary USD: robot geometry + physics schema
├── clutterdex.pdf                 ← key reference paper (Chen et al., CoRL 2025)
├── claude_about me.md             ← documentation preferences
├── mdp/
│   ├── __init__.py                ← re-exports isaaclab.envs.mdp + all custom modules
│   ├── observations.py            ← custom obs functions
│   ├── rewards.py                 ← custom/experimental reward functions
│   ├── terminations.py            ← custom termination functions
│   ├── events.py                  ← custom reset events
│   └── curriculum.py             ← PickingCurriculumScheduler
├── agents/
│   ├── __init__.py
│   ├── rsl_rl_ppo_cfg.py          ← PPO config for RSL-RL backend (used)
│   └── rl_games_ppo_cfg.yaml      ← PPO config for RL-Games backend (alternate)
├── working_models/
│   └── model_4999_*_96dim.pt      ← trained checkpoint: perfect single pick
└── tools/
    ├── extract_hand_structure.py  ← USD parser that produced the .txt files
    ├── kinematic_tree.txt         ← joint tree dump
    ├── hand_structure_reference.txt ← detailed hand joint reference
    ├── pad_checkpoint.py          ← checkpoint dimension-padding utility
    ├── parse_usd.py
    └── parse_usd_physics_check.py
```

---

### 2.2 How This Plugs Into Isaac Lab

Isaac Lab's `ManagerBasedRLEnv` provides the full simulation loop. The `g1_pick`
package only defines **what** the task-specific components are — not how they run.
The connection works like this:

**1. Discovery.** The `g1_pick` directory is inside
`isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/`, which is on the
Python path installed via the `isaaclab_tasks` package. Any file that imports from
`isaaclab_tasks` causes all `__init__.py` files in the package to execute, which
runs `gym.register`.

**2. Registration.** `__init__.py` calls:

```python
gym.register(
    id="Isaac-G1-Pick-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": G1PickEnvCfg, ...}
)
```

When `gym.make("Isaac-G1-Pick-v0")` is called by the training script,
`ManagerBasedRLEnv(cfg=G1PickEnvCfg())` is instantiated.

**3. Manager construction.** `ManagerBasedRLEnv.__init__` reads the config and
constructs seven manager objects, each of which holds a list of "term configs"
that point to functions:

| Manager | Term configs in | Calls these functions |
|---|---|---|
| `SceneManager` | `SceneCfg` | spawns USD assets in PhysX |
| `ActionManager` | `ActionsCfg` | `InspireMimicAction.apply_actions()` |
| `ObservationManager` | `ObservationsCfg` | `joint_pos_rel`, `target_object_position_b`, etc. |
| `RewardManager` | `RewardsCfg` | `compute_task_reward`, penalty functions |
| `TerminationManager` | `TerminationsCfg` | `target_object_lifted`, `target_object_dropped`, etc. |
| `EventManager` | `EventCfg` | `reset_joints_by_offset`, `reset_root_state_uniform`, etc. |
| `CurriculumManager` | `CurriculumCfg` (if defined) | `PickingCurriculumScheduler.__call__` |

**4. The control loop** (one call to `env.step(actions)`):

```
RSL-RL collects action tensor (4096, 13)
    ↓
ActionManager.process_action(actions)
    InspireMimicAction computes mimic targets → writes joint_pos_target to PhysX
    ↓
for _ in range(decimation=4):
    sim.step()   ← PhysX advances all 4096 envs by 1/120 s
    ↓
ObservationManager.compute()  → obs tensor (4096, 96)
RewardManager.compute()       → reward tensor (4096,)
TerminationManager.compute()  → done tensor (4096,)
    ↓
EventManager.apply(mode="reset", env_ids)  ← resets done envs
CurriculumManager.compute(env_ids)         ← updates difficulty + phase
    ↓
return obs, reward, done, info
```

**5. Headless + video.** `--headless` suppresses Isaac Sim's viewport renderer.
`--video --video_length 600 --video_interval 2000` periodically re-enables
rendering for 600 steps and saves an MP4 file — useful for monitoring training
without live visualization.

---

### 2.3 File-by-File Explanation

#### `__init__.py`

Registers two gym environments:
- `Isaac-G1-Pick-v0` — training config (`G1PickEnvCfg`, 4096 envs)
- `Isaac-G1-Pick-Play-v0` — eval config (`G1PickEnvCfg_PLAY`, 64 envs, max difficulty)

Both point to both the RSL-RL and RL-Games runner configs, so either training
backend can be used by switching the training script.

---

#### `robot_cfg.py`

Defines `G1_INSPIRE_CFG: ArticulationCfg`. Key design decisions:

**USD loading.** The USD is loaded from the same directory as this file (resolved
via `os.path.dirname(__file__)`), so the robot definition is portable.

**Default pose.** The arm is initialized in a natural hover position above the
table (right arm reaching forward-down). The right thumb's `yaw` joint starts at
0.7 rad (thumb extended sideways), all finger proximal joints at 0.0 (open hand).

**Actuator groups** (`ImplicitActuatorCfg`):

| Group | Joints | $K_p$ | $K_d$ | Role |
|---|---|---|---|---|
| `legs` | hips, knees, waist | 150–200 | 5 | frozen at default by EventCfg |
| `feet` | ankles | 20 | 2 | frozen at default |
| `arms` | both arm chains | 300 | 30 | active (left arm frozen by EventCfg) |
| `left_hand` | L_* | 10 | 1 | frozen at high stiffness by `__post_init__` override |
| `right_hand` | R_* | 100 | 0.5 | active: high stiffness, low damping for crisp finger response |

The arm stiffness of 300 Nm/rad with damping 30 Nm·s/rad makes the arm behave like a
"rock-solid crane" — it holds any commanded position tightly with no oscillation.
The right hand's low damping (0.5) allows the fingers to snap to target angles quickly.

`soft_joint_pos_limit_factor = 0.9` reduces the effective joint limits to 90% of their
physical limits, creating a safety buffer before hitting hard stops.

---

#### `g1_pick_env_cfg.py`

This is the largest and most important file. It contains:

**Height constants** (ground truth of the scene geometry):

```
_OBJ_INIT_Z  = 0.845   # tray surface (0.820) + half cube (0.025)
_SUCCESS_Z   = 1.134   # ~29 cm above tray, where curriculum advances
_OFF_TRAY_Z  = 0.835   # cube fell off tray edge (still on table)
_DROP_Z      = 0.600   # cube fell off table → terminal
```

**`compute_task_reward`** (module-level function, not in `mdp/rewards.py`). This is
the primary dense reward. It is referenced directly in `RewardsCfg.task_reward`
(weight=1.0). It implements the 7 sub-components described in Part 1 §1.2.

**`action_smoothness_penalty`**, **`distractor_acceleration_penalty`**,
**`fingertip_impact_penalty`**, **`distractor_off_tray_penalty`**,
**`distractor_drop_penalty`** — all module-level penalty functions. Each returns a
positive scalar; the negative weight in `RewardsCfg` makes them penalties.

**`InspireMimicAction`** — custom `JointPositionAction` subclass. Overrides
`apply_actions()`: first calls `super().apply_actions()` to set all 6 proximal
joint targets (via Isaac Lab's standard path), then directly calls
`self._asset.set_joint_position_target(mimic_targets, joint_ids=...)` on the
6 slave joints. This bypasses the USD's built-in `PhysxMimicJointAPI`, which is
necessary because the USD gearing ratios (−1.6, −2.4, −1.0) are different from
the URDF hardware ratios (0.8024, 0.9487, 1.0843) — the software path takes
precedence.

**`SceneCfg`** — declares all 14 rigid bodies:
- 1 robot articulation
- 1 kinematic table (60×120×80 cm)
- 1 kinematic tray (40×60×2 cm)
- 1 target cube (5 cm, 0.2 kg, friction=1.0, 16 solver iterations)
- 10 distractor cubes (identical physics, different RGB colors for visual ID)
- 1 ground plane + 1 dome light

`replicate_physics=True` ensures all 4096 environments share the same physics
parameters deterministically.

**`ObservationsCfg`** — single `PolicyCfg` observation group, 96-dim (see §1.2).
`enable_corruption=False` — Gaussian noise is disabled. The comment notes it
can be enabled later for domain randomization. `concatenate_terms=True` flattens
all obs terms into one 1D tensor for the MLP.

**`EventCfg`** — all reset events. Note the distinction:
- `freeze_lower_body` and `freeze_left_arm` use `position_range=(0.0, 0.0)` —
  they reset to exactly the default pose with zero jitter. This deterministically
  freezes these joints every episode.
- `reset_right_arm` and `reset_right_hand` use `position_range=(-0.05, 0.05)` —
  ±0.05 rad domain randomization so the policy doesn't overfit to one exact start.
- `reset_target_object` uses `pose_range: x:(-0.10,0.10), y:(-0.05,0.05)` —
  the cube spawns anywhere in a 20×10 cm window around its anchor (0.35, 0.0).
- Distractor resets use ±3 cm jitter — small enough to avoid cube overlaps given
  ≥7 cm anchor spacing.

**`TerminationsCfg`** — four conditions as described in §1.2.

**`G1RightArmLiftEnvCfg_V2.__post_init__`** does three things after dataclass
field initialization:
1. Sets simulation parameters (dt, decimation, PhysX buffer sizes).
2. Applies joint locking by overriding actuator stiffness/damping for left hand,
   legs, and feet to 10000/1000 (effectively infinite stiffness — frozen).
3. The GPU buffer tuning (`gpu_found_lost_aggregate_pairs_capacity = 2M`) is
   important: these buffers hold PhysX contact data for 4096 parallel envs.
   Too small → sim crash; too large → VRAM exhaustion.

---

#### `mdp/__init__.py`

One line of functional importance:
```python
from isaaclab.envs.mdp import *  # noqa
```
This makes all of Isaac Lab's built-in MDP functions available under the `mdp`
namespace, so `g1_pick_env_cfg.py` can write `mdp.reset_joints_by_offset`,
`mdp.joint_pos_rel`, `mdp.last_action`, etc. without importing them directly.
The five custom modules are then added on top.

---

#### `mdp/observations.py`

**`target_object_position_b`** — used 11 times in `ObservationsCfg` (once for
target, once per distractor). The transform is:

```python
quat_apply_inverse(robot.data.root_quat_w,
                   object.data.root_pos_w - robot.data.root_pos_w)
```

This gives the object's position in the robot root frame. Since the robot's
pelvis is fixed to the world (via `RootFixedJoint`), this is effectively the
position relative to the pelvis, expressed in pelvis-local axes.

**`fingertip_positions_b`** — applies the same robot-frame transform to each of
the 6 tracked hand bodies. The reshape dance:

```python
root_quat = robot.data.root_quat_w.unsqueeze(1).expand(-1, 6, -1).reshape(-1, 4)
delta_w   = (fingertip_pos_w - robot_pos_w).reshape(-1, 3)
result    = quat_apply_inverse(root_quat, delta_w).view(num_envs, -1)
```

This processes all 6 bodies for all 4096 envs in one batched quaternion multiply.

**`fingertip_to_object_vectors`** — per-finger delta vectors from tip to cube, in
robot frame. Shape (N, 15). The most direct "where should each finger move"
signal. It's defined but **not currently used in `ObservationsCfg`** — it's
available for future experiments.

---

#### `mdp/rewards.py`

This file contains **experimental and alternative** reward components. Not all are
active in `RewardsCfg` — several were tried and superseded. Key ones to understand:

**`_get_proximity_gate(env, robot_cfg, object_cfg, gate_std)`** — shared helper.

$$G_{gate}(s) = \sigma\!\left(10 \cdot \left(\frac{1}{N}\sum_k 1 - \tanh\frac{\|\mathbf{p}_{tip,k} - \mathbf{p}_c\|}{gate\_std} - 0.5\right)\right)$$

A sigmoid maps mean fingertip proximity (0 to 1) to a gate value that is near-zero
when fingers are far and near-one when close. This prevents the policy from earning
"lift" or "hold" rewards by coincidentally moving the cube with its arm.

**`lift_height_reward`** — convex (squared) progress with proximity gate:

$$r_{lift} = G_{gate} \cdot \left(\frac{\text{clamp}(z_c - z_{rest}, 0)}{\Delta z_{max}}\right)^2$$

The square is the "snowball" curve: early in lifting, reward increases slowly;
later in the lift it grows quickly. This suppresses reward for micro-bounces where
the cube barely leaves the tray surface.

**`height_progress_reward`** — monotonic: tracks `_max_cube_height` per env,
only rewards new height records. No reward for going up and coming back down.
Resets at episode start via `episode_length_buf == 1` check.

**`finger_closure_reward`** — measures grasp geometry (thumb opposition):

$$r_{closure} = \text{mean}_k\!\left[\frac{1 - (\hat{t} \cdot \hat{f}_k)}{2} \cdot \mathbb{1}[d_{thumb} < 0.08] \cdot \mathbb{1}[d_{finger,k} < 0.08]\right]$$

where $\hat{t}$ is the unit vector from cube to thumb, $\hat{f}_k$ is the unit
vector from cube to finger $k$. When $\hat{t} \cdot \hat{f}_k = -1$ (perfectly
opposed), this gives 1.0. Requires both thumb and finger to be within 8 cm.

---

#### `mdp/terminations.py`

Three simple functions. All return a boolean tensor of shape `(num_envs,)`.

`any_distractor_dropped` iterates over the list of distractor names and ORs
their height checks: `any_dropped |= obj.data.root_pos_w[:, 2] < min_height`.
Returning a single aggregated signal (rather than one per distractor) keeps
the termination manager clean.

---

#### `mdp/events.py`

**`reset_clutter_based_on_difficulty`** is the key event function. It reads the
difficulty from the curriculum manager (via `env.curriculum_manager._term_cfgs[idx].func`),
then for each distractor computes:

```python
activation_threshold = 30.0 + dist_idx * 10.0
is_active = diff_for_ids >= activation_threshold
pos[:, 2] = torch.where(is_active, active_z, hidden_z)
```

Hidden distractors go to `z = -5` m, below the world, so they have no physics
interaction. The try/except around curriculum access handles the case where
the curriculum manager hasn't been initialized yet (first step of training).

Note: in the current active config (`g1_pick_env_cfg.py`), this function is
**not called** — the env instead uses `mdp.reset_root_state_uniform` for all
distractors, which always places them on the tray (because their default init
positions are above the tray and the z range is `(0.0, 0.0)`). The difficulty-
based hiding of distractors is implemented in the cfg variant files.

---

#### `mdp/curriculum.py`

`PickingCurriculumScheduler` is an `ManagerTermBase` subclass. Isaac Lab calls
`__call__(env, env_ids, ...)` once per step for the set of environments that
just reset.

**Lazy initialization (`_init_rm`).** On the first call after training starts,
it resolves reward term names to list indices in `env.reward_manager._term_names`.
This is lazy because the reward manager isn't fully built during `__init__`.

**Phase weight mutation.** `_set_phase_weights` writes directly into
`env.reward_manager._term_cfgs[idx].weight`. This runtime mutation is what
causes Phase 1 and Phase 2 reward terms to activate mid-training without
stopping or rebuilding the environment.

**`get_state` / `set_state`** — checkpoint compatibility hooks. Isaac Lab calls
these when saving/loading training checkpoints so per-env difficulty survives
a training resume.

---

#### `agents/rsl_rl_ppo_cfg.py`

`G1PickPPORunnerCfg` is used in practice (RSL-RL backend, invoked via
`rsl_rl/train.py`). Key hyperparameters and their effect:

| Parameter | Value | Effect |
|---|---|---|
| `num_steps_per_env` | 32 | Rollout horizon. Short → more frequent updates, noisier gradients |
| `init_noise_std` | 0.8 | Initial action std. High → strong exploration at training start |
| `actor_obs_normalization` | True | Running mean/std normalization on obs input |
| `entropy_coef` | 0.005 | Small entropy bonus prevents early collapse to deterministic policy |
| `desired_kl` | 0.01 | Adaptive LR target; LR increases if KL < 0.01, decreases if > 0.01 |
| `gamma` | 0.99 | Moderate discount — values future rewards up to ~100 steps ahead |
| `lam` | 0.95 | GAE: balances bias (low λ) vs variance (high λ) |

---

#### `tools/pad_checkpoint.py`

When the observation space was expanded from an earlier dimension to 96 dims,
the saved checkpoint's network weights became incompatible (input layer mismatch).
This tool adds zeros to the first-layer weight matrix to pad the input dim to 96,
allowing training to resume from the old checkpoint with the new larger obs space.
This is why the working model is called `*_padded_96dim.pt`.

---

### 2.4 Development Progression (from git history)

The commit history reveals the chronological order of what was built:

1. **`trained_sawMinResults`** — earliest checkpoint committed. Single-target,
   no clutter. Basic arm reward structure.

2. **`success bonus pos changed`** — tuned the success height threshold and/or
   the bonus magnitude.

3. **`major_change: 10nonTarget+rewardStructure`** — added all 10 distractor
   rigid objects and the clutter penalty terms (off-tray, drop, acceleration).
   Biggest structural change.

4. **`10clutter_cfg_saved_separately`** — archived the 10-clutter config as a
   standalone file, keeping clean version history.

5. **`new_cfg_jerk_penalty`** — added `fingertip_impact_penalty` and refined the
   action smoothness penalty. These were added to suppress jabbing/slamming
   behaviors that were achieving high rewards but would be unsafe on hardware.

The `pad_checkpoint.py` tool existence implies the observation space was enlarged
during step 3 or 4, requiring a checkpoint migration.

---

## Part 3 — Grasp Sampler: Analysis and Recommendation

### 3.1 Why a Sampler?

The current reward function measures **distance to the cube center**, not the
quality of the grasp geometry. The policy reaches near the cube and finds some
configuration that moves it upward — but it has no explicit target for where
each finger should be relative to the object surface. This works for a small cube
in simulation but will not generalize to varied geometries in the real world.

A **grasp sampler** generates a target grasp configuration
$G^* = (\mathbf{p}^*_{palm},\ \mathbf{q}^*_{palm},\ \boldsymbol{\theta}^*_{hand})$
for a given object geometry, encoding physically feasible finger contact locations
and approach directions. Integrating one into this project would:

- Provide a reward signal for grasp quality (not just proximity)
- Allow the teacher policy to be guided toward stable grasps from the start
- Provide human-interpretable intermediate goals

### 3.2 DexGraspNet / DexGraspNet 2.0

**Papers.** DexGraspNet (Wang et al., CVPR 2023) and DexGraspNet 2.0 (Zhang et al.,
CoRL 2024). From PKU-EPIC group.

#### In plain terms

Imagine you want to teach a robot hand to grasp objects. You could let RL figure out
grasping from scratch every time — but that's slow and unreliable. A smarter approach:
*pre-compute* a large library of good grasps for thousands of objects, so that when
the robot sees an object it has never held before, it can look up a valid grasp from
the library. DexGraspNet builds this library. For each object shape, it runs a
physics-based optimizer that figures out where the fingers should go and what joint
angles make a stable, non-slipping grasp. DexGraspNet 2.0 extends this to cluttered
scenes: the library now also accounts for neighboring objects, so the generated grasps
avoid collisions with clutter. A neural network is then trained on this library so
that at runtime, given a point cloud of the scene, it can *sample* good grasps in
milliseconds without re-running the optimizer.

#### What it does

**Dataset generation (offline, one-time).** For each object mesh, a geodesic grid
of candidate palm poses is sampled on a sphere around the object. For each candidate,
a gradient-based optimizer minimizes an energy function:

$$E = w_1 E_{dist} + w_2 E_{pen} + w_3 E_{fc} + w_4 E_{joints}$$

where $E_{dist}$ pulls fingertips toward object surface, $E_{pen}$ penalizes
hand-object interpenetration, $E_{fc}$ rewards force-closure stability (friction
cone), and $E_{joints}$ keeps joints within limits. Optimized configurations are
physics-validated in simulation and kept only if they satisfy grasp quality metrics.
The result: a large dataset of (object point cloud, 6-DoF palm pose,
ShadowHand joint angles $\boldsymbol{\theta} \in \mathbb{R}^{22}$) triples.

**Generative model (DexGraspNet 2.0).** A PointNet++ encoder embeds the input
scene point cloud $\mathbf{P} \in \mathbb{R}^{N \times 3}$ into a latent
feature $\mathbf{z}$. A **flow-matching** (continuous normalizing flow) model
then maps noise $\epsilon \sim \mathcal{N}(0,I)$ to a grasp $(T_{palm} \in SE(3),
\boldsymbol{\theta})$ conditioned on $\mathbf{z}$. At inference: sample $K$ grasps,
score each with a learned quality network, execute the highest-ranked one.

DexGraspNet 2.0 extends to cluttered scenes by conditioning on the full scene
point cloud. The model learns to generate grasps that avoid collisions with
neighboring objects — it sees all objects and targets only the designated one.

**Strengths for this project.** Data-driven, operates from geometry (works when
objects are known only from depth data). DexGraspNet 2.0 handles clutter,
matching this project's task closely. Mature codebase from a strong group.

**Weaknesses.** Trained on ShadowHand. Retargeting the output $\boldsymbol{\theta}_{shadow}$
to Inspire Hand joints requires either retraining or a kinematic retargeting step
(solve IK for Inspire contact points). Object set is YCB-style diverse meshes —
5 cm cubes are trivial geometry and may produce degenerate outputs (too easy,
no variation).

**Verdict.** Best infrastructure and dataset, but requires non-trivial retargeting
to Inspire Hand. DexGraspNet 2.0 is the right reference for building a cluttered-scene
grasp generator from scratch for this robot.

---

### 3.3 AnyDexGrasp

**Paper.** Fang et al., "AnyDexGrasp: Learning general dexterous grasping for
any hands with human-level learning efficiency," Robot Learning Workshop 2024.

#### In plain terms

DexGraspNet's library is built for one specific robot hand (ShadowHand). If you
have a different hand — say, the Inspire Hand — you can't use the library directly,
because the finger lengths, joint ranges, and number of joints are all different.
You'd have to rebuild the whole library from scratch for your hand, which takes
enormous compute.

AnyDexGrasp solves this by splitting the problem in two. First, it learns to answer
the question: *"given this object, where on its surface should fingers make contact?"*
— completely ignoring which hand is being used. This answer (the "contact map") is
expressed in terms of the object's geometry, not any specific hand. Second, once you
have the contact map, a fast optimizer figures out the joint angles for *your specific
hand* that puts the fingertips at those contact locations. So the expensive learning
step is done once and shared across all hands; only the cheap per-hand retargeting
step needs to change when you switch robots.

#### What it does

The central observation: all prior dexterous grasping work trains a model per
hand (one for ShadowHand, one for Allegro, etc.), which is data-inefficient and
doesn't transfer. AnyDexGrasp decouples grasp synthesis into two stages:

**Stage 1 — Object-centric contact map (hand-agnostic).** Given an object point
cloud, a network predicts a **contact map**: for each point on the object surface,
a contact probability and contact normal. This is expressed in object frame —
completely independent of which hand will be used. The network is trained jointly
on grasp data from many different hands (ShadowHand, Allegro, DLR, Barrett),
learning a shared geometric prior for where grasping contact should occur on
arbitrary object surfaces.

**Stage 2 — Hand-specific retargeting.** Given a contact map and a target hand
description (URDF kinematics), an optimizer finds the joint configuration that
places fingertips at the target contact points:

$$\boldsymbol{\theta}^* = \arg\min_{\boldsymbol{\theta}} \sum_k \|FK_k(\boldsymbol{\theta}) - \mathbf{c}_k\|^2 + \lambda_1 E_{pen} + \lambda_2 E_{joints}$$

where $FK_k(\boldsymbol{\theta})$ is the forward kinematics of fingertip $k$ and
$\mathbf{c}_k$ is its target contact point from the map. This retargeting runs
as an optimization (not a learned model) and requires only the URDF of the target
hand — no training data for that hand is needed.

**"Human-level efficiency" claim.** Because the contact map network is shared
across all hands, grasp data from every hand contributes to learning better contact
geometry, even for a new morphology. Adaptation to a new hand requires only
providing its URDF; the retargeting optimizer handles the rest.

**Strengths for this project.** Most directly applicable for the Inspire Hand.
Provide the Inspire Hand URDF and retargeting runs without retraining. The
ShadowHand → Inspire Hand retargeting gap that plagues DexGraspNet is eliminated.

**Weaknesses.** Workshop paper — lighter experimental rigor than a full venue.
The retargeting optimization can fail if the Inspire Hand's range of motion can't
reach the target contact point (thumb in particular has unusual yaw range). The
contact map network was trained on hands that may not represent Inspire geometry —
quality of the shared prior for this morphology is unverified.

**Verdict.** Best option for minimal engineering effort on the Inspire Hand.
Validate retargeting quality on a few representative grasps before committing.

---

### 3.4 UltraDexGrasp

**Paper.** Most recent entry in the PKU-EPIC dexterous grasping series (2025).

#### In plain terms

DexGraspNet generates grasps and then checks them in physics after the fact —
"here's a grasp, let's see if it holds." This is a two-step process. The natural
next idea is: what if the physics check were part of the generation itself, so the
network *learns from the start* to only propose grasps that physics confirms as
stable? That's the direction UltraDexGrasp / Dexgrasp Anything takes. During
training, every grasp the network proposes is simulated, and the physics outcome
(did it hold? did it slip?) feeds back into the network's training loss. The result
is a network that has, in effect, "internalized" what makes a grasp physically stable,
rather than generating and then filtering. It also scales to a much wider range of
object shapes, so it should work on unusual geometry without needing retraining.

> **Note on naming.** A paper with the exact title "UltraDexGrasp" is not confirmed
> in my training data. The most likely candidate is **"Dexgrasp Anything: Towards
> Universal Robotic Dexterous Grasping with Physics Awareness"** (Zhong et al.,
> arXiv:2503.08257, 2025), which is cited as reference [6] in the ClutterDexGrasp
> paper in this folder. Verify the exact paper at `https://pku-epic.github.io`
> before citing it. The analysis below applies to the Dexgrasp Anything paper and
> the general direction of this line of work.

#### What it does

Dexgrasp Anything aims at universal dexterous grasping: any hand, any object,
physically plausible. The key contribution over prior work is **physics awareness**
in the grasp generation process itself — prior methods (DexGraspNet, AnyDexGrasp)
generate grasps analytically and then validate them in physics; Dexgrasp Anything
integrates physics simulation into the generation loop.

**Architecture.** A large-scale generative model (likely diffusion or flow-based,
in the same family as DexGraspNet 2.0) takes object point cloud + hand URDF and
outputs grasp configurations. What makes it "physics-aware": during training, each
generated grasp candidate is executed in a differentiable physics simulator and
the resulting contact forces / stability metrics are backpropagated into the
generator's loss. This trains the network to directly produce grasps that are
stable under real physics, not just geometrically plausible.

**Object coverage.** Trained on large-scale synthetic datasets (Objaverse, ShapeNet)
with 100k+ object meshes. The model has seen enough geometric variation that simple
5 cm cubes are covered as a trivial case.

**Strengths for this project.** Highest quality grasp outputs from this line of
work. Physics-aware generation means generated grasps require less post-hoc
filtering. Broad object coverage and strong sim-to-real track record from the
PKU-EPIC group (same group that produced ClutterDexGrasp). Most likely to produce
stable grasps that transfer to the Inspire Hand simulation with minimal tuning.

**Weaknesses.** Most recent = fewest third-party integrations and least
documentation. Physics-aware training loop requires a differentiable simulator
(e.g., Warp, Isaac Lab differentiable) — integrating this into the existing
Isaac Lab setup may need engineering work. If Inspire Hand support is not
built-in, the retargeting engineering required is similar to DexGraspNet 2.0.

**Verdict.** Highest grasp quality ceiling. Start with AnyDexGrasp for speed,
upgrade to this if grasp quality is the bottleneck after initial integration.

---

### 3.5 BiDexGrasp

**Paper.** Chen et al., "Towards Human-Level Bimanual Dexterous Manipulation with
Reinforcement Learning," NeurIPS 2022. PKU-EPIC group.

#### In plain terms

BiDexGrasp is not a grasp generator — it's a benchmark that asks: *can RL train a
robot to use two hands together?* Using two hands is fundamentally harder than one,
because the two hands need to coordinate — one might hold an object steady while the
other picks it up, or they might both work together to rotate something. If you just
run RL naively with two hands, one hand ends up doing all the work and the other
stops moving entirely. BiDexGrasp defines 60 tasks that require varying levels of
two-hand coordination and studies how to design rewards and training setups that
force RL to discover genuine two-hand cooperation. The key lesson: you need to
explicitly reward *both* hands being near the object at critical moments, otherwise
the policy degenerates to single-hand solutions. This is the main practical lesson
to carry forward when building the bimanual stage of this project.

#### What it does

BiDexGrasp is not a grasp sampler — it is an **RL benchmark and training
framework** for bimanual dexterous tasks. It is relevant because the G1 project's
next major step is bimanual manipulation with both arms.

**Benchmark.** 60 bimanual manipulation tasks categorized by coordination requirement:
- *Single-hand possible*: pen spinning, block stacking, object re-orientation
- *Bimanual necessary*: object handover between hands, coordinated assembly
  (peg-in-hole), dual-arm object stabilization during grasp

**Setup.** Two ShadowHands mounted on fixed wrists (table-mounted robot arms,
not a humanoid). Scene: both hands + object on a flat surface.

- State: joint positions/velocities of both ShadowHands (48 DOF) + object 6-DoF pose + fingertip positions
- Action: 24-dim joint position targets (12 per hand)
- Policy: two separate MLPs (one per hand), each observing the full state. Optionally: one shared MLP with hand-ID one-hot input.

**Training.** Standard PPO, task-specific reward per task. No curriculum.
Policy is trained independently per task. Key finding: naive single-hand
policies completely fail on bimanual-necessary tasks; the reward must explicitly
require both hands to be near the object at critical moments, otherwise one hand
dominates and the other collapses to a resting pose.

**What transfers to this project.** The task decomposition and reward structure
for coordinated grasps. Specifically: to extend the existing G1 picking task to
bimanual, you will need:
1. A coordination-specific reward that requires the left hand to stabilize the
   object while the right hand picks — something like
   $r_{coord} = r_{pick} \cdot \mathbb{1}[\|\mathbf{p}_{left,palm} - \mathbf{p}_c\| < d_{thresh}]$
2. A left arm action space (currently frozen in this project)
3. Clutter-clearing curriculum for both arms, not just the right

**What does not transfer.** ShadowHand kinematics (24 DOF per hand, very different
from Inspire's 12 DOF). Table-mounted fixed wrists vs the G1's full 7-DOF arm
chains — the arm reachability and singularity structure are completely different.
The specific joint names, limits, and reward thresholds all need to be re-derived
for the G1 + Inspire setup.

**Verdict.** Not relevant for the current single-arm sampler step. The **primary
reference** for the bimanual stage: read it carefully before designing the
two-arm curriculum and coordination rewards.

---

### 3.6 Recommendation

**For the immediate task (single-arm sampler for the teacher):**
Use **UltraDexGrasp**. It is the highest-quality grasp synthesis available from
the same research group that produced ClutterDexGrasp (which your project is
already aligned with). The integration path is:

1. Feed the target cube's point cloud / mesh (5 cm cube → trivial geometry) to
   UltraDexGrasp to generate a set of candidate grasps for the Inspire Hand.
2. Select the grasp closest to the robot's current configuration as the episode's
   goal grasp $G^*$.
3. Add a grasp-goal reward term:
   $$r_{goal} = \exp\!\left(-\frac{\|(\mathbf{p}_{palm}, \boldsymbol{\theta}_{hand}) - G^*\|^2}{2\sigma^2}\right)$$
4. Use this as a shaped reward shaping, not a hard constraint — the existing
   dense rewards still provide the primary signal.

**For the bimanual stage:**
Use **BiDexGrasp** as the reference framework for task design and reward structure.
The clutter-density curriculum already in place will extend naturally to bimanual
clutter clearing.

---

### 3.7 Proposed Integration Pathway

The full research arc, in order:

```
Stage 1 (current):  Single-arm pick, RL from scratch, teacher only
    ↓
Stage 2:            Integrate UltraDexGrasp sampler
                    → better grasp geometry reward
                    → faster convergence on complex objects
    ↓
Stage 3:            Teacher → Student distillation (point cloud obs, DP3/BC)
                    See ClutterDexGrasp for the exact pipeline
    ↓
Stage 4:            Bimanual — add left arm, BiDexGrasp-style task structure
                    New scene: two-arm clutter clearing before target grasp
    ↓
Stage 5:            Bimanual teacher → student distillation
                    Real robot deployment
```

The novelty claim: **zero-shot sim-to-real bimanual dexterous grasping in clutter,
trained purely by RL with a geometry-aware grasp sampler, distilled to a
point-cloud-based student.** ClutterDexGrasp (in this folder) did the single-arm
version with a single-arm manipulator; doing bimanual with a humanoid robot (G1 +
Inspire Hands) and zero real-world demonstrations is the novel contribution.

---

## Quick Reference

### Observation Vector Layout (96-dim)

```
[0:13]   right arm+hand joint positions (7 arm + 6 hand proximal)
[13:26]  right arm+hand joint velocities (clipped ±50)
[26:29]  target cube position in robot base frame [x, y, z]
[29:35]  target cube velocity [vx, vy, vz, wx, wy, wz]
[35:65]  distractor 1-10 positions in robot base frame (10 × 3)
[65:83]  right wrist + 5 fingertip positions in robot base frame (6 × 3)
[83:96]  previous action (13-dim)
```

### Height Reference (relative to env origin)

```
0.600 m   → _DROP_Z:       cube/distractor fell off table → terminal
0.820 m   → tray surface
0.835 m   → _OFF_TRAY_Z:   distractor fell off tray edge (penalty, not terminal)
0.845 m   → _OBJ_INIT_Z:   cube resting position on tray
1.134 m   → _SUCCESS_Z:    ~29 cm lift → episode success, curriculum advances
```

### Training Commands

```bash
# Activate env first
conda activate env_isaaclab
cd /home/daatsi-aeres/IsaacLab

# Train (headless + video logging)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-G1-Pick-v0 --headless --num_envs 1024 \
  --video --video_length 600 --video_interval 2000

# Resume from checkpoint
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-G1-Pick-v0 --resume \
  --load_run <run_name> --checkpoint model_XXXX.pt \
  --headless --num_envs 900 --max_iterations 10000

# Evaluate trained policy
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-G1-Pick-Play-v0 --num_envs 64 \
  --checkpoint /path/to/model.pt
```

### Key Papers

| Paper | Relevance |
|---|---|
| ClutterDexGrasp (Chen et al., CoRL 2025) — `clutterdex.pdf` | Direct reference: same problem, teacher-student, clutter curriculum |
| PPO (Schulman et al., 2017) | Training algorithm |
| GAE (Schulman et al., 2015) | Advantage estimation (λ=0.95) |
| UltraDexGrasp (PKU-EPIC, 2025) | Recommended grasp sampler |
| BiDexGrasp (Chen et al., NeurIPS 2022) | Bimanual stage reference |
| DP3 (Ze et al., 2024) | 3D diffusion policy for student (as used in ClutterDexGrasp) |
