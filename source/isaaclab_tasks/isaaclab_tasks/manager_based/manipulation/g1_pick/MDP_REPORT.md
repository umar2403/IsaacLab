# G1 Pick — MDP Technical Report

A full specification of the Markov Decision Process defined in this environment, derived from the source code.

---

## 1. Problem Statement

Train a Unitree G1 humanoid robot (with Inspire dexterous right hand) to reach into a cluttered tray and pick a specific red target cube, lifting it ~29 cm clear of the tray surface, while leaving all distractor cubes on the tray.

The robot's lower body (legs, waist) and left arm are frozen. Only the **right arm** (7 DOF) and **right Inspire hand** (6 controllable proximal joints) are active.

---

## 2. Scene / Physical Setup

| Entity | Description | Position (x, y, z) |
|--------|-------------|---------------------|
| Robot (G1 + Inspire) | Fixed standing pose, root at (−0.1, 0, 0.74) | frozen in place |
| Table | Static cuboid 0.6 × 1.2 × 0.80 m | (0.4, 0, 0.40) |
| Tray | Kinematic slab 0.4 × 0.6 × 0.02 m | (0.4, 0, 0.81) |
| **Target cube** (red) | 5 cm × 5 cm × 5 cm, mass 0.2 kg, friction 1.0 | (0.35, 0, **0.845**) ± random jitter |
| Distractors 1–10 (colored) | Identical 5 cm cubes, mass 0.2 kg | Ring layout around target |

**Tray surface height**: 0.820 m. **Target cube center** init at 0.845 m.

**Distractor anchor positions** (3 rings around target at [0.35, 0.0]):

| Ring | Distractors | Purpose |
|------|-------------|---------|
| Inner (≤8 cm) | 1, 2, 3 | Block direct approach path |
| Mid (12 cm) | 4, 5, 6, 7 | Mid-workspace clutter |
| Outer (18 cm) | 8, 9, 10 | Edge clutter |

Each distractor resets with ±3 cm uniform jitter per episode.

---

## 3. Action Space

**Dimension: 13** (continuous, joint position targets)

Actions are relative offsets from the default pose (`use_default_offset=True`).

### 3a. Right Arm — 7 DOF (scale 0.3)

| # | Joint |
|---|-------|
| 0 | `right_shoulder_pitch_joint` |
| 1 | `right_shoulder_roll_joint` |
| 2 | `right_shoulder_yaw_joint` |
| 3 | `right_elbow_joint` |
| 4 | `right_wrist_roll_joint` |
| 5 | `right_wrist_pitch_joint` |
| 6 | `right_wrist_yaw_joint` |

### 3b. Right Hand — 6 DOF proximal joints (scale 0.5)

| # | Joint | Controls |
|---|-------|----------|
| 7 | `R_thumb_proximal_yaw_joint` | Thumb abduction |
| 8 | `R_thumb_proximal_pitch_joint` | Thumb flex (drives intermediate × 0.8024, distal × 0.7622) |
| 9 | `R_index_proximal_joint` | Index flex (drives intermediate × 1.0843) |
| 10 | `R_middle_proximal_joint` | Middle flex (drives intermediate × 1.0843) |
| 11 | `R_ring_proximal_joint` | Ring flex (drives intermediate × 1.0843) |
| 12 | `R_pinky_proximal_joint` | Pinky flex (drives intermediate × 1.0843) |

**Mimic coupling** (`InspireMimicAction`): The policy only controls the 6 proximal joints. The 6 downstream intermediate/distal joints are driven automatically using hardware transmission ratios from the Inspire URDF, reducing the effective action space while preserving physical realism.

**Arm actuator**: stiffness 300, damping 30 — stiff position control.  
**Hand actuator**: stiffness 100, damping 0.5 — compliant for grasping.

---

## 4. Observation Space

**Total dimension: ~96** (all concatenated into a flat 1D tensor for an MLP policy)

| # | Observation Term | Dim | Description |
|---|-----------------|-----|-------------|
| 1 | `joint_pos` | 13 | Relative joint positions (arm 7 + hand 6) |
| 2 | `joint_vel` | 13 | Relative joint velocities, clipped ±50 |
| 3 | `object_pos_b` | 3 | Target cube XYZ in robot root frame |
| 4 | `object_vel` | 6 | Target cube linear + angular velocity (world frame) |
| 5–14 | `distractor_{1–10}_pos_b` | 30 | All 10 distractor positions in robot root frame |
| 15 | `right_fingertip_pos` | 18 | 6 tracked hand bodies (palm + 5 tips) × 3, in robot root frame |
| 16 | `actions` | 13 | Previous action (action history, 1 step) |

**Total: 13 + 13 + 3 + 6 + 30 + 18 + 13 = 96 dims**

**Tracked hand bodies** (for rewards and observations):

| Index | Body | Role |
|-------|------|------|
| 0 | `right_wrist_yaw_link` | Palm anchor |
| 1 | `R_thumb_distal` | Thumb tip |
| 2 | `R_index_intermediate` | Index tip |
| 3 | `R_middle_intermediate` | Middle tip |
| 4 | `R_ring_intermediate` | Ring tip |
| 5 | `R_pinky_intermediate` | Pinky tip |

No observation noise is added (`enable_corruption = False`). All positions are expressed in the robot's local frame via `quat_apply_inverse` to be pose-invariant.

---

## 5. Reward Function

All rewards are computed per-step. The active reward in the environment config is `compute_task_reward` (defined inline in `g1_pick_env_cfg.py`) plus five penalty terms.

### 5a. Task Reward (weight 1.0)

A **single hierarchical shaping function** that returns the sum of five sub-rewards:

```
R_task = R_posture + R_reach + 2·R_grasp + R_lift_cont + R_success
```

| Sub-reward | Formula | Range | Purpose |
|-----------|---------|-------|---------|
| **Posture** | `1 − tanh(clamp(‖palm − (cube+0.08ẑ)‖ − 0.03, 0) / 0.3)` | [0, 1] | Palm hovers 8 cm above cube |
| **Reach** | `1 − tanh(‖fingertip_midpoint − cube‖ / 0.25)` | [0, 1] | Finger cluster centroid near cube |
| **Grasp** | `0.5·(1−tanh(thumb_dist/0.055)) + 0.5·(1−tanh(finger_dist/0.055))` | [0, 1] | Both thumb and fingers within 2.5 cm of cube surface (25 mm deadband) |
| **Lift (cont.)** | `clamp(cube_z − 0.845, 0, 0.30) × 2.0 × is_grasped` | [0, 0.6] | Proportional to lift height, gated by grasp quality |
| **Success bonus** | `is_lifted × is_grasped × 1000.0` | {0, 1000} | Sparse: cube above 1.134 m AND grasped |

**Grasp gate** used for lift/success:
```
is_grasped = (1 − tanh(thumb_dist/0.06)) × (1 − tanh(finger_dist/0.06))
```

**Drop override**: If `cube_z < 0.600` (fell off table), the entire task reward is replaced with **−0.5**.

### 5b. Penalty Terms

| Term | Weight | Function | Purpose |
|------|--------|----------|---------|
| `action_smoothness` | −3.0 | `0.005·Σ(aₜ−aₜ₋₁)² + 0.001·Σω²` | Penalizes jerky motions and joint velocities |
| `fingertip_impact` | −2.0 | `mean(tanh(‖Δv_fingertip‖ / 3.0))` over 6 hand bodies | Penalizes sudden hand acceleration (slam/jab contact) |
| `distractor_accel` | −3.0 | `mean(tanh(‖Δv_distractor‖ / 2.0))` over 10 distractors | Penalizes bumping/knocking distractors |
| `distractor_off_tray` | −10.0 | Count of distractors with `z < 0.835` per step | Per-step accumulating penalty for each cube off the tray |
| `distractor_drop` | −100.0 | 1 if ANY distractor `z < 0.600` | Sparse terminal penalty for knocking a distractor off the table |

---

## 6. Termination Conditions

Episodes end when any of the following occur:

| Condition | Type | Trigger |
|-----------|------|---------|
| `time_out` | Truncation | 8 seconds elapsed (960 sim steps at 120 Hz × 4 decimation = 240 control steps) |
| `target_lifted` | **Success** | Target cube `z > 1.134 m` (~29 cm above tray) |
| `target_dropped` | Failure | Target cube `z < 0.600 m` (fell off table) |
| `distractor_dropped` | Failure | ANY of the 10 distractors `z < 0.600 m` |

**Episode length**: 8 seconds → 960 physics steps (120 Hz) → **240 control steps** (4× decimation).

---

## 7. Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Physics timestep | 1/120 s (120 Hz) |
| Control decimation | 4 (policy runs at 30 Hz) |
| Episode length | 8 s (240 control steps) |
| Parallel environments (train) | 4096 |
| Parallel environments (eval) | 64 |
| Solver position iterations (robot) | 32 |
| Solver position iterations (objects) | 16 |
| Cube contact offset | 0.005 m |
| Physics: `bounce_threshold_velocity` | 0.2 m/s |

---

## 8. Domain Randomization (Reset Events)

Per episode reset, the following randomizations are applied:

| Event | Joints/Objects | Randomization |
|-------|---------------|---------------|
| `reset_right_arm` | 7 arm joints | ±0.05 rad offset from default pose |
| `reset_right_hand` | 6 hand joints | ±0.05 rad offset from default pose |
| `freeze_left_arm` | 13 left arm+hand joints | Exact zero offset (hard frozen) |
| `freeze_lower_body` | All leg + waist joints | Exact zero offset (hard frozen) |
| `reset_target_object` | Target cube | x: ±10 cm, y: ±5 cm from anchor |
| `reset_distractor_{1–10}` | Each distractor | x: ±3 cm, y: ±3 cm from anchor |

The ±10 cm / ±5 cm target jitter forces the policy to generalize across a workspace region rather than memorizing a single cube location.

---

## 9. Curriculum

The curriculum operates on two independent axes: **training phase** (reward gating) and **clutter difficulty** (number of active distractors).

### 9a. Phase-Based Reward Gating (3 phases)

The `PickingCurriculumScheduler` monitors rolling episode reward statistics and gates reward terms on/off.

| Phase | Trigger | Newly Enabled Reward Terms |
|-------|---------|---------------------------|
| **0** — Reaching only | Start of training | `reaching_target` |
| **1** — Grasping + Lifting + Clutter | Mean reaching reward/step ≥ 0.4 over last 500 episodes | `lifting_target` (×5.0), `declutter` (×2.0) |
| **2** — Full pick | Mean lifting reward/step ≥ 0.75 over last 500 episodes | `pick_success` (×1.0) |

Minimum history of 50 completed episodes required before any phase transition.

### 9b. Clutter Difficulty (0–60 scale)

Each environment tracks an individual difficulty score. It increments by 1 on a successful pick (cube lifted above 15 cm) and decrements by 1 on failure (or holds with `promotion_only=True`), clamped to [0, 60].

| Difficulty Range | Active Distractors on Tray |
|-----------------|---------------------------|
| 0 – 29 | 0 (all hidden below table at z = −5 m) |
| 30 – 39 | 1–2 (random) |
| 40 – 49 | 3–5 (random) |
| 50 – 60 | 5–7 (random) |

Distractor activation is staggered: distractor N activates at difficulty ≥ 30 + (N−1)×10. All inactive distractors are teleported to z = −5 m so they cannot interfere.

---

## 10. Policy Architecture

**Algorithm**: PPO (Proximal Policy Optimization) via RSL-RL

| Hyperparameter | Value |
|---------------|-------|
| Actor/Critic network | MLP [512, 256, 128] with ELU activation |
| Input normalization | Enabled for both actor and critic |
| Rollout steps per env | 32 |
| Mini-batches | 4 |
| Learning epochs per rollout | 5 |
| Learning rate | 3×10⁻⁴ (adaptive schedule) |
| PPO clip ε | 0.2 |
| Entropy coefficient | 0.005 |
| Discount γ | 0.99 |
| GAE λ | 0.95 |
| Desired KL | 0.01 |
| Max grad norm | 1.0 |
| Initial noise std | 0.8 |
| Max training iterations | 5000 |
| Checkpoint interval | 50 iterations |

---

## 11. Height Thresholds Summary

| Constant | Value (m) | Role |
|----------|-----------|------|
| `_OBJ_INIT_Z` | 0.845 | Cube center at rest on tray |
| `_OFF_TRAY_Z` | 0.835 | Cube below this → distractor_off_tray penalty fires |
| `_SUCCESS_Z` | 1.134 | Cube above this → `target_lifted` success termination |
| `_DROP_Z` | 0.600 | Below this → early termination (target OR distractor) |
| Lift threshold (curriculum) | 0.150 | Used internally by curriculum to count a pick as successful |

---

## 12. Key Design Choices

**Mimic joints**: The Inspire hand has 12 joints but the policy only controls 6 proximal joints. Downstream joints are driven by fixed gear ratios (`×0.8024`, `×0.9487`, `×1.0843`) matching hardware transmission specs. This halves the hand action space without losing physical fidelity.

**Deadband in grasp reward**: Fingertip distances are clamped: `clamp(raw_dist − 0.025, min=0)` — a 2.5 cm contact radius. Tips physically touching the cube surface (within 2.5 cm of center) receive maximum grasp reward, preventing reward from peaking at impossible zero-distance configurations.

**Drop override in reward**: When the target cube falls off the table (`z < 0.600`), the task reward is hard-overridden to −0.5, providing a clear negative gradient before the termination fires.

**Left arm/lower body frozen via stiffness override**: Instead of removing joints, the left hand and leg actuators are given extreme stiffness (10,000) and damping (1,000) at environment init. This keeps the articulation physically consistent while making those DOF immovable for the policy.

**Distractor penalty hierarchy**: Three layers create a smooth gradient away from knocking distractors: (1) acceleration penalty on any contact, (2) per-step off-tray penalty while cube is displaced, (3) large sparse penalty + termination if cube falls off the table.
