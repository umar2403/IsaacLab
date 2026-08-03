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

> **Rewritten from source on 2026-07-30.** The previous version of this section only
> covered `compute_task_reward` and the 5 penalty terms — **6 terms total**. The
> active config on `umar/g1-pick-no-topdown` has **8 reward terms**: it also carries
> two grasp-goal pose-mimicking terms (`grasp_goal_palm`, `grasp_goal_hand`) that this
> section previously omitted entirely, and `compute_task_reward` itself is called with
> different parameters on this branch (`use_posture=False`, `pose_gated_success=True`)
> than the version originally documented here. Everything below is verified against the
> current `g1_pick_env_cfg.py` and `mdp/grasp_goal.py`.

All reward terms are computed every control step (30 Hz) and summed with their configured
weight.

### 5.0 Term overview

| Term | Weight | Role |
|---|---:|---|
| `task_reward` | $1.0$ | reach + grasp + lift + pose-gated success — §5.1 |
| `grasp_goal_palm` | $2.0$ | pulls the palm toward the UltraDexGrasp goal pose — §5.2.1 |
| `grasp_goal_hand` | $1.0$ | pulls the 6 finger joints toward the goal hand shape, gated on palm proximity — §5.2.2 |
| `action_smoothness` | $-3.0$ | penalizes jerky actions / high joint velocity |
| `fingertip_impact` | $-2.0$ | penalizes sudden hand-body acceleration (slams) |
| `distractor_accel` | $-3.0$ | penalizes sudden distractor acceleration (bumps) |
| `distractor_off_tray` | $-10.0$ | per-step penalty while any distractor sits below tray height |
| `distractor_drop` | $-100.0$ | one-time penalty (+ termination) if any distractor falls off the table |
| `grasp_reach` | $1.0$ | pulls each fingertip toward its own contact point on the cube — §5.2.4 |

**Two different bodies are both informally "the palm" in this codebase — keep them
distinct:**

- `right_wrist_yaw_link` — read by `compute_task_reward`'s (currently disabled) posture
  term.
- `R_hand_base_link` — read by both grasp-goal terms (§5.2) and by the pose-gated
  success bonus (§5.1.6). The two links sit a few centimeters apart.

### 5.1 Task reward — `compute_task_reward` (weight $1.0$)

Let $\mathbf{p}_c \in \mathbb{R}^3$ be the cube position, $\mathbf{p}_{\text{palm}}$ the
`right_wrist_yaw_link` position, and $\{\mathbf{p}_{\text{tip},i}\}_{i=0}^{4}$ the 5
tracked fingertip bodies ($i=0$: thumb, $i=1\ldots4$: index/middle/ring/pinky).

**Deadband distances** (subtract the cube's 2.5 cm half-edge, so touching the surface
reads as zero):

$$
d_{\text{thumb}} = \max\!\Big(\lVert \mathbf{p}_{\text{tip},0} - \mathbf{p}_c \rVert - 0.025,\ 0\Big)
\qquad
d_{\text{finger}} = \max\!\Big(\tfrac{1}{4}\textstyle\sum_{i=1}^{4}\lVert \mathbf{p}_{\text{tip},i} - \mathbf{p}_c \rVert - 0.025,\ 0\Big)
$$

#### 5.1.1 Posture reward — **disabled** on this branch (`use_posture=False`)

$$
r_{\text{posture}} =
\begin{cases}
\displaystyle 1 - \tanh\!\frac{\max\big(\lVert \mathbf{p}_{\text{palm}} - (\mathbf{p}_c + [0,0,0.08])\rVert - 0.03,\ 0\big)}{0.3} & \texttt{use\_posture=True} \\[6pt]
0 & \texttt{use\_posture=False\ (current)}
\end{cases}
$$

Why it's off: this term hard-codes the palm target 8 cm **straight above** the cube — a
top-down assumption. Grasp #12 is a ~65° side approach, so this term would fight the
grasp-goal palm reward (§5.2.1) instead of agreeing with it. Palm placement is now owned
entirely by `grasp_goal_palm`.

#### 5.1.2 Reach reward

Centroid of all 5 fingertips toward the cube:

$$
\bar{\mathbf{p}}_{\text{tip}} = \frac{1}{5}\sum_{i=0}^{4} \mathbf{p}_{\text{tip},i}
\qquad
r_{\text{reach}} = 1 - \tanh\!\left(\frac{\lVert \bar{\mathbf{p}}_{\text{tip}} - \mathbf{p}_c \rVert}{0.25}\right)
$$

#### 5.1.3 Grasp reward

$$
r_{\text{thumb}} = 1 - \tanh\!\left(\frac{d_{\text{thumb}}}{0.055}\right)
\qquad
r_{\text{finger}} = 1 - \tanh\!\left(\frac{d_{\text{finger}}}{0.055}\right)
\qquad
r_{\text{grasp}} = \frac{r_{\text{thumb}} + r_{\text{finger}}}{2}
$$

Counted with weight $2$ inside the task-reward sum (§5.1.7).

#### 5.1.4 Grasp gate

Soft AND of "thumb close" and "fingers close", used to gate lift + success so the policy
can't earn them by shoving the cube with an open palm:

$$
g = \Big(1 - \tanh\tfrac{d_{\text{thumb}}}{0.06}\Big)\Big(1 - \tanh\tfrac{d_{\text{finger}}}{0.06}\Big) \ \in [0,1]
$$

#### 5.1.5 Continuous lift reward

$$
\Delta h = \operatorname{clamp}(p_{c,z} - 0.845,\ 0,\ 0.30)
\qquad
r_{\text{lift}} = 2.0\,\Delta h \cdot g \ \in [0,\ 0.6]
$$

#### 5.1.6 Success bonus — now **pose-gated** (`pose_gated_success=True`)

This is where the grasp goal reaches into the terminal bonus. Every step the live goal
is re-attached to the cube's current pose (mechanism shared with §5.2), then compared
against the current `R_hand_base_link` pose $(\mathbf{p}_{\text{hand}}, \mathbf{q}_{\text{hand}})$
against the goal $(\mathbf{p}^\star, \mathbf{q}^\star)$:

$$
e_{\text{pos}} = \lVert \mathbf{p}_{\text{hand}} - \mathbf{p}^\star \rVert
\qquad
e_{\text{ang}} = \operatorname{quat\_error\_magnitude}(\mathbf{q}_{\text{hand}}, \mathbf{q}^\star) \ \in [0,\pi]
$$

$$
\text{pose\_match} = \Big(1 - \tanh\tfrac{e_{\text{pos}}}{0.10}\Big)\Big(1 - \tanh\tfrac{e_{\text{ang}}}{0.8}\Big) \ \in [0,1]
$$

$$
\text{success\_scale} = \underbrace{0.2}_{\text{success\_floor}} + (1-0.2)\cdot\text{pose\_match} \ \in [0.2,\ 1.0]
$$

$$
r_{\text{success}} = \mathbb{1}[p_{c,z} > 1.134]\cdot g \cdot \text{success\_scale}\cdot 1000
$$

**Interpretation**: lifting the cube always pays **at least 200** (the `success_floor`
$=0.2$), so the pick signal never fully vanishes even if the exact UltraDex pose proves
unreachable — but lifting it **in grasp #12's pose** pays the full **1000**. This is a
structural fix, not just reward shaping: instead of *adding* a small pose term next to a
1000-point bonus (which the policy learns to ignore entirely — see `CONTEXT.md` §7–8),
the size of the bonus itself now *depends on* the pose match.

#### 5.1.7 Assembly + drop override

$$
r_{\text{task}} =
\begin{cases}
-0.5 & p_{c,z} < 0.600 \quad \text{(cube fell off the table)} \\[4pt]
r_{\text{posture}} + r_{\text{reach}} + 2\,r_{\text{grasp}} + r_{\text{lift}} + r_{\text{success}} & \text{otherwise}
\end{cases}
$$

---

### 5.2 Grasp-goal pose-mimicking rewards — `grasp_goal_palm` (weight $2.0$) + `grasp_goal_hand` (weight $1.0$)

These are the two terms this doc previously omitted, and the ones most worth
understanding in detail: they pull the robot toward a **specific grasp pulled from the
UltraDexGrasp/BODex library** (`grasp_sampler/grasp_dataset/cube_5cm_grasps_valid.npz`),
rather than just rewarding "get close to the cube center" the way §5.1 does.

**The goal itself.** At every episode reset, the `sample_grasp_goal` event term
(`mdp/grasp_goal.py`) assigns the **same fixed library entry to every environment** —
currently **grasp #12**, the sphere-contact-gated FSWO optimizer's pick (a 5-finger
envelope wrap; confirmed reachable off-center by the arm, unlike the far-tray positions
where no grasp in the library works because the arm physically can't reach). This
produces three per-env goal buffers, all in **world frame**:

- $\mathbf{p}^\star \in \mathbb{R}^3$ — target `R_hand_base_link` position (`goal_pos_w`)
- $\mathbf{q}^\star \in \mathbb{R}^4$ — target `R_hand_base_link` orientation (`goal_quat_w`)
- $\mathbf{q}^\star_{\text{hand}} \in \mathbb{R}^6$ — target 6 proximal joint angles, in
  policy joint order (`goal_hand_q`)

Because grasp #12 is stored **relative to the cube**, $\mathbf{p}^\star$ and
$\mathbf{q}^\star$ are **re-attached to the cube's live pose every step**
(`update_live_goals`, cached per `env.common_step_counter` so it only recomputes once
even though both reward terms call it):

$$
\mathbf{p}^\star_t = R(\mathbf{q}_{\text{cube},t})\,\mathbf{p}^\star_{\text{obj}} + \mathbf{p}_{\text{cube},t}
\qquad
\mathbf{q}^\star_t = \mathbf{q}_{\text{cube},t} \otimes \mathbf{q}^\star_{\text{obj}}
$$

where $\mathbf{p}^\star_{\text{obj}}, \mathbf{q}^\star_{\text{obj}}$ are grasp #12's pose
in the cube's own frame (constant, loaded once from the `.npz` at env construction).
Without this live re-attachment, the goal would freeze at the cube's spawn pose — the
moment anything nudges the cube, the reward would point at empty air (this was an
actual bug early in the project; see `grasp_sampler/README.md` problem #14).

#### 5.2.1 Palm-pose reward — `grasp_goal_palm_reward` (weight $2.0$)

Pulls `R_hand_base_link` toward $(\mathbf{p}^\star, \mathbf{q}^\star)$ — position **and**
orientation, in the current config (`pos_mode="full"`):

$$
d_{\text{pos}} = \lVert \mathbf{p}_{\text{hand}} - \mathbf{p}^\star \rVert
\qquad
d_{\text{ang}} = \operatorname{quat\_error\_magnitude}(\mathbf{q}_{\text{hand}}, \mathbf{q}^\star)
$$

$$
r_{\text{pos}} = 1 - \tanh\!\left(\frac{d_{\text{pos}}}{0.15}\right)
\qquad
r_{\text{orient}} = 1 - \tanh\!\left(\frac{d_{\text{ang}}}{0.6}\right)
$$

$$
r_{\text{palm}} = (1-w_o)\, r_{\text{pos}} + w_o\, r_{\text{orient}}, \qquad w_o = 0.5
$$

i.e. $r_{\text{palm}} = 0.5\, r_{\text{pos}} + 0.5\, r_{\text{orient}} \in [0,1]$. This is
a **dense, ungated** reward — it pays from anywhere in the workspace, purely as a
function of how close the hand's pose is to the goal, every single step.

(`pos_mode` also supports `"height"`, which would reward only the vertical offset
$|\,p_{\text{hand},z} - p^\star_z\,|$ — meant for when a separate posture term already
centers the palm horizontally. Not used here since `use_posture=False`.)

#### 5.2.2 Hand-configuration reward — `grasp_goal_hand_config_reward` (weight $1.0$)

Pulls the 6 controllable proximal joints $\mathbf{q} \in \mathbb{R}^6$ toward the goal's
joint angles $\mathbf{q}^\star_{\text{hand}}$ — but **gated** on palm proximity, so the
policy can't get finger-shape credit while the hand is still across the room:

$$
\text{gate} = \operatorname{clamp}\!\left(1 - \tanh\frac{d_{\text{pos}}}{0.20},\ 0,\ \infty\right)
$$

$$
r_{\text{hand}} = \text{gate}\cdot\left(1 - \tanh\frac{\lVert \mathbf{q} - \mathbf{q}^\star_{\text{hand}}\rVert}{0.5}\right)
$$

The gate reuses the **same** $d_{\text{pos}}$ from §5.2.1, but with its own width ($0.20$
m vs. $0.15$ m) — wide enough that the finger-shaping signal starts to switch on well
before the palm has fully arrived, giving a smooth handoff instead of a cliff.

#### 5.2.3 Four independent pose-matching signals — don't conflate them

It's tempting to think of this as one "match the grasp" reward. There are actually
**four**, all reading the same live goal but with different tolerances and different
jobs:

| # | Where | Std devs | When it pays | Job |
|---|---|---|---|---|
| 1 | `grasp_goal_palm` (§5.2.1) | $0.15$ m pos / $0.6$ rad ang | dense, every step | pulls the palm toward the goal pose continuously |
| 2 | `grasp_goal_hand` (§5.2.2) | gate $0.20$ m; joint std $0.5$ rad | dense, once palm-gated | pulls fingers toward the goal *joint angles* once the palm is close |
| 3 | pose-gated success (§5.1.6) | $0.10$ m pos / $0.8$ rad ang | **only at the instant of a successful lift** | scales the terminal 1000-point bonus by pose quality |
| 4 | `grasp_reach` (§5.2.4) | $\sigma_{\text{reach}}=0.05$ m per finger | dense, every step | pulls each fingertip toward its own *Cartesian contact point* on the cube |

Per `CONTEXT.md` §7–8, signal (1) has sat essentially flat at $\approx 0.028$–$0.030$
across 6000+ training iterations despite being dense and weighted $2.0$ — the policy
isn't moving toward grasp #12's pose at all, even though it picks the cube reliably
(94–96%) using whatever grasp it discovered on its own. That's the open problem this
whole reward structure exists to diagnose. Now that reachability of #12 is confirmed
(your IK check), signal (4) is a complementary shaping term that attacks the same
problem from task space instead of configuration space — see the rationale in §5.2.4.

#### 5.2.4 Fingertip contact-point reward — `grasp_reach` (weight $1.0$, implemented 2026-07-30)

**Motivation.** Signal (2), `grasp_goal_hand`, matches the policy's 6 proximal **joint
angles** to grasp #12's joint angles. But joint-space matching is once removed from what
actually matters: whether each fingertip lands on the *specific patch of cube surface*
the optimizer identified as a good contact (per `grasp_selection`'s sphere-contact
scoring — §"Grasp gate" in `OPTIMIZER_IMPLEMENTATION_SPEC.md`). Two hands with slightly
different joint angles can produce nearly the same fingertip placement, and — per
`grasp_sampler/README.md` Discovery 2 — the synthesis URDF and the simulated USD hand are
known to disagree geometrically by 1–3 cm at the fingertips, so joint-angle matching
doesn't guarantee contact-point matching anyway. A reward defined directly in **task
space** (Cartesian distance from real fingertip to intended contact point) is more
directly tied to what a good grasp physically requires, and is robust to that model
mismatch in a way joint matching isn't.

**Fingertips vs. collision spheres.** The offline optimizer scores wrap quality using
~41 collision spheres across the whole hand (§"Version 2" of the optimizer spec), not
just the 5 fingertips. In principle the online reward could do the same. My
recommendation, matching your instinct: **use the 5 fingertips**, not the spheres, for
this online term:

- The 5 fingertip bodies are *already* tracked (`_RIGHT_HAND_BODIES`) and already read by
  `compute_task_reward`'s reach/grasp terms — no new body tracking, no new per-step FK.
- Tracking ~41 spheres online would mean live-FK'ing ~13 hand links every step for every
  parallel env — real engineering cost for what's likely marginal benefit here: the
  spheres already did their job *offline* (they're why grasp #12 was selected as the
  best wrap in the first place). The online reward doesn't need to re-derive wrap
  quality; it just needs to nudge the policy toward the finger placement that a
  pre-vetted grasp already specifies.
- If fingertip-only guidance turns out to be too coarse (e.g. the policy matches the 5
  tip points but still doesn't wrap correctly), sphere-based shaping is the natural
  escalation — but it's not the right place to start.

**Target contact points (precomputed offline, cached, loaded lazily at runtime).**
`ultradex_repo/` (the URDF + collision-sphere YAML `grasp_selection` needs for FK) is
gitignored and not present in every checkout — the same reason `get_optimal_grasp_idx()`
falls back to the cached `scores.json` instead of recomputing from the optimizer every
time. This term follows the identical idiom: `grasp_selection/cache_fingertip_contacts.py`
(run once against grasp #12's stored root pose $(\mathbf p_g, \mathbf q_g)$ and 6 joint
angles $\boldsymbol\theta_g$) forward-kinematics the 5 fingertip links
(`thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip`) in the cube frame, and caches
the result to `grasp_selection/fingertip_contacts.json` (committed to the repo). The
reward function (`mdp.grasp_reach_reward`) just loads that cache on first call — no
URDF/FK dependency at training time. If the optimizer's pick ever changes from #12, rerun
the caching script.

$$
\mathbf t_i = \mathrm{FK}_i(\mathbf p_g, \mathbf q_g, \boldsymbol\theta_g), \qquad i \in \{\text{thumb, index, middle, ring, pinky}\}
$$

then project each onto the nearest cube face — exactly the `cube_contact()` step
`grasp_selection/hand_model.py` already implements and the optimizer already uses (here
reused purely as a geometry helper, not for re-selecting a grasp):

$$
\mathbf c^\star_i = \Pi_{\text{cube}}(\mathbf t_i), \qquad
\Pi_{\text{cube}}(\mathbf p) = \mathbf p + \big(h\,\mathrm{sign}(p_k) - p_k\big)\hat{\mathbf e}_k,
\quad k = \arg\max_{a\in\{x,y,z\}} \frac{|p_a|}{h},\ \ h=0.025\text{ m}
$$

This yields 5 fixed points $\mathbf c^\star_i$ in the **cube's own frame** — "where this
fingertip should touch the cube surface for grasp #12."

**Live tracking (goal follows the cube, identical mechanism to §5.2's palm/hand goals):**

$$
\mathbf c^\star_{i,t} = R(\mathbf q_{\text{cube},t})\,\mathbf c^\star_i + \mathbf p_{\text{cube},t}
$$

**Reward** — mean, over the 5 tracked fingertips, of a bounded per-finger term (the same
mean-of-tanh style already used by `fingertip_impact`/`distractor_accel`, chosen over
tanh-of-mean so one badly-placed finger can't be washed out by four good ones):

$$
d_{\text{reach},i} = \big\lVert \mathbf p_{\text{tip},i} - \mathbf c^\star_{i,t} \big\rVert
$$

$$
r_{\text{grasp\_reach}} = \frac{1}{5}\sum_{i=0}^{4}\left(1 - \tanh\frac{d_{\text{reach},i}}{\sigma_{\text{reach}}}\right), \qquad \sigma_{\text{reach}} = 0.05\text{ m}
$$

Dense and **ungated** — like `grasp_goal_palm`, it should pay from anywhere in the
workspace, since "move each fingertip toward its own target point" is meaningful at any
distance (it's a per-finger refinement of the existing §5.1.2 reach reward, which only
uses one shared centroid target for all 5 tips).

**Plain uniform mean over the 5 fingertips — no thumb/finger split.** `compute_task_reward`'s
`grasp_rew` (§5.1.3) splits 0.5-thumb/0.5-fingers because it compares distance to *one
shared target* (the cube center), where a badly-placed thumb could get diluted by four
decent fingers averaged together before the `tanh`. That risk doesn't apply here: each
finger already has its **own unique target point**, and each is passed through `tanh`
*individually* before averaging (mean-of-tanh, not tanh-of-mean) — so a badly-placed
thumb still shows up as its own near-zero term regardless of the other four. The thumb's
special importance is already encoded in *where* its target point sits (the optimizer
chose it as the opposing contact for force closure); weighting it again here would
double-count that.

**Weight**: $w_{\text{grasp\_reach}} = 1.0$ — same order as `grasp_goal_hand`, since it
plays a similar complementary role.

$$
r_{\text{grasp\_reach}} \cdot w_{\text{grasp\_reach}} \ \text{ (proposed addition to §5.4's total)}
$$

---

### 5.3 Penalty terms

| Term | Weight | Formula | Purpose |
|---|---:|---|---|
| `action_smoothness` | $-3.0$ | $0.005\sum_i(a_i-a_i^{\text{prev}})^2 + 0.001\sum_j \dot q_j^2$ | penalizes jerky actions and joint velocity |
| `fingertip_impact` | $-2.0$ | $\dfrac{1}{6}\sum_{k=1}^{6}\tanh\!\big(\lVert\Delta \mathbf{v}_{\text{tip},k}\rVert/3.0\big)$ | penalizes sudden hand-body acceleration (slam/jab) |
| `distractor_accel` | $-3.0$ | $\dfrac{1}{10}\sum_{d=1}^{10}\tanh\!\big(\lVert\Delta \mathbf{v}_d\rVert/2.0\big)$ | penalizes bumping/knocking distractors |
| `distractor_off_tray` | $-10.0$ | $\sum_{d=1}^{10}\mathbb{1}[z_d < 0.835]$ | per-step, accumulates while any distractor sits off the tray |
| `distractor_drop` | $-100.0$ | $\mathbb{1}[\exists\, d: z_d < 0.600]$ | one-time; episode also terminates |

### 5.4 Full per-step reward

$$
R = 1.0\, r_{\text{task}} \;+\; 2.0\, r_{\text{palm}} \;+\; 1.0\, r_{\text{hand}} \;+\; 1.0\, r_{\text{grasp\_reach}}
\;-\; 3.0\, p_{\text{smooth}} \;-\; 2.0\, p_{\text{impact}} \;-\; 3.0\, p_{\text{accel}}
\;-\; 10.0\, p_{\text{off\_tray}} \;-\; 100.0\, p_{\text{drop}}
$$

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
