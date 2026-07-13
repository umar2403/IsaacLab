# G1 Pick — Complete Mathematical MDP Specification

---

## 1. MDP Tuple

The environment is a finite-horizon, continuous-state, continuous-action **Markov Decision Process**:

```
M = ( S, A, T, R, γ, ρ₀, D )
```

| Symbol | Meaning |
|--------|---------|
| S | State space ⊆ ℝ⁹⁶ |
| A | Action space ⊆ ℝ¹³ |
| T | Transition kernel p(s′ \| s, a) (implicit, via PhysX) |
| R | Reward function R : S × A × S → ℝ |
| γ = 0.99 | Discount factor |
| ρ₀ | Initial state distribution (reset distribution) |
| D | Termination predicate D : S × ℕ → {0,1} |

---

## 2. Time

| Quantity | Value |
|----------|-------|
| Physics timestep | Δt_sim = 1/120 s |
| Decimation | k = 4 (policy held constant for 4 sim steps) |
| Control timestep | Δt_ctrl = k · Δt_sim = 1/30 s |
| Max episode steps | T = 240 control steps (8 s) |
| Step index | t ∈ {0, 1, …, T} |

The policy receives a new observation and produces a new action every Δt_ctrl. The same joint position target is applied for k = 4 consecutive physics sub-steps.

---

## 3. State Space S ⊆ ℝ⁹⁶

The observation vector sₜ ∈ ℝ⁹⁶ is a concatenation of:

```
sₜ = [ q, q̇, p_obj, v_obj, p_d1, …, p_d10, p_tips, aₜ₋₁ ]
```

### 3.1 Proprioception

**Joint positions** (relative to default pose):

```
q = [ q_shoulder_pitch, q_shoulder_roll, q_shoulder_yaw,
      q_elbow,
      q_wrist_roll, q_wrist_pitch, q_wrist_yaw,          (7 arm)
      q_thumb_yaw, q_thumb_pitch,
      q_index_prox, q_middle_prox, q_ring_prox, q_pinky_prox ]  (6 hand)

q ∈ ℝ¹³
```

**Joint velocities** (clipped):

```
q̇ = clip( dq/dt, −50, +50 )    ∈ ℝ¹³
```

### 3.2 Exteroception — Target Object

**Position in robot root frame**:

```
p_obj = R_root^T · ( p_obj^w − p_robot^w )    ∈ ℝ³
```

where R_root is the rotation matrix of the robot root link, computed via `quat_apply_inverse`.

**Velocity in world frame** (linear + angular):

```
v_obj = [ v_lin^w, ω^w ]    ∈ ℝ⁶,   clipped ±50
```

### 3.3 Exteroception — Distractors

For each distractor i ∈ {1, …, 10}:

```
p_di = R_root^T · ( p_di^w − p_robot^w )    ∈ ℝ³
```

Concatenated: `[ p_d1, …, p_d10 ] ∈ ℝ³⁰`

### 3.4 Fingertip Positions

Six hand bodies B = { wrist_yaw_link, thumb_distal, index_intermediate,
middle_intermediate, ring_intermediate, pinky_intermediate }:

```
p_tips = concat_{ b ∈ B }[ R_root^T · ( p_b^w − p_robot^w ) ]    ∈ ℝ¹⁸
```

### 3.5 Action History

```
aₜ₋₁ ∈ ℝ¹³   (previous control step's raw action)
```

### 3.6 Dimension Summary

| Component | Dim |
|-----------|-----|
| q | 13 |
| q̇ | 13 |
| p_obj | 3 |
| v_obj | 6 |
| p_d1 … p_d10 | 30 |
| p_tips | 18 |
| aₜ₋₁ | 13 |
| **Total** | **96** |

---

## 4. Action Space A ⊆ ℝ¹³

The policy outputs raw actions **ã ∈ ℝ¹³**. These are scaled and applied as **joint position targets** relative to the default pose:

```
q_target^arm  = q_default^arm  + σ_arm  · ã_arm      σ_arm  = 0.3
q_target^hand = q_default^hand + σ_hand · ã_hand     σ_hand = 0.5
```

### 4.1 Action Components

| Index | Joint | Scale |
|-------|-------|-------|
| 0 | right_shoulder_pitch | 0.3 |
| 1 | right_shoulder_roll | 0.3 |
| 2 | right_shoulder_yaw | 0.3 |
| 3 | right_elbow | 0.3 |
| 4 | right_wrist_roll | 0.3 |
| 5 | right_wrist_pitch | 0.3 |
| 6 | right_wrist_yaw | 0.3 |
| 7 | R_thumb_proximal_yaw | 0.5 |
| 8 | R_thumb_proximal_pitch | 0.5 |
| 9 | R_index_proximal | 0.5 |
| 10 | R_middle_proximal | 0.5 |
| 11 | R_ring_proximal | 0.5 |
| 12 | R_pinky_proximal | 0.5 |

### 4.2 Mimic Joint Coupling (InspireMimicAction)

The policy controls only **6 proximal hand joints**. Six downstream joints are driven deterministically by hardware transmission ratios, reducing the effective hand DOF without loss of physical realism:

```
θ_thumb_intermediate  = 0.8024 · q_target[ thumb_pitch ]
θ_thumb_distal        = 0.9487 · θ_thumb_intermediate
                      = 0.7611 · q_target[ thumb_pitch ]

θ_index_intermediate  = 1.0843 · q_target[ index_prox ]
θ_middle_intermediate = 1.0843 · q_target[ middle_prox ]
θ_ring_intermediate   = 1.0843 · q_target[ ring_prox ]
θ_pinky_intermediate  = 1.0843 · q_target[ pinky_prox ]
```

These 6 mimic targets are written directly to the joint position target buffer after the main action is applied. The overall physical hand has 12 joints controlled by a 6-dimensional action.

### 4.3 Actuator Model (Implicit PD)

Torques are computed as implicit PD controllers inside PhysX:

```
τ = Kp · (q_target − q) − Kd · q̇
```

| Actuator group | Kp | Kd | Effort limit (sim) |
|---------------|----|----|-------------------|
| Arms (both) | 300 | 30 | 300 Nm |
| Right hand | 100 | 0.5 | 30 Nm |
| Left hand (frozen) | 10,000 | 1,000 | 10,000 Nm |
| Legs / waist (frozen) | 10,000 | 1,000 | 10,000 Nm |

The extreme Kp/Kd on left hand and legs makes them physically immovable without removing them from the articulation tree.

---

## 5. Reward Function R(s, a, s′)

The per-step reward decomposes as:

```
R(s, a, s′) = R_task(s′) 
            − 3.0 · P_smooth(s, a)
            − 2.0 · P_impact(s, s′)
            − 3.0 · P_dist_accel(s, s′)
            − 10.0 · P_off_tray(s′)
            − 100.0 · P_dist_drop(s′)
```

All quantities below are computed per environment and per step, returning scalars in ℝ^N for N parallel environments.

---

### 5.1 Task Reward R_task(s′)

Let the following shorthands be defined from state s′:

```
p_obj         := cube position in world frame                  ∈ ℝ³
p_palm        := right_wrist_yaw_link position                 ∈ ℝ³
p_tips        := [ p_thumb, p_index, p_middle, p_ring, p_pinky ]  ∈ ℝ^{5×3}
```

**Deadband distances** (contact surface offset 2.5 cm):

```
d_thumb  = max( ‖ p_thumb − p_obj ‖ − 0.025,  0 )
d_finger = max( mean_{i∈{index,middle,ring,pinky}} ‖ p_tipᵢ − p_obj ‖ − 0.025,  0 )
```

#### 5.1.1 Posture Reward

Palm should hover 8 cm directly above the cube:

```
p_palm_target  = p_obj + [0, 0, 0.08]

R_posture = 1 − tanh( max( ‖ p_palm − p_palm_target ‖ − 0.03,  0 ) / 0.3 )

Range: [0, 1]
```

The inner clamp gives a 3 cm tolerance dead-zone before the reward decays.

#### 5.1.2 Reach Reward

Centroid of all five fingertips toward the cube:

```
p_mid = (1/5) Σᵢ p_tipᵢ

R_reach = 1 − tanh( ‖ p_mid − p_obj ‖ / 0.25 )

Range: [0, 1]
```

#### 5.1.3 Grasp Reward

Thumb AND fingers must be close. Uses the deadband distances:

```
R_thumb  = 1 − tanh( d_thumb  / 0.055 )
R_finger = 1 − tanh( d_finger / 0.055 )

R_grasp  = 0.5 · R_thumb + 0.5 · R_finger

Range: [0, 1]
```

#### 5.1.4 Grasp Gate (soft AND)

Used to gate lift and success rewards — the policy must have both thumb and fingers engaged:

```
g = [ 1 − tanh( d_thumb  / 0.06 ) ] · [ 1 − tanh( d_finger / 0.06 ) ]

g ∈ [0, 1]   (≈1 only when both d_thumb ≈ 0 and d_finger ≈ 0)
```

#### 5.1.5 Lift Reward (continuous, gated)

Proportional to height gained above resting position, gated by grasp quality:

```
Δh = clamp( p_obj_z − 0.845,  0,  0.30 )      (meters lifted, max 30 cm)

R_lift = 2.0 · Δh · g

Range: [0, 0.6]
```

#### 5.1.6 Success Bonus (sparse, gated)

```
R_success = 𝟙[ p_obj_z > 1.134 ] · g · 1000.0

Range: {0, 1000}
```

Success height 1.134 m = 0.845 + 0.289 m, i.e., ~29 cm above the tray.

#### 5.1.7 Task Reward Assembly + Drop Override

```
R_task = R_posture + R_reach + 2 · R_grasp + R_lift + R_success

          ⎧ R_task      if  p_obj_z ≥ 0.600
R_task = ⎨
          ⎩ −0.5        if  p_obj_z <  0.600   (cube fell off table)
```

Expected range (no clutter): [−0.5, ~1005]  
Typical step reward during approach: [2, 4]  
Typical step reward during grasp: [4, 6]

---

### 5.2 Action Smoothness Penalty P_smooth(s, a)

```
P_smooth = α_rate · ‖ aₜ − aₜ₋₁ ‖²  +  α_vel · ‖ q̇ ‖²

α_rate = 0.005
α_vel  = 0.001
```

Applied with weight −3.0 in the full reward. Penalizes both jerk (action rate) and sustained joint velocity.

---

### 5.3 Fingertip Impact Penalty P_impact(s, s′)

Proxy for hard/sudden hand contact, computed as the normalized velocity change across 6 hand bodies:

```
Δv_b = v_b(s′) − v_b(s)          for each b ∈ B_hand

P_impact = (1/6) Σ_{b ∈ B_hand} tanh( ‖ Δv_b ‖ / 3.0 )

Range: [0, 1]
```

tanh saturates near 1 for `‖Δv‖ ≫ 3 m/s²` (violent slam). Gentle grasp approach yields ≈ 0.1.  
Applied with weight −2.0.

---

### 5.4 Distractor Acceleration Penalty P_dist_accel(s, s′)

Same principle applied to all 10 distractor objects:

```
Δv_dᵢ = v_dᵢ(s′) − v_dᵢ(s)       for i ∈ {1, …, 10}

P_dist_accel = (1/10) Σᵢ tanh( ‖ Δv_dᵢ ‖ / 2.0 )

Range: [0, 1]
```

Lower threshold (2.0 vs 3.0) makes distractors more sensitive than fingertips.  
Applied with weight −3.0.

---

### 5.5 Distractor Off-Tray Penalty P_off_tray(s′)

Per-step count of distractors that have fallen below the tray surface:

```
P_off_tray = Σᵢ 𝟙[ p_dᵢ_z < 0.835 ]

Range: {0, 1, …, 10}
```

Applied with weight −10.0, so each distractor off the tray costs −10/step. Accumulates for every step the distractor remains displaced — creating strong pressure to avoid tipping cubes. Episode does **not** terminate here.

---

### 5.6 Distractor Drop Penalty P_dist_drop(s′)

Sparse signal: fires if any distractor falls completely off the table:

```
P_dist_drop = 𝟙[ ∃i :  p_dᵢ_z < 0.600 ]

Range: {0, 1}
```

Applied with weight −100.0. Episode terminates simultaneously (see §6).

---

### 5.7 Full Reward Summary

```
R(s, a, s′) =  1.0 · R_task(s′)
             − 3.0 · [ 0.005 ‖Δa‖² + 0.001 ‖q̇‖² ]
             − 2.0 · (1/6) Σ_b tanh(‖Δv_b‖/3)
             − 3.0 · (1/10) Σᵢ tanh(‖Δv_dᵢ‖/2)
             − 10.0 · Σᵢ 𝟙[p_dᵢ_z < 0.835]
             − 100.0 · 𝟙[∃i: p_dᵢ_z < 0.600]
```

where `R_task` is overridden to −0.5 when the target cube falls below 0.600 m.

---

## 6. Termination Predicate D(s, t)

Episode terminates (d = 1) when **any** of the following holds:

```
D(s, t) = 1  iff

  (1)  t ≥ T = 240                          [ timeout — truncation ]
  (2)  p_obj_z > 1.134                      [ target lifted — SUCCESS ]
  (3)  p_obj_z < 0.600                      [ target dropped — failure ]
  (4)  ∃i ∈ {1,…,10} : p_dᵢ_z < 0.600     [ distractor dropped — failure ]
```

Condition (1) is a timeout (truncation, value bootstrap); conditions (2–4) are true terminal states. Only condition (2) yields the +1000 success bonus.

---

## 7. Initial State Distribution ρ₀

On each reset, states are sampled as:

### 7.1 Robot Joints

```
q_arm  ~ q_default^arm  + U(−0.05, +0.05)^7      (arm joints)
q_hand ~ q_default^hand + U(−0.05, +0.05)^6      (hand joints)
q_left = q_default^left                            (exact, frozen)
q_legs = q_default^legs                            (exact, frozen)
```

Default arm pose (in radians): shoulder_pitch = −0.583, shoulder_roll = −0.864, shoulder_yaw = 0.426, elbow = 0.370, wrist_roll = −0.410, wrist_pitch = 0.000, wrist_yaw = −0.103.

### 7.2 Target Object

```
p_obj ~ p_obj^anchor + [ U(−0.10, +0.10),  U(−0.05, +0.05),  0 ]

p_obj^anchor = [0.35,  0.0,  0.845]
```

### 7.3 Distractors (when active)

```
p_dᵢ ~ p_dᵢ^anchor + [ U(−0.03, +0.03),  U(−0.03, +0.03),  0 ]   if  δ ≥ θᵢ
p_dᵢ_z = −5.0  (hidden below table)                                  if  δ < θᵢ
```

Anchor positions and activation thresholds θᵢ (see §8).

---

## 8. Curriculum

The curriculum operates on two independent axes.

### 8.1 Per-Environment Clutter Difficulty δₑ

Each of the N = 4096 environments maintains a scalar difficulty δₑ ∈ ℤ, updated at every episode reset:

```
                ⎧ min(δₑ + 1, δ_max)    if  p_obj_z > 0.995   (= env_origin_z + 0.15)
δₑ ←  ⎨
                ⎩ max(δₑ − 1, δ_min)    otherwise

δ_min = 0,  δ_max = 60
```

(With `promotion_only = False`. If set to True, failures do not decrease difficulty.)

**Distractor activation schedule**: distractor i activates when `δₑ ≥ θᵢ`:

```
θᵢ = 30 + (i − 1) · 10     for i ∈ {1, …, 10}
```

| Distractor | Activation threshold θᵢ | Active distractors at δ = θᵢ |
|-----------|------------------------|------------------------------|
| 1 | 30 | 1 |
| 2 | 40 | 2 |
| 3 | 50 | 3 |
| … | … | … |
| 10 | 120 (unreachable at δ_max=60) | — |

At δ_max = 60, distractors 1–4 are active (θ₁…θ₄ ≤ 60), giving 4 on-tray obstacles maximum. Inactive distractors are placed at z = −5 m (5 m below the table, never contacted).

**Mean difficulty fraction** (logged as curriculum metric):

```
difficulty_frac = (1/N) Σₑ δₑ / δ_max    ∈ [0, 1]
```

### 8.2 Phase-Based Reward Gating φ ∈ {0, 1, 2}

A global training phase φ controls which reward terms are active. Transitions are triggered by rolling mean episode sums, computed over a sliding window of the last H = 500 completed episodes (minimum 50 required):

```
μ_reach  = mean over last H episodes of:  Σₜ R_reaching_target(t)  / max_ep_len
μ_lift   = mean over last H episodes of:  Σₜ R_lifting_target(t)   / max_ep_len
```

**Phase transition rules**:

```
φ: 0 → 1   when  μ_reach ≥ τ₁ = 0.4
φ: 1 → 2   when  μ_lift  ≥ τ₂ = 0.75
```

Transitions are **monotone** (irreversible).

**Reward weights per phase**:

| Reward Term | Phase 0 | Phase 1 | Phase 2 |
|------------|---------|---------|---------|
| reaching_target | active | active | active |
| lifting_target | 0 (off) | 5.0 | 5.0 |
| declutter | 0 (off) | 2.0 | 2.0 |
| pick_success | 0 (off) | 0 (off) | 1.0 |

This prevents reward interference: the agent first learns to reach, then to lift, then to complete a full pick.

---

## 9. Policy Optimization (PPO)

### 9.1 Policy Architecture

Stochastic diagonal Gaussian policy:

```
π_θ(a | s) = N( μ_θ(s),  diag(σ²) )
```

Both actor and critic are **MLPs** with layer sizes [512, 256, 128] and ELU activations. Observations are normalized via running mean/variance before input.

```
f_actor  : ℝ⁹⁶ → ℝ¹³    (mean of action distribution)
f_critic : ℝ⁹⁶ → ℝ       (value function estimate)

σ_init = 0.8   (initial exploration noise, learnable)
```

### 9.2 Return Estimation (GAE)

For a rollout of length L = 32 steps per environment, the advantage estimate uses Generalized Advantage Estimation:

```
δₜ = rₜ + γ · V(sₜ₊₁) · (1 − dₜ) − V(sₜ)

Âₜ = Σ_{k=0}^{L−t−1} (γλ)^k · δₜ₊ₖ

γ = 0.99,   λ = 0.95
```

where dₜ = 1 if episode terminates at step t (with value bootstrap = 0 for true terminals, V(sₜ₊₁) for truncations).

### 9.3 PPO Objective

The actor loss (maximize):

```
L_actor(θ) = E_t [ min( rₜ(θ) · Âₜ,  clip( rₜ(θ), 1−ε, 1+ε ) · Âₜ ) ]

rₜ(θ) = π_θ(aₜ | sₜ) / π_θ_old(aₜ | sₜ)    (importance ratio)
ε = 0.2
```

The critic loss (minimize):

```
L_critic(θ) = E_t [ ( Vₜ_target − V_θ(sₜ) )² ]

with clipped value loss enabled.
```

The entropy bonus (maximize):

```
L_entropy(θ) = E_t [ H[ π_θ(· | sₜ) ] ]
             = E_t [ (1/2) Σⱼ log(2πe σⱼ²) ]
```

Total loss (minimize):

```
L(θ) = − L_actor(θ)  +  c₁ · L_critic(θ)  −  c₂ · L_entropy(θ)

c₁ = 1.0   (value loss coefficient)
c₂ = 0.005 (entropy coefficient)
```

### 9.4 Update Schedule

```
Rollout length per env:     L = 32 steps
Total transitions per update: N · L = 4096 × 32 = 131,072
Mini-batches per update:    M = 4  → batch size = 32,768
Learning epochs per rollout: K = 5

Optimizer: Adam,  lr = 3×10⁻⁴  (adaptive schedule, target KL = 0.01)
Gradient clipping: ‖∇‖ ≤ 1.0
Max training iterations: 5000   (≈ 655 M environment steps)
```

**Adaptive learning rate rule**: after each epoch, the empirical KL divergence KL(π_θ_old ‖ π_θ) is measured. If KL > 2 · KL_target, lr is decreased; if KL < 0.5 · KL_target, lr is increased.

---

## 10. Key Constants Reference

| Symbol | Value | Meaning |
|--------|-------|---------|
| z_rest | 0.845 m | Target cube center at rest on tray |
| z_off_tray | 0.835 m | Below this → distractor_off_tray fires |
| z_success | 1.134 m | Above this → success termination |
| z_drop | 0.600 m | Below this → failure termination |
| z_hidden | −5.0 m | Inactive distractor teleport height |
| z_lift_curriculum | env_z + 0.15 m | Curriculum success threshold |
| σ_arm | 0.3 | Arm action scale |
| σ_hand | 0.5 | Hand action scale |
| Kp_arm | 300 | Arm position stiffness |
| Kd_arm | 30 | Arm velocity damping |
| Kp_hand | 100 | Hand position stiffness |
| Kd_hand | 0.5 | Hand velocity damping |
| Kp_frozen | 10,000 | Stiffness to freeze left arm / legs |
| Kd_frozen | 1,000 | Damping to freeze left arm / legs |
| deadband | 0.025 m | Contact radius subtracted from tip distances |
| grasp_std | 0.055 m | tanh scale for grasp reward |
| grasp_gate_std | 0.060 m | tanh scale for grasp gate g |
| α_rate | 0.005 | Action rate penalty scale |
| α_vel | 0.001 | Joint velocity penalty scale |
| accel_thresh_tips | 3.0 m/s | Fingertip Δv normalization |
| accel_thresh_dist | 2.0 m/s | Distractor Δv normalization |
| w_task | 1.0 | Task reward weight |
| w_smooth | −3.0 | Smoothness penalty weight |
| w_impact | −2.0 | Impact penalty weight |
| w_accel | −3.0 | Distractor accel penalty weight |
| w_off_tray | −10.0 | Per-distractor off-tray weight |
| w_dist_drop | −100.0 | Distractor drop penalty weight |
| γ | 0.99 | Discount factor |
| λ | 0.95 | GAE lambda |
| ε | 0.2 | PPO clip |
| c₁ | 1.0 | Value loss coefficient |
| c₂ | 0.005 | Entropy coefficient |
| lr | 3×10⁻⁴ | Base learning rate |
| N | 4096 | Parallel environments |
| L | 32 | Rollout steps per env |
| K | 5 | Epochs per PPO update |
| M | 4 | Mini-batches per epoch |
| T | 240 | Max episode control steps |
