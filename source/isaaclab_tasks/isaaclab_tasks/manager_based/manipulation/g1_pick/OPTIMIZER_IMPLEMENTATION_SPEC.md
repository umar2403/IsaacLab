# Grasp-Selection Optimizer — Complete Implementation Specification

**Purpose of this document.** It is a *self-contained, from-scratch* specification of the
grasp-selection optimizer that replaces a **human-chosen grasp pose** with an
**automatically-selected one** in the `g1_pick` (Unitree G1 + Inspire Hand) task. It is
written so that an implementer who has the g1_pick task + a working UltraDexGrasp/BODex
pipeline (which produces a *library* of candidate grasps) — but no grasp-*selection*
logic — can reproduce this optimizer exactly, down to every equation and variable.

The optimizer's job, in one sentence:

> Given a library of `N` candidate grasps (each = a hand root pose + 6 finger-joint
> angles, expressed in the object/cube frame) and the hand's URDF, compute a **single
> best grasp index** (and a full ranking) by how well each grasp mechanically wraps and
> force-closes the cube — then feed that index wherever your pipeline currently reads
> the human-chosen grasp.

There are **two versions** of the contact model, both documented here:

1. **Fingertip FSWO** (the original) — 5 contact points (the fingertips).
2. **Sphere-contact-gated FSWO** (the improved one you should implement) — ~40 contact
   points sampled along the whole hand via the URDF collision spheres, with a proximity
   gate so grasps whose fingers do not actually reach the cube are rejected.

Version 2 is the recommended target. Version 1 is documented because Version 2 reuses its
FSWO core verbatim, and because Version 1 has a **known failure mode** (Section 9) that
motivated Version 2.

---

## 0. Notation and conventions

- Vectors are column vectors; `·` is the dot product; `×` is the 3-D cross product.
- `‖x‖` is the Euclidean 2-norm.
- Quaternions are **`[w, x, y, z]` (scalar-first)** unless stated otherwise.
- All geometry for a single grasp is in the **cube frame**: the cube is an
  **axis-aligned cube centered at the origin**, with **half-edge `h = CUBE_HALF_EDGE = 0.025 m`**
  (a 5 cm cube). This is the frame BODex/UltraDex already expresses its output in.
- **No calibration transform is applied inside the optimizer.** (Your online reward code
  may apply a URDF↔sim `T_usdbase_urdfbase` correction; the optimizer must **not** —
  the "axis-aligned cube at the origin" assumption only holds in the raw synthesis frame.)

### Dependencies

```
numpy
scipy               # scipy.optimize.nnls, scipy.spatial.transform.Rotation
pytorch-kinematics  # CPU-only URDF forward kinematics  (pip install pytorch-kinematics)
torch               # pulled in by pytorch-kinematics
pyyaml              # to read the collision spheres from the robot config
```

This is a **CPU-only, offline** computation. It does **not** import Isaac Lab / Isaac Sim
and does not need a GPU. Run it once whenever the grasp library changes.

---

## 1. Inputs

### 1.1 The grasp library (from UltraDex/BODex synthesis)

A NumPy `.npz` (in this repo: `grasp_dataset/cube_5cm_grasps_valid.npz`) with:

- `grasp_pose`: `float32` array of shape **`(N, 1, 3, 13)`**
  - axis 0: `N` grasps (here `N = 32`).
  - axis 1: hand index (always `1` hand → size 1).
  - axis 2: **stage** ∈ {`0 = pregrasp`, `1 = grasp`, `2 = squeeze`}. **Use stage `1`.**
  - axis 3: the **13-vector** per (grasp, hand, stage):

    ```
    index:  0  1  2   3   4   5   6    7          8            9      10      11    12
    value:  x  y  z   qw  qx  qy  qz   thumb_yaw  thumb_pitch  index  middle  ring  pinky
            └─ pos ─┘ └──── quat ────┘ └──────────── 6 joint angles (rad) ────────────┘
    ```

  - `(x, y, z)` + `(qw, qx, qy, qz)` = the pose of the hand **root link** (the URDF's
    `"base"` link) **in the cube frame**.
  - The 6 joint angles are the **actuated (policy-controlled) joints**, in the fixed order
    `[thumb_proximal_yaw, thumb_proximal_pitch, index_proximal, middle_proximal,
    ring_proximal, pinky_proximal]`.
- `T_usdbase_urdfbase`: `(4, 4)` — a calibration transform used **only** by the online
  reward. **The optimizer ignores it.**

> **Porting note.** If your repo stores the library differently, map it to this schema
> first. The only things the optimizer needs per grasp are: the **root pose in the cube
> frame** `(pos, quat_wxyz)` and the **6 actuated joint angles**, from the **grasp stage**.

### 1.2 The hand URDF

The Inspire-hand URDF that BODex/UltraDex synthesized against. Key facts about it (verify
against your file):

- Root link is a zero-size **`"base"`** link (constant `URDF_ROOT_LINK = "base"`).
- The 6 actuated joints are named (bare, no `R_`/`L_` prefix):

  ```python
  ACTUATED_JOINT_NAMES = [
      "thumb_proximal_yaw_joint",
      "thumb_proximal_pitch_joint",
      "index_proximal_joint",
      "middle_proximal_joint",
      "ring_proximal_joint",
      "pinky_proximal_joint",
  ]
  ```
- The **coupled "slave" joints are baked in as `type="fixed"`** at the empirically-measured
  postures (thumb intermediate/distal ≈ −0.16/−0.24 rad; finger intermediates ≈ 1.15 rad).
  So the 6 actuated angles fully determine the hand shape via FK — no mimic handling needed.
- Fingertip links exist as zero-size `*_tip` links:
  `FINGERTIP_LINKS = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]`
  (policy order [thumb, index, middle, ring, pinky]).

### 1.3 The collision spheres (Version 2 only)

The hand's ~40 collision spheres, read from the robot config YAML BODex uses (in this repo:
`.../content/configs/robot/inspire_right_sim2real.yml`), under
`robot_cfg → kinematics → collision_spheres`. Structure:

```yaml
collision_spheres:
  hand_base_link:                         # link name (must exist in the URDF)
    - "center": [-0.0032, -0.0380, -0.0093]   # sphere center in the LINK's local frame (m)
      "radius": 0.01010                        # sphere radius (m)
    - "center": [-0.0032, -0.0380, 0.0000]
      "radius": 0.01010
    # ... several spheres per link ...
  thumb_proximal_base:
    - "center": [-0.0039, 0.0004, -0.0073]
      "radius": 0.00760
  # ... etc for every finger link ...
```

The sphere-bearing links (13 of them) and the finger each belongs to:

```python
LINK_FINGER = {
    "thumb_proximal_base": "thumb", "thumb_proximal": "thumb",
    "thumb_intermediate":  "thumb", "thumb_distal":   "thumb",
    "index_proximal":  "index",  "index_intermediate":  "index",
    "middle_proximal": "middle", "middle_intermediate": "middle",
    "ring_proximal":   "ring",   "ring_intermediate":   "ring",
    "pinky_proximal":  "pinky",  "pinky_intermediate":  "pinky",
    # "hand_base_link" (the palm) also carries spheres but is NOT counted as a finger.
}
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
```

> The sphere centers are the *original definition of the hand's surface geometry* — that
> is exactly why we sample them: they give multiple points spread along every phalanx, for
> free, in each link's local frame.

### 1.4 Reference constants

```python
CUBE_HALF_EDGE = 0.025            # m; cube edge = 0.05 m
URDF_ROOT_LINK = "base"
STAGE_GRASP    = 1                # 0=pregrasp, 1=grasp, 2=squeeze
# Version-2 gate defaults:
CONTACT_DIST   = 0.005           # m; a sphere counts as "touching" if surface within 5 mm
MIN_FINGERS    = 3               # need >=3 distinct fingers touching to keep a grasp
FC_GATE        = -0.05           # discard grasps whose best FSWO score < this (not force-closable)
LAMBDA         = 1.0             # FSWO torque-vs-force weight
```

---

## 2. Forward kinematics (both versions)

Build the FK chain **once** from the URDF (float64 for numerical stability), then evaluate
per grasp. `pytorch_kinematics` gives, for a vector of joint angles, the 4×4 pose of every
link relative to the root `"base"`.

```python
import numpy as np, torch, pytorch_kinematics as pk
from scipy.spatial.transform import Rotation as R

def load_chain(urdf_path):
    with open(urdf_path, "rb") as f:
        return pk.build_chain_from_urdf(f.read()).to(dtype=torch.float64)

def link_transforms(chain, joint_q):
    """Return {link_name: 4x4 pose relative to the root 'base' link}, at the given
    6 actuated joint angles (frozen slave joints are already fixed in the URDF)."""
    jnames = chain.get_joint_parameter_names()
    th = torch.zeros(len(jnames), dtype=torch.float64)
    for i, n in enumerate(jnames):
        if n in ACTUATED_JOINT_NAMES:
            th[i] = float(joint_q[ACTUATED_JOINT_NAMES.index(n)])   # match by ORDER
    tf = chain.forward_kinematics(th)
    return {name: tf[name].get_matrix()[0].numpy() for name in tf}   # each 4x4, root-relative
```

The **root pose** (hand base in the cube frame) as a 4×4 homogeneous matrix:

```python
def palm_matrix(pos, quat_wxyz):
    T = np.eye(4)
    T[:3, :3] = R.from_quat(np.asarray(quat_wxyz, float), scalar_first=True).as_matrix()
    T[:3, 3]  = np.asarray(pos, float)
    return T
```

A point `p_local` fixed in link `L`'s frame maps to the **cube frame** as

```
p_cube = T_palm · T_L · [p_local; 1]         # T_L = link_transforms(...)[L]   (root-relative)
```

For a **fingertip** (zero-size `*_tip` link), `p_local = 0`, so `p_cube = (T_palm · T_L)[:3,3]`.

---

## 3. Cube geometry primitives

### 3.1 Signed distance from a point to the cube surface

For an axis-aligned cube of half-edge `h` centered at the origin, the signed distance of a
point `p = (px, py, pz)` to the surface (negative inside):

```
d = |p| − h            (componentwise:  d_a = |p_a| − h,  a ∈ {x,y,z})
sdf(p) = ‖max(d, 0)‖ + min(max_a d_a, 0)
```

```python
def cube_surface_dist(p, h=CUBE_HALF_EDGE):
    d = np.abs(p) - h
    outside = float(np.linalg.norm(np.clip(d, 0.0, None)))   # distance if the point is outside
    inside  = float(min(np.max(d), 0.0))                     # negative penetration if inside
    return outside + inside
```

- `sdf(p) > 0`: point is outside, value = Euclidean distance to the nearest surface.
- `sdf(p) ≤ 0`: point is on/inside the cube, value = −(penetration depth).

### 3.2 Projecting a point onto the cube surface → a frictionless contact

Snap `p` to the nearest cube **face** and return the 6-vector `[contact_position(3),
inward_unit_normal(3)]`. The dominant axis (largest `|p_a| / h`) selects the face; the
inward normal points **into** the cube (opposite the outward face normal):

```python
def cube_contact(p, h=CUBE_HALF_EDGE):
    k = int(np.argmax(np.abs(p) / h))         # which face: axis of largest |coordinate|
    contact = p.copy()
    contact[k] = np.sign(p[k]) * h            # snap that coordinate onto the face
    normal = np.zeros(3)
    normal[k] = -np.sign(p[k])                # unit inward normal (into the cube)
    return np.concatenate([contact, normal])  # shape (6,) = [cx, cy, cz, nx, ny, nz]
```

Because the cube is axis-aligned, every contact normal is one of ±e_x, ±e_y, ±e_z.

---

## 4. The FSWO force-closure score (the "optimizer" core — identical in both versions)

**FSWO = Frictionless Self-balancing Wrench Optimizer** (the force-closure stage of
"Lightning Grasp", Yin & Abbeel, arXiv:2511.07418, Eq. 1). It scores **one grasp** given
its `k` contacts `{(p_i, n_i)}` (contact position `p_i`, **inward** unit normal `n_i`, both
in the cube frame). Higher (closer to 0) = better; `0` = perfect frictionless force closure.

### 4.1 The physics

A **frictionless** contact `i` can exert force only along its inward normal, with magnitude
`α_i ≥ 0`. The **wrench** (force + torque about the origin) of unit-magnitude contact `i` is

```
force:   n_i
torque:  τ_i = p_i × n_i
```

The total wrench of the grasp under contact magnitudes `α = (α_1, …, α_k)` is
`Σ_i α_i [n_i; τ_i]`. Force closure asks whether the contacts can produce a self-balancing
internal load: `α_i ≥ 0`, **not all zero**, with

```
Σ_i α_i n_i = 0        (force balance)
Σ_i α_i τ_i = 0        (torque balance)
```

To make "not all zero" well-posed we **normalize by the max**: `max_i α_i = 1`.

### 4.2 The relaxed QP and its Gram matrix

Minimize the squared residual wrench. Stack, with a torque weight `λ ≥ 0`, the 6-vector
`w_i = [n_i ; √λ · τ_i]` and `W = [w_1 … w_k] ∈ ℝ^{6×k}`. Then

```
‖Σ_i α_i w_i‖² = αᵀ (Wᵀ W) α = αᵀ Q α
```

with the **k×k Gram matrix**

```
Q_ij = n_i · n_j  +  λ (τ_i · τ_j)
```

Equivalently, with `N = [n_1 … n_k]ᵀ ∈ ℝ^{k×3}` and `T = [τ_1 … τ_k]ᵀ ∈ ℝ^{k×3}`:

```
Q = N Nᵀ + λ (T Tᵀ)          # (k, k), symmetric PSD by construction
```

`Q` is PSD because it is a Gram matrix of the stacked vectors `w_i`.

The optimization is

```
minimize   αᵀ Q α
subject to α_i ≥ 0,  max_i α_i = 1
```

and the **score** is

```
S = − (minimum of αᵀ Q α)  ∈ (−∞, 0].     S = 0  ⇔  perfect force closure.
```

- `S = 0`: there exist `α_i ≥ 0` (max 1) making the total wrench exactly zero → force
  closure is achievable with these contacts.
- `S < 0`: the best achievable residual wrench is nonzero → **not** force-closable with
  these contact positions/normals. (E.g. 3 contacts on the flat faces of a cube have
  axis-aligned normals that cannot cancel → `S ≈ −1`.)

**Meaning of `λ`.** It trades force balance against torque balance. On a small object the
torque arms `p_i` are numerically small, so torque terms are naturally down-weighted;
`λ = 1` is the Lightning-Grasp default. Raise `λ` (e.g. 10) to penalize rocking/torque
imbalance more.

### 4.3 Handling `max_i α_i = 1` (the nonconvex constraint)

`max_i α_i = 1` is nonconvex. Handle it by enumerating which contact attains the max: for
each `j ∈ {1…k}`, **pin `α_j = 1`** and minimize over the rest `α_{−j} ≥ 0`; take the best
over all `j`. Pinning at least one `α` to 1 also **excludes the trivial `α = 0`** solution
(which would give `αᵀQα = 0` for every grasp and make the score meaningless).

### 4.4 The correct NNLS reduction — **use the matrix square root `L`, not `Q`**

For fixed `j`, `minimize_{α_{−j} ≥ 0} αᵀ Q α` is a nonnegative least-squares problem — **but
only if the design matrix is a square root of `Q`**, not `Q` itself. Compute a symmetric
PSD square root `L` with `Q = Lᵀ L = L L` (so `L` symmetric):

```python
def psd_sqrt(Q, eps=1e-12):
    w, V = np.linalg.eigh(Q)              # Q symmetric PSD -> real eigendecomposition
    w = np.clip(w, eps, None) - eps       # clip tiny/negative (numerical) eigenvalues to 0
    return V @ np.diag(np.sqrt(w)) @ V.T  # symmetric sqrt: L = V diag(sqrt w) V^T
```

Then `αᵀ Q α = αᵀ Lᵀ L α = ‖L α‖²`, and with `α_j = 1` fixed:

```
‖L α‖² = ‖ L[:, −j] · α_{−j} + L[:, j] · 1 ‖²
       = ‖ L[:, −j] · α_{−j} − ( −L[:, j] ) ‖²
```

which is solved by `NNLS(A = L[:, −j], b = −L[:, j])` (scipy `nnls` solves
`min_{x≥0} ‖A x − b‖²`).

> **Why not `Q` directly.** Recasting `min αᵀQα` as `min ‖Q[:,−j] α_{−j} + Q[:,j]‖²` uses
> `Q` as the design matrix, whose normal equations involve `QᵀQ = Q²`, **not** `Q`. That
> changes the KKT/complementary-slackness conditions and gives a *wrong* (worse-than-true)
> optimum **whenever some `α_i` is pinned to 0 at the true optimum** — verified numerically
> against a brute-force constrained SLSQP solver on random contact sets. Only `L` (with
> `Q = LᵀL`) makes `‖Lα‖² = αᵀQα` an exact identity, so NNLS's complementary slackness on
> the residual `Lα` matches the QP's on `Qα`. **This is the single most important
> correctness detail in the whole optimizer — do not use raw `Q`.**

### 4.5 Reference implementation (drop-in, unchanged from the working repo)

```python
import numpy as np
from scipy.optimize import nnls

def psd_sqrt(Q, eps=1e-12):
    w, V = np.linalg.eigh(Q)
    w = np.clip(w, eps, None) - eps
    return V @ np.diag(np.sqrt(w)) @ V.T

def fswo_score(contacts, lam=1.0):
    """contacts: (k, 6) rows [px,py,pz, nx,ny,nz] (inward unit normals), cube frame.
       returns S in (-inf, 0]; 0 = perfect frictionless force closure."""
    if contacts.shape[0] < 2:
        raise ValueError(f"FSWO needs >= 2 contacts, got {contacts.shape[0]}.")
    p, n = contacts[:, :3], contacts[:, 3:]
    tau  = np.cross(p, n)                         # (k,3) torque of unit force at each contact
    Q    = n @ n.T + lam * (tau @ tau.T)          # (k,k) Gram matrix, PSD
    L    = psd_sqrt(Q)                            # Q = L @ L ; design matrix must be L (see 4.4)
    k = contacts.shape[0]
    best_val = np.inf
    for j in range(k):                            # enumerate which contact has alpha = 1
        mask = np.arange(k) != j
        alpha_free, _ = nnls(L[:, mask], -L[:, j])   # min_{x>=0} || L[:,~j] x + L[:,j] ||^2
        alpha = np.zeros(k); alpha[mask] = alpha_free; alpha[j] = 1.0
        val = float(alpha @ Q @ alpha)
        best_val = min(best_val, val)
    return -best_val
```

**Sanity checks for `fswo_score`:**
- Two antipodal contacts (`p_2 = −p_1`, `n_2 = −n_1`): `S = 0` (they cancel exactly).
- 3 contacts on 3 mutually orthogonal cube faces (normals `−e_x, −e_y, −e_z`): `S ≈ −1`
  (axis-aligned normals cannot sum to zero).
- Score is invariant to a global rotation/translation of all contacts (it is frame-free
  up to the torque origin, which is the cube center = origin here).

---

## 5. Version 1 — Fingertip FSWO (original; documented for completeness)

For each grasp, take the **5 fingertips**, project each onto the cube, feed to FSWO.

```python
def fk_tips(chain, palm_pos, palm_quat, joint_q):
    """(5,3) fingertip positions in the cube frame, order [thumb,index,middle,ring,pinky]."""
    Tp = palm_matrix(palm_pos, palm_quat)
    L  = link_transforms(chain, joint_q)
    return np.array([(Tp @ L[name])[:3, 3] for name in FINGERTIP_LINKS])

def build_contacts_tips(chain, grasp13):
    pos, quat, jq = grasp13[:3], grasp13[3:7], grasp13[7:]
    tips = fk_tips(chain, pos, quat, jq)                 # (5,3)
    return np.array([cube_contact(t) for t in tips])     # (5,6)

# selection: best_idx = argmax_i fswo_score(build_contacts_tips(chain, lib[i,0,1,:]))
```

Output of the original selector (written to `scores.json`):
`{"best_idx", "best_score", "ranked_indices", "scores", "lam"}`, with
`best_idx = argmax_i scores[i]`. The consumer sets its grasp index to `best_idx`.

**Known failure mode (why Version 2 exists).** `cube_contact` **snaps every fingertip onto
the nearest face and discards the gap** — a fingertip 27 mm off the cube is treated as a
genuine contact. So Version 1 scores the *arrangement* of contact points, never whether the
fingers actually reach the cube. It therefore ranks a **loose fingertip-hover** grasp as
"perfect" (`S ≈ 0`) over a real wrap. This actually happened (it picked a grasp with the
thumb 17 mm and pinky 27 mm off the surface). **Do not ship Version 1.**

---

## 6. Version 2 — Sphere-contact-gated FSWO (**implement this**)

Idea: instead of 5 fingertips, sample the **whole hand** via its ~40 URDF collision spheres
(multiple points per phalanx), keep only spheres whose surface actually reaches the cube,
require enough fingers to be touching, and score the near contacts with the same FSWO.

### 6.1 Per-sphere gap (accounts for finger thickness)

For a sphere with center `c` (link-local) and radius `r`, map the center to the cube frame
`c_cube = T_palm · T_L · [c;1]`, then the **effective gap** of the sphere surface to the
cube surface is

```
gap = sdf(c_cube) − r
```

`gap ≤ 0` ⇒ the sphere already overlaps the cube; `gap = CONTACT_DIST` ⇒ its surface is
`CONTACT_DIST` away. Keep spheres with `gap ≤ CONTACT_DIST`.

### 6.2 Validity gate (keeps the "fingers actually near the cube" requirement)

A grasp is **valid** iff:
1. at least `MIN_FINGERS` (= 3) **distinct fingers** (of thumb/index/middle/ring/pinky) have
   ≥ 1 contacting sphere, **and**
2. its FSWO score on the near contacts is `≥ FC_GATE` (= −0.05), i.e. genuinely
   force-closable (not a one/two-face touch).

Invalid grasps are **discarded** (never selected).

### 6.3 Contacts fed to FSWO

For every **near** sphere, project its center onto the cube (`cube_contact(c_cube)`) and
collect these as the contact set for `fswo_score`. (The palm's `hand_base_link` spheres are
included as contacts if near, but the palm is **not** counted toward the `MIN_FINGERS`
finger gate.)

### 6.4 Ranking metric

Once you gate on real contact, **FSWO saturates to ≈ 0 for essentially every valid grasp**
(finger spheres touching multiple faces ⇒ force closure is achievable). So FSWO can no
longer *rank* — it is the **validity gate**. Rank the valid grasps by **wrap quality**:

```
sort key (descending):  ( n_fingers_touching,  n_contacting_spheres,  fswo_score )
```

i.e. most fingers engaged first, then most finger-surface on the cube, then FSWO as a final
tie-break. `best_idx` = the top of this ranking.

> This deliberately rewards **enveloping** grasps (whole hand wraps the cube) over
> **pinch** grasps (thumb opposing 1–2 fingertips). For a compact object like a cube the
> envelope is the more robust, forgiving target — and it is the direct meaning of
> "the whole finger is near the cube".

### 6.5 Reference implementation (complete)

```python
import numpy as np, yaml
# reuse from above: load_chain, link_transforms, palm_matrix, cube_surface_dist,
#                    cube_contact, fswo_score, and the constants in Section 1.4.

def load_spheres(robot_yaml_path):
    cfg = yaml.safe_load(open(robot_yaml_path))["robot_cfg"]["kinematics"]["collision_spheres"]
    return {link: [(np.asarray(s["center"], float), float(s["radius"])) for s in lst]
            for link, lst in cfg.items()}

def score_grasp_spheres(chain, spheres, grasp13,
                        contact_dist=CONTACT_DIST, min_fingers=MIN_FINGERS,
                        fc_gate=FC_GATE, lam=LAMBDA):
    pos, quat, jq = grasp13[:3], grasp13[3:7], grasp13[7:]
    Tp = palm_matrix(pos, quat)
    L  = link_transforms(chain, jq)

    near_contacts, fingers_touch, n_near = [], set(), 0
    for link, lst in spheres.items():
        if link not in L:
            continue
        Tlw = Tp @ L[link]                                   # link pose in cube frame
        finger = LINK_FINGER.get(link)                       # None for the palm
        for center, radius in lst:
            c_cube = (Tlw @ np.append(center, 1.0))[:3]
            gap = cube_surface_dist(c_cube) - radius
            if gap <= contact_dist:                          # sphere surface reaches the cube
                near_contacts.append(cube_contact(c_cube))
                n_near += 1
                if finger is not None:
                    fingers_touch.add(finger)

    if len(fingers_touch) < min_fingers or len(near_contacts) < 2:
        return None                                          # invalid: not a real wrap
    S = fswo_score(np.array(near_contacts), lam=lam)
    if S < fc_gate:
        return None                                          # invalid: not force-closable
    return {"fswo": S, "n_fingers": len(fingers_touch), "n_spheres": n_near}

def select_best_grasp(library_npz, urdf_path, robot_yaml_path, **kw):
    chain   = load_chain(urdf_path)
    spheres = load_spheres(robot_yaml_path)
    gp = np.load(library_npz, allow_pickle=True)["grasp_pose"]   # (N,1,3,13)
    scored = []
    for i in range(gp.shape[0]):
        r = score_grasp_spheres(chain, spheres, gp[i, 0, STAGE_GRASP, :], **kw)
        if r is not None:
            scored.append((i, r))
    # rank by wrap: (n_fingers, n_spheres, fswo) descending
    scored.sort(key=lambda ir: (ir[1]["n_fingers"], ir[1]["n_spheres"], ir[1]["fswo"]),
                reverse=True)
    ranking  = [i for i, _ in scored]
    best_idx = ranking[0] if ranking else -1
    return best_idx, ranking, scored
```

**Empirically** on our 32-grasp library this selects an all-five-fingers **envelope wrap**
(our library's grasp #12) with the most finger-surface on the cube, whereas Version 1 (and
the human choice it mimicked) picked a loose fingertip hover. The top of the ranking is
stable across `contact_dist ∈ [5, 8] mm`.

---

## 7. Integration — replacing the human-chosen grasp

Wherever your pipeline currently reads a **human-chosen grasp index** (in this repo it is a
constant `_FIXED_GRASP_IDX` in `g1_pick_env_cfg.py`, consumed by the grasp-goal reset event
`SampleGraspGoal` which loads `grasp_pose[idx]` from the library), replace the source of
that integer with `best_idx` from `select_best_grasp(...)`.

Two equivalent ways:

1. **Offline (recommended, matches this repo):** run `select_best_grasp` once, print
   `best_idx`, and set the config constant to it (and/or write a small `scores.json` with
   `{"best_idx", "ranked_indices", ...}` for provenance). Training then reads that constant.
   *Nothing in the training loop changes*, so it is a drop-in swap for the manual pick.
2. **At load time:** call `select_best_grasp` inside your grasp-goal term's constructor and
   use the returned `best_idx` directly. Same result, no hand-edited constant.

**Critical**: the optimizer selects an index into the **same library** your synthesis
already produced, and it consumes the **raw cube-frame poses with no calibration applied**
(Section 0). Everything downstream (the reward's `T_usdbase_urdfbase` handling, the
observation/action spaces, the MDP) is **unchanged** — you are only changing *which* library
entry is chosen.

> Since your target repo has UltraDex fully working, the only new pieces are: `fswo.py`
> (Section 4.5), the sphere selector (Section 6.5), and one integer swap (this section).
> No change to the MDP, observations, actions, rewards, or the synthesis pipeline.

---

## 8. Frame-convention checklist (get these right or the scores are garbage)

1. Cube is **axis-aligned, centered at the origin, half-edge 0.025 m** — the frame the
   library's `(pos, quat)` are already in. Do **not** move the cube.
2. Use the **grasp stage** (`axis-2 index 1`), not pregrasp/squeeze.
3. `(pos, quat_wxyz)` is the **root `"base"` link** pose; quaternion is **scalar-first**.
4. FK link poses are **root-relative**; compose as `T_palm · T_link`.
5. Match the 6 joint angles to URDF joints **by the order in `ACTUATED_JOINT_NAMES`**, not
   by name (the URDF uses bare names, the policy may use `R_`-prefixed names).
6. Contact normals point **into** the cube.
7. **No `T_usdbase_urdfbase`** anywhere in the optimizer.
8. Sphere centers are in each **link's local frame** — transform by `T_palm · T_link`, then
   subtract the sphere **radius** when computing the gap.

---

## 9. Why the improved version matters (summary of the failure it fixes)

- **Bug:** projecting fingertips onto the cube (`cube_contact`) discards the gap, so FSWO
  scored a grasp with fingers *hovering off the cube* as perfect force closure. The single
  "far from surface" guard only fired on non-finite scores, which projection makes
  impossible — so it never fired.
- **Fix:** (a) sample the *whole hand* (collision spheres), (b) **gate** each contact by its
  real distance to the cube (surface gap, radius-aware), (c) require ≥ 3 fingers actually
  touching, and (d) rank by **wrap** (fingers + finger-surface on the cube), using FSWO as
  the force-closure validity gate. Result: the selector picks a genuine enveloping grasp.

---

## 10. Parameter reference

| Parameter | Symbol | Default | Meaning / effect |
|---|---|---|---|
| Cube half-edge | `h` | 0.025 m | Object size; used by `sdf` and `cube_contact`. |
| Grasp stage | — | 1 | Use the "grasp" stage of `(N,1,3,13)`. |
| Torque weight | `λ` | 1.0 | FSWO torque-vs-force balance; raise to punish rocking more. |
| Contact distance | — | 0.005 m | Max sphere-surface gap to count as touching. Loosen (0.008) to include near-contacts; tighten for stricter wraps. Ranking of the top grasp is stable across 5–8 mm. |
| Min fingers | — | 3 | Distinct fingers required to keep a grasp. |
| Force-closure gate | — | −0.05 | Discard grasps whose best FSWO < this. |
| `eps` (psd_sqrt) | — | 1e-12 | Clips tiny/negative eigenvalues of `Q`. |

---

*Reference implementations in the source repo (all CPU-only, no Isaac Lab):
`grasp_selection/fswo.py` (Section 4), `grasp_selection/select_optimal_grasp.py`
(Section 5), `grasp_selection/rank_grasps_sphere_fswo.py` (Section 6),
`check_grasps_offline.py` (Sections 1–3 helpers). An offline viewer that renders the real
Inspire-hand meshes on the cube for any grasp index is `visualize_grasp_offline.py`.*
