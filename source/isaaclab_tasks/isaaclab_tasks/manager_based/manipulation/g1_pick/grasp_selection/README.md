# Grasp-selection optimizer

Picks **which** of the 32 UltraDexGrasp/BODex candidate grasps the policy should mimic,
replacing the hand-chosen `fixed_grasp_idx`. Implements **Version 2 (sphere-contact-gated
FSWO)** of [`../OPTIMIZER_IMPLEMENTATION_SPEC.md`](../OPTIMIZER_IMPLEMENTATION_SPEC.md).

CPU-only and offline: no Isaac Lab, no Isaac Sim, no GPU. Runs in ~1 s for 32 grasps.

## Result on the current library

```
BEST GRASP INDEX: 23     (= pool grasp #132)
  4 fingers (thumb,index,middle,ring), 9 contacting spheres, FSWO -1.4e-04
  3.0 cm lateral, 17.0 cm above the cube, 36.0 deg from straight down
  valid: 2/32 grasps passed the contact + force-closure + penetration gates
```

This is the top-down library the working policy (`logs/rsl_rl/g1_pick/2026-07-23_21-14-35/
model_7998.pt`) was trained against, so the optimizer's pick and the deployed policy agree.
The same grasp (pool #132) wins if you re-select from the full 300-grasp pool with the
top-down + tray + penetration gates — see `../grasp_sampler/build_goal_library.py`.

Runner-up is #18 (= pool #3), which has zero fingertip penetration; it wins if
`--max-penetration` is tightened to 2 mm or less. Both are top-down and compatible with
the reward function as-is.

## Files

| File | Role |
|---|---|
| `fswo.py` | Frictionless Self-balancing Wrench Optimizer — the force-closure score (spec §4). |
| `hand_model.py` | URDF FK, cube SDF + face projection, collision-sphere and library loading (spec §1–3, §6.1). |
| `rank_grasps_sphere_fswo.py` | **Version 2 selector** (spec §6) + CLI; writes `scores.json`. |
| `select_optimal_grasp.py` | Version 1 (fingertip-only FSWO) — reference/comparison only, do not ship (spec §5, §9). |
| `selftest.py` | Sanity + correctness checks (spec §3, §4.4, §4.5, §8). |
| `scores.json` | Cached optimizer output: `best_idx`, full ranking, per-grasp stats. |

## Usage

```bash
cd source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/g1_pick/grasp_selection

python selftest.py                                  # verify the math
python rank_grasps_sphere_fswo.py                   # rank + refresh scores.json
python rank_grasps_sphere_fswo.py --sweep           # robustness of the top pick
python rank_grasps_sphere_fswo.py --compare-v1      # what the fingertip-only version picks
python select_optimal_grasp.py                      # Version 1 in detail (its failure mode)
```

Re-run `rank_grasps_sphere_fswo.py` whenever the grasp library is re-synthesized.

## How the selection works

For each candidate grasp (the **grasp** stage, index 1, of the raw `(N,1,3,13)` library —
**no `T_usdbase_urdfbase` calibration**, which belongs to the online reward only):

1. **FK the whole hand** at the grasp's 6 actuated joint angles and place it in the cube
   frame via the root-link pose. The URDF's coupled slave joints are baked in as `fixed`,
   so the 6 angles fully determine the hand shape.
2. **Sample ~41 collision spheres** (from `inspire_right.yml`) across 13 links instead of
   just the 5 fingertips. A sphere touches when `sdf(center) - radius <= 5 mm`, so finger
   *thickness* is accounted for.
3. **Gate**: keep the grasp only if ≥ 3 distinct fingers touch **and** its FSWO score on
   the near contacts is ≥ −0.05 (genuinely force-closable). Contacts scattered over
   non-opposing faces score ≈ −1 and are discarded — that rejection is what makes the
   selector prefer a real wrap over a hand resting on the cube.
4. **Rank** the survivors by wrap quality `(n_fingers, n_spheres, fswo)` descending: most
   fingers engaged, then most finger-surface on the cube, FSWO as the final tie-break.

The palm's `hand_base_link` spheres count as contacts but never toward the finger gate.

## Integration

`g1_pick_env_cfg.py` reads the index at import:

```python
_OPTIMAL_GRASP_IDX = get_optimal_grasp_idx()          # cached scores.json, else recompute
...
sample_grasp_goal = EventTerm(..., params={..., "fixed_grasp_idx": _OPTIMAL_GRASP_IDX, ...})
```

`get_optimal_grasp_idx()` prefers the cached `scores.json` and only recomputes when the
grasp library is newer than the cache. The cache matters because the BODex assets it needs
(`ultradex_repo/…` URDF + sphere YAML) are **gitignored**, so a fresh checkout can still
train. Override with the env var `G1_PICK_GRASP_IDX=23` to reproduce an older checkpoint.

Nothing else changes: the MDP, observations, actions, rewards and the `T_usdbase_urdfbase`
handling in `mdp/grasp_goal.py` are untouched — only *which* library entry is the goal.

## Implementation notes (deviations from the spec, and why)

1. **FK without `pytorch-kinematics`.** The spec's reference uses it; this package ships an
   equivalent numpy chain (`hand_model.HandChain`) instead, so no new dependency lands in
   the Isaac Lab conda env. `selftest.py` cross-checks it against the URDF FK already used
   by `grasp_sampler/check_grasps_offline.py` — fingertip positions agree to 0.0 m exactly.

2. **FSWO scores the NNLS residual, not `alpha^T Q alpha`.** The two are identical in exact
   arithmetic, but on a near-degenerate contact set NNLS can return `|alpha| ~ 1e11` along a
   null direction of `Q`, and the quadratic form then cancels catastrophically. The spec's
   reference implementation hits this on the current library: **grasp #18 scored `+3.5e7`**,
   a positive value the score's own definition (`S <= 0`) forbids, which corrupted the
   tie-break ordering. `||L alpha||^2` — which `scipy.optimize.nnls` already returns as its
   residual — is a sum of squares, so it stays non-negative and accurate (#18 → `-2.1e-33`).
   The `L`-vs-`Q` design-matrix requirement of spec §4.4 is unchanged and is regression-tested
   against a brute-force SLSQP solve.

3. **A fingertip-penetration gate was added (`MAX_TIP_PENETRATION`, default 3 mm)** — not in
   the spec. The spec's gate is `gap <= contact_dist` with `gap` allowed to be arbitrarily
   negative, so a finger buried inside the cube counts as a contact; worse, deeper
   penetration puts MORE spheres inside, inflating the `n_spheres` that drives the ranking.
   Without the gate the optimizer picked a grasp whose middle fingertip sat **7.6 mm inside**
   the cube. The gate measures the zero-size `*_tip` links, which track the real mesh far
   better than the deliberately inflated collision spheres. Disable with
   `--max-penetration -1` to reproduce the spec exactly.

   The threshold is a judgement call and it decides the winner:

   | limit | 0 mm | 2 mm | **3 mm** | 5 mm | off |
   |---|---|---|---|---|---|
   | best | 18 | 18 | **23** | 23 | 17 (penetrating) |

   The top-down family holds several near-equivalent four-finger wraps; treat #23 and #18 as
   interchangeable on grasp quality alone.

4. **The top pick also moves somewhat with `contact_dist`** (`--sweep`), for the same reason:
   the leaders are separated by one or two contacting spheres. Don't read significance into
   the exact winner.
