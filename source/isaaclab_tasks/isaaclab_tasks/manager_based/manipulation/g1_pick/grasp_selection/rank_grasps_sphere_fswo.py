# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Version 2 — sphere-contact-gated FSWO grasp selection (spec Section 6).

Replaces the human-chosen grasp index with an automatically selected one:

1. sample the WHOLE hand via its ~40 URDF collision spheres (multiple points per
   phalanx) instead of just the 5 fingertips,
2. keep only spheres whose SURFACE actually reaches the cube (radius-aware gap),
3. require >= MIN_FINGERS distinct fingers touching and an FSWO score >= FC_GATE
   (force-closable) — everything else is discarded,
4. rank the survivors by wrap quality ``(n_fingers, n_spheres, fswo)`` descending.

Run as a script to (re)generate ``scores.json`` next to this file:

    python rank_grasps_sphere_fswo.py                 # rank + write scores.json
    python rank_grasps_sphere_fswo.py --top 10        # show more of the ranking
    python rank_grasps_sphere_fswo.py --compare-v1    # also show the Version-1 pick

CPU-only, offline, no Isaac Lab / Isaac Sim. Re-run whenever the grasp library changes.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os

import numpy as np

try:  # allow both `python rank_grasps_sphere_fswo.py` and package import
    from .fswo import fswo_score
    from .hand_model import (
        CUBE_HALF_EDGE,
        DEFAULT_LIBRARY,
        DEFAULT_ROBOT_YAML,
        DEFAULT_URDF,
        FINGERS,
        FINGERTIP_LINKS,
        LINK_FINGER,
        STAGE_GRASP,
        HandChain,
        cube_contact,
        cube_surface_dist,
        load_chain,
        load_library,
        load_spheres,
        palm_matrix,
    )
except ImportError:  # pragma: no cover - direct script execution
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fswo import fswo_score
    from hand_model import (
        CUBE_HALF_EDGE,
        DEFAULT_LIBRARY,
        DEFAULT_ROBOT_YAML,
        DEFAULT_URDF,
        FINGERS,
        FINGERTIP_LINKS,
        LINK_FINGER,
        STAGE_GRASP,
        HandChain,
        cube_contact,
        cube_surface_dist,
        load_chain,
        load_library,
        load_spheres,
        palm_matrix,
    )

__all__ = ["CONTACT_DIST", "MIN_FINGERS", "FC_GATE", "LAMBDA", "score_grasp_spheres", "select_best_grasp"]

# ---- Version-2 gate defaults (spec Section 1.4 / Section 10) --------------------------
CONTACT_DIST = 0.005  # m; a sphere counts as touching if its surface is within 5 mm
MIN_FINGERS = 3       # distinct fingers required for a grasp to stay in the running
FC_GATE = -0.05       # discard grasps whose FSWO score is below this (not force-closable)
LAMBDA = 1.0          # FSWO torque-vs-force weight
# Max fingertip penetration INTO the cube, in metres. NOT part of the spec: the spec's
# gate is `gap <= contact_dist`, where gap may be arbitrarily negative, so a finger
# buried deep inside the cube counts as a contact — and since deeper penetration puts
# MORE spheres inside, it inflates `n_spheres`, which drives the ranking. The result is
# a bias toward interpenetrating grasps (observed: a pick whose middle fingertip sat
# 7.6 mm inside the cube). Measured on the zero-size `*_tip` links, which track the real
# mesh far better than the deliberately inflated collision spheres. Set to None to
# disable and reproduce the spec's behaviour exactly.
MAX_TIP_PENETRATION = 0.003

SCORES_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scores.json")


def score_grasp_spheres(
    chain: HandChain,
    spheres: dict,
    grasp13: np.ndarray,
    contact_dist: float = CONTACT_DIST,
    min_fingers: int = MIN_FINGERS,
    fc_gate: float = FC_GATE,
    lam: float = LAMBDA,
    half_edge: float = CUBE_HALF_EDGE,
    max_tip_penetration: float | None = MAX_TIP_PENETRATION,
) -> dict:
    """Score one grasp (13-vector at the grasp stage) — spec Section 6.5.

    Returns a dict with the wrap statistics and ``valid`` / ``reject`` fields. An
    invalid grasp is never selected; the stats are kept for provenance/debugging.
    """
    pos, quat, jq = grasp13[:3], grasp13[3:7], grasp13[7:]
    Tp = palm_matrix(pos, quat)
    L = chain.link_transforms(jq)

    # fingertip penetration (see MAX_TIP_PENETRATION) — computed first so it is reported
    # even for grasps that are rejected for another reason
    tip_gaps = [cube_surface_dist((Tp @ L[t])[:3, 3], half_edge) for t in FINGERTIP_LINKS if t in L]
    tip_min = min(tip_gaps) if tip_gaps else float("nan")

    near_contacts: list[np.ndarray] = []
    fingers_touch: set[str] = set()
    per_finger: dict[str, int] = {f: 0 for f in FINGERS}
    faces: dict[str, int] = {}
    n_palm = 0
    min_gap = float("inf")

    for link, lst in spheres.items():
        if link not in L:
            continue
        Tlw = Tp @ L[link]  # link pose in the cube frame
        finger = LINK_FINGER.get(link)  # None for the palm (hand_base_link)
        for center, radius in lst:
            c_cube = (Tlw @ np.append(center, 1.0))[:3]
            gap = cube_surface_dist(c_cube, half_edge) - radius  # sphere SURFACE to cube surface
            min_gap = min(min_gap, gap)
            if gap <= contact_dist:
                contact = cube_contact(c_cube, half_edge)
                near_contacts.append(contact)
                # which cube face this contact sits on (outward = -inward normal)
                axis = int(np.argmax(np.abs(contact[3:])))
                face = f"{'-+'[int(contact[3 + axis] < 0)]}{'xyz'[axis]}"
                faces[face] = faces.get(face, 0) + 1
                if finger is not None:
                    fingers_touch.add(finger)
                    per_finger[finger] += 1
                else:
                    n_palm += 1

    result = {
        "n_fingers": len(fingers_touch),
        "n_spheres": len(near_contacts),
        "n_palm_spheres": n_palm,
        "fingers": sorted(fingers_touch, key=FINGERS.index),
        "spheres_per_finger": per_finger,
        "faces": faces,  # contacts per cube face; force closure needs OPPOSING faces
        "min_gap": None if min_gap == float("inf") else round(min_gap, 5),
        "tip_min_gap": round(tip_min, 5),
        "fswo": None,
        "valid": False,
        "reject": None,
    }

    if max_tip_penetration is not None and tip_min < -max_tip_penetration:
        result["reject"] = (f"fingertip {abs(tip_min) * 1e3:.1f} mm inside the cube "
                            f"(limit {max_tip_penetration * 1e3:.1f} mm)")
        return result

    if len(fingers_touch) < min_fingers or len(near_contacts) < 2:
        result["reject"] = f"only {len(fingers_touch)} finger(s) / {len(near_contacts)} sphere(s) touching"
        return result

    S = fswo_score(np.array(near_contacts), lam=lam)
    result["fswo"] = float(S)
    if S < fc_gate:
        opposed = any(f"-{a}" in faces and f"+{a}" in faces for a in "xyz")
        why = "no opposing faces touched" if not opposed else "residual wrench too large"
        result["reject"] = (f"not force-closable (FSWO {S:.4f} < {fc_gate}; {why}, "
                            f"faces {'/'.join(sorted(faces))})")
        return result

    result["valid"] = True
    return result


def _sort_key(entry: dict) -> tuple:
    """Wrap-quality ranking key (spec Section 6.4), descending."""
    return (entry["n_fingers"], entry["n_spheres"], entry["fswo"])


def select_best_grasp(
    library_npz: str = DEFAULT_LIBRARY,
    urdf_path: str = DEFAULT_URDF,
    robot_yaml_path: str = DEFAULT_ROBOT_YAML,
    stage: int = STAGE_GRASP,
    **kw,
) -> tuple[int, list[int], list[dict]]:
    """Select the optimal grasp from the library.

    Returns:
        ``(best_idx, ranked_indices, per_grasp)`` where ``per_grasp`` holds the full
        stats of EVERY library grasp (valid and rejected), indexed by library index.
        ``best_idx`` is ``-1`` if no grasp passes the gates.
    """
    chain = load_chain(urdf_path)
    spheres = load_spheres(robot_yaml_path)
    grasps = load_library(library_npz, stage=stage)

    per_grasp = []
    for i in range(grasps.shape[0]):
        entry = score_grasp_spheres(chain, spheres, grasps[i], **kw)
        entry["idx"] = i
        per_grasp.append(entry)

    valid = [e for e in per_grasp if e["valid"]]
    valid.sort(key=_sort_key, reverse=True)
    ranking = [e["idx"] for e in valid]
    best_idx = ranking[0] if ranking else -1
    return best_idx, ranking, per_grasp


def write_scores_json(
    path: str,
    best_idx: int,
    ranking: list[int],
    per_grasp: list[dict],
    library_npz: str,
    urdf_path: str,
    robot_yaml_path: str,
    params: dict,
) -> None:
    payload = {
        "version": 2,
        "method": "sphere-contact-gated FSWO (OPTIMIZER_IMPLEMENTATION_SPEC.md Section 6)",
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "best_idx": int(best_idx),
        "ranked_indices": [int(i) for i in ranking],
        "n_valid": len(ranking),
        "n_grasps": len(per_grasp),
        "params": params,
        "library": os.path.relpath(library_npz, os.path.dirname(path)),
        "urdf": os.path.relpath(urdf_path, os.path.dirname(path)),
        "robot_yaml": os.path.relpath(robot_yaml_path, os.path.dirname(path)),
        "ranking_key": "(n_fingers, n_spheres, fswo) descending",
        "scores": per_grasp,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--library", default=DEFAULT_LIBRARY, help="grasp library .npz")
    p.add_argument("--urdf", default=DEFAULT_URDF, help="Inspire hand URDF used by BODex synthesis")
    p.add_argument("--robot-yaml", default=DEFAULT_ROBOT_YAML, help="BODex robot config with collision_spheres")
    p.add_argument("--contact-dist", type=float, default=CONTACT_DIST, help="max sphere-surface gap [m]")
    p.add_argument("--min-fingers", type=int, default=MIN_FINGERS, help="distinct fingers required")
    p.add_argument("--fc-gate", type=float, default=FC_GATE, help="minimum FSWO score to stay valid")
    p.add_argument("--max-penetration", type=float, default=MAX_TIP_PENETRATION,
                   help="max fingertip penetration into the cube [m]; -1 disables the gate")
    p.add_argument("--lam", type=float, default=LAMBDA, help="FSWO torque weight lambda")
    p.add_argument("--top", type=int, default=8, help="how many ranked grasps to print")
    p.add_argument("--out", default=SCORES_JSON, help="where to write scores.json ('' to skip)")
    p.add_argument("--compare-v1", action="store_true", help="also report the Version-1 fingertip-FSWO pick")
    p.add_argument("--sweep", action="store_true",
                   help="report how the ranking moves with contact_dist (robustness check)")
    args = p.parse_args()

    kw = dict(
        contact_dist=args.contact_dist,
        min_fingers=args.min_fingers,
        fc_gate=args.fc_gate,
        lam=args.lam,
        max_tip_penetration=None if args.max_penetration < 0 else args.max_penetration,
    )
    best_idx, ranking, per_grasp = select_best_grasp(
        args.library, args.urdf, args.robot_yaml, **kw
    )

    print(f"library: {args.library}")
    print(f"gates:   contact_dist={args.contact_dist} m  min_fingers={args.min_fingers}  "
          f"fc_gate={args.fc_gate}  lambda={args.lam}  "
          f"max_tip_penetration={'off' if args.max_penetration < 0 else f'{args.max_penetration * 1e3:g} mm'}")
    print(f"valid:   {len(ranking)}/{len(per_grasp)} grasps passed the contact + force-closure gates\n")
    print(f"{'rank':>4} {'grasp':>5} {'fingers':>7} {'spheres':>7} {'palm':>4} {'fswo':>11}  "
          f"{'faces':<20} touching")
    for r, idx in enumerate(ranking[: args.top]):
        e = per_grasp[idx]
        faces = ",".join(f"{k}:{v}" for k, v in sorted(e["faces"].items()))
        print(f"{r:>4} {idx:>5} {e['n_fingers']:>7} {e['n_spheres']:>7} {e['n_palm_spheres']:>4} "
              f"{e['fswo']:>11.2e}  {faces:<20} {','.join(e['fingers'])}")
    if len(ranking) > args.top:
        print(f"  ... {len(ranking) - args.top} more valid grasp(s)")

    rejected = [e for e in per_grasp if not e["valid"]]
    if rejected:
        print(f"\nrejected ({len(rejected)}):")
        for e in rejected:
            print(f"  #{e['idx']:>3}  {e['reject']}")

    print(f"\nBEST GRASP INDEX: {best_idx}")
    if best_idx >= 0:
        e = per_grasp[best_idx]
        print(f"  {e['n_fingers']} fingers ({','.join(e['fingers'])}), {e['n_spheres']} contacting spheres "
              f"({e['n_palm_spheres']} on the palm), FSWO {e['fswo']:.6f}")

    if args.sweep:
        # The top few valid grasps are all 4-finger wraps separated by 1-2 contacting
        # spheres, so which one comes first DOES move with contact_dist. This reports
        # the whole family and how often each heads the ranking.
        print("\ncontact-distance sweep:")
        tops: dict[int, int] = {}
        for cd in (0.004, 0.005, 0.006, 0.007, 0.008):
            b, r, _ = select_best_grasp(args.library, args.urdf, args.robot_yaml,
                                        contact_dist=cd, min_fingers=args.min_fingers,
                                        fc_gate=args.fc_gate, lam=args.lam)
            tops[b] = tops.get(b, 0) + 1
            print(f"  contact_dist={cd * 1e3:>4.1f} mm -> best {b:>3}   top5 {r[:5]}")
        consensus = sorted(tops.items(), key=lambda kv: -kv[1])
        print("  wins across the sweep: " + ", ".join(f"#{i} x{n}" for i, n in consensus))

    if args.compare_v1:
        try:
            from .select_optimal_grasp import select_best_grasp_fingertips
        except ImportError:  # pragma: no cover - direct script execution
            from select_optimal_grasp import select_best_grasp_fingertips
        v1_idx, v1_ranking, v1_scores = select_best_grasp_fingertips(
            args.library, args.urdf, lam=args.lam
        )
        print(f"\n[Version 1 — fingertip FSWO, DO NOT SHIP] best_idx={v1_idx} "
              f"score={v1_scores[v1_idx]:.6f}  top5={v1_ranking[:5]}")
        e = per_grasp[v1_idx]
        print(f"  version-2 view of #{v1_idx}: {e['n_fingers']} finger(s) touching, "
              f"{e['n_spheres']} sphere(s), min surface gap {e['min_gap']} m"
              f"{'' if e['valid'] else '  -> REJECTED by the contact gate'}")

    if args.out:
        write_scores_json(
            args.out, best_idx, ranking, per_grasp,
            args.library, args.urdf, args.robot_yaml,
            params={"contact_dist": args.contact_dist, "min_fingers": args.min_fingers,
                    "fc_gate": args.fc_gate, "lam": args.lam, "stage": STAGE_GRASP,
                    "cube_half_edge": CUBE_HALF_EDGE},
        )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
