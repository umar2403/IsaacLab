# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Automatic grasp selection for the g1_pick task.

Replaces the human-chosen grasp index with the output of a mechanical
force-closure/wrap optimizer over the BODex/UltraDexGrasp candidate library.
See ``OPTIMIZER_IMPLEMENTATION_SPEC.md`` (Version 2 = sphere-contact-gated FSWO).

Typical use from the environment config::

    from .grasp_selection import get_optimal_grasp_idx
    _OPTIMAL_GRASP_IDX = get_optimal_grasp_idx()

CPU-only and offline: no Isaac Lab / Isaac Sim / GPU involvement.
"""

from __future__ import annotations

import json
import os

from .fswo import fswo_score, psd_sqrt
from .hand_model import (
    CUBE_HALF_EDGE,
    DEFAULT_LIBRARY,
    DEFAULT_ROBOT_YAML,
    DEFAULT_URDF,
    STAGE_GRASP,
)
from .rank_grasps_sphere_fswo import (
    CONTACT_DIST,
    FC_GATE,
    LAMBDA,
    MIN_FINGERS,
    SCORES_JSON,
    score_grasp_spheres,
    select_best_grasp,
    write_scores_json,
)

__all__ = [
    "fswo_score",
    "psd_sqrt",
    "select_best_grasp",
    "score_grasp_spheres",
    "get_optimal_grasp_idx",
    "load_scores",
    "SCORES_JSON",
    "CONTACT_DIST",
    "MIN_FINGERS",
    "FC_GATE",
    "LAMBDA",
    "CUBE_HALF_EDGE",
    "STAGE_GRASP",
    "DEFAULT_LIBRARY",
    "DEFAULT_URDF",
    "DEFAULT_ROBOT_YAML",
]


def load_scores(scores_json: str = SCORES_JSON) -> dict | None:
    """Read the cached optimizer output, or ``None`` if it is absent/unreadable."""
    if not os.path.exists(scores_json):
        return None
    try:
        with open(scores_json) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def get_optimal_grasp_idx(
    scores_json: str = SCORES_JSON,
    library_npz: str = DEFAULT_LIBRARY,
    urdf_path: str = DEFAULT_URDF,
    robot_yaml_path: str = DEFAULT_ROBOT_YAML,
    refresh: bool = False,
    verbose: bool = True,
) -> int:
    """Index of the optimizer-selected grasp in ``library_npz``.

    Uses the cached ``scores.json`` when it is present and not older than the grasp
    library; otherwise runs the optimizer (a few seconds, CPU) and refreshes the cache.
    The cache exists so that training does not depend on the gitignored ``ultradex_repo``
    assets (URDF + collision-sphere YAML) being present.

    Args:
        refresh: ignore the cache and recompute.

    Raises:
        RuntimeError: no cache and the optimizer cannot run (missing assets).
    """
    cached = None if refresh else load_scores(scores_json)
    if cached is not None and os.path.exists(library_npz):
        # stale cache -> recompute (the library changed after the scores were written)
        if os.path.getmtime(scores_json) < os.path.getmtime(library_npz):
            cached = None
    if cached is not None:
        idx = int(cached["best_idx"])
        if verbose:
            print(f"[grasp_selection] optimal grasp #{idx} (cached: {os.path.basename(scores_json)}, "
                  f"{cached.get('n_valid', '?')}/{cached.get('n_grasps', '?')} valid)")
        return idx

    try:
        best_idx, ranking, per_grasp = select_best_grasp(library_npz, urdf_path, robot_yaml_path)
    except (OSError, KeyError) as exc:
        fallback = load_scores(scores_json)
        if fallback is not None:
            return int(fallback["best_idx"])
        raise RuntimeError(
            f"grasp optimizer could not run ({exc}) and no cached {scores_json} exists. "
            f"Run `python -m isaaclab_tasks.manager_based.manipulation.g1_pick.grasp_selection"
            f".rank_grasps_sphere_fswo` with the BODex assets available."
        ) from exc

    if best_idx < 0:
        raise RuntimeError(
            f"grasp optimizer rejected every grasp in {library_npz} — loosen --contact-dist "
            f"or --min-fingers, or re-synthesize the library."
        )

    try:
        write_scores_json(
            scores_json, best_idx, ranking, per_grasp, library_npz, urdf_path, robot_yaml_path,
            params={"contact_dist": CONTACT_DIST, "min_fingers": MIN_FINGERS, "fc_gate": FC_GATE,
                    "lam": LAMBDA, "stage": STAGE_GRASP, "cube_half_edge": CUBE_HALF_EDGE},
        )
    except OSError:
        pass  # read-only checkout: the index is still valid, just not cached

    if verbose:
        e = per_grasp[best_idx]
        print(f"[grasp_selection] optimal grasp #{best_idx} (computed: {e['n_fingers']} fingers, "
              f"{e['n_spheres']} contacting spheres, FSWO {e['fswo']:.4f})")
    return best_idx
