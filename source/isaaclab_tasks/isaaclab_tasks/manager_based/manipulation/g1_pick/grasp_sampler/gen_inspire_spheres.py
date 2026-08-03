"""Populate `collision_spheres` in inspire_right_sim2real.yml by fitting spheres to
the real collision meshes, using BODex_api's own bodex.geom.sphere_fit (pure
trimesh/numpy/torch geometry -- no CUDA optimization pipeline needed for this step).

Coverage note: 4 of the 13 collision_link_names have NO mesh at all in this URDF's
asset set (not even a collision-decomposition gap -- the source geometry is absent):
middle_proximal, ring_proximal, ring_intermediate, pinky_proximal. For those, this
script substitutes a similarly-shaped SIBLING link's real mesh (index_proximal for
the *_proximal links, index_intermediate for ring_intermediate) rather than fabricating
geometry from nothing. hand_base_link has no collision .obj but does have a visual
.glb, which is used directly. This is a documented approximation, not a precise fit --
see the "source" field written into fingertip_contacts-style provenance below.

Run (ultradex env, CPU is fine, GPU optional):
    conda activate ultradex
    python gen_inspire_spheres.py
"""
import os
import sys

import numpy as np
import trimesh
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
MESH_DIR = os.path.join(HERE, "ultradex_repo", "third_party", "BODex_api", "src", "bodex",
                        "content", "assets", "robot", "inspire_hand", "meshes")
YAML_TARGETS = [
    os.path.join(HERE, "ultradex_repo", "third_party", "BODex_api", "src", "bodex",
                "content", "configs", "robot", "inspire_right_sim2real.yml"),
    os.path.join(HERE, "ultradex_repo", "third_party", "BODex_api", "src", "bodex",
                "content", "configs", "robot", "inspire_right.yml"),
    os.path.join(HERE, "bodex_config_templates", "inspire_right_sim2real.yml"),
]

sys.path.insert(0, os.path.join(HERE, "ultradex_repo", "third_party", "BODex_api", "src"))
from bodex.geom.sphere_fit import SphereFitType, fit_spheres_to_mesh  # noqa: E402

# link -> (mesh_file, is_glb, n_spheres, radius, source_note)
LINKS = {
    "hand_base_link":       ("visual/right_base_link.glb", True, 4, 0.012, "own visual mesh (no collision .obj exists)"),
    "thumb_proximal_base":  ("collision/right_thumb_proximal_base.obj", False, 2, 0.008, "own collision mesh"),
    "thumb_proximal":       ("collision/right_thumb_proximal.obj", False, 2, 0.008, "own collision mesh"),
    "thumb_intermediate":   ("collision/right_thumb_intermediate.obj", False, 2, 0.007, "own collision mesh"),
    "thumb_distal":         ("collision/right_thumb_distal.obj", False, 2, 0.007, "own collision mesh"),
    "index_proximal":       ("collision/right_index_proximal.obj", False, 2, 0.008, "own collision mesh"),
    "index_intermediate":   ("collision/right_index_intermediate.obj", False, 2, 0.007, "own collision mesh"),
    "middle_proximal":      ("collision/right_index_proximal.obj", False, 2, 0.008, "SUBSTITUTE: index_proximal mesh (no own mesh exists)"),
    "middle_intermediate":  ("collision/right_middle_intermediate.obj", False, 2, 0.007, "own collision mesh"),
    "ring_proximal":        ("collision/right_index_proximal.obj", False, 2, 0.008, "SUBSTITUTE: index_proximal mesh (no own mesh exists)"),
    "ring_intermediate":    ("collision/right_index_intermediate.obj", False, 2, 0.007, "SUBSTITUTE: index_intermediate mesh (no own mesh exists)"),
    "pinky_proximal":       ("collision/right_index_proximal.obj", False, 2, 0.008, "SUBSTITUTE: index_proximal mesh (no own mesh exists)"),
    "pinky_intermediate":   ("collision/right_pinky_intermediate.obj", False, 2, 0.007, "own collision mesh"),
}


def load_mesh(path: str, is_glb: bool) -> trimesh.Trimesh:
    m = trimesh.load(path, force="mesh")
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate([g for g in m.geometry.values()])
    return m


def main() -> None:
    collision_spheres = {}
    provenance = {}
    for link, (rel_path, is_glb, n_spheres, radius, note) in LINKS.items():
        mesh_path = os.path.join(MESH_DIR, rel_path)
        mesh = load_mesh(mesh_path, is_glb)
        pts, radii = fit_spheres_to_mesh(
            mesh, n_spheres, surface_sphere_radius=radius,
            fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
        )
        pts = np.asarray(pts, dtype=float)
        collision_spheres[link] = [
            {"center": [round(float(c), 5) for c in p], "radius": round(float(r), 5)}
            for p, r in zip(pts, radii)
        ]
        provenance[link] = note
        print(f"  {link:<20} {len(pts)} spheres  <- {rel_path}  ({note})")

    for target in YAML_TARGETS:
        with open(target) as f:
            data = yaml.safe_load(f)
        data["robot_cfg"]["kinematics"]["collision_spheres"] = collision_spheres
        with open(target, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=None, sort_keys=False)
        print(f"wrote collision_spheres -> {target}")

    prov_path = os.path.join(HERE, "grasp_dataset", "collision_spheres_provenance.yml")
    with open(prov_path, "w") as f:
        yaml.safe_dump(provenance, f, sort_keys=False)
    print(f"wrote provenance -> {prov_path}")


if __name__ == "__main__":
    main()
