"""Offline grasp-pose viewer -- NO Isaac Lab / Isaac Sim / GPU required.

Renders the real Inspire Hand (URDF visual meshes) posed at a chosen grasp from the
offline goal library, together with the 5 cm cube, so you can eyeball whether the
selected grasp is actually any good BEFORE training anything.

READ-ONLY with respect to the pipeline: it loads
    grasp_dataset/cube_5cm_grasps_valid.npz              (the goal library)
    ../grasp_selection/scores.json                        (which grasp the optimizer picked)
    assets/robot/inspire_hand/inspire_hand_right.urdf     (+ its visual meshes)
and writes only the output files you ask for.

Frame convention: the library stores the hand ROOT ("base" link) pose in the CUBE
frame, so every link sits at ``T_palm @ FK_root->link`` with the cube at the origin.
No ``T_usdbase_urdfbase`` is applied (that belongs to the online reward only).

Modes
    mesh     : the hand's visual meshes (default)
    spheres  : the ~41 collision spheres the optimizer reasons about
    contacts : meshes + the contacting spheres highlighted green, the rest faint --
               this shows *why* a grasp won

Outputs
    <out>.html : self-contained three.js scene -> open it in Firefox/Chrome, orbit with the mouse
    <out>.glb  : standard 3D file (Blender, any glTF viewer)
    <out>.png  : quick static matplotlib preview (2 viewpoints)

Usage (CPU only, any env with trimesh):
    python visualize_grasp_offline.py                  # the optimizer's pick (scores.json)
    python visualize_grasp_offline.py --index 12       # some other grasp
    python visualize_grasp_offline.py --mode contacts  # show what is touching the cube
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "grasp_selection"))

# reuse the optimizer's own validated FK / constants / asset paths (read-only import)
from hand_model import (  # noqa: E402
    ACTUATED_JOINT_NAMES,
    CUBE_HALF_EDGE,
    DEFAULT_LIBRARY,
    DEFAULT_ROBOT_YAML,
    DEFAULT_URDF,
    cube_contact,
    cube_surface_dist,
    load_chain,
    load_spheres,
    palm_matrix,
)
import rank_grasps_sphere_fswo as rk  # noqa: E402

_SCORES = os.path.join(os.path.dirname(_HERE), "grasp_selection", "scores.json")
_STAGE_NAMES = {0: "pregrasp", 1: "grasp", 2: "squeeze"}


def _urdf_visuals(urdf_path: str) -> list[tuple[str, str, np.ndarray]]:
    """(link_name, absolute_mesh_path, visual_origin_4x4) for every <visual> mesh."""
    root_dir = os.path.dirname(urdf_path)
    out = []
    for link in ET.parse(urdf_path).getroot().findall("link"):
        for vis in link.findall("visual"):
            mesh = vis.find("geometry/mesh")
            if mesh is None:
                continue
            o = vis.find("origin")
            xyz = [float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy") if o is not None else "0 0 0").split()]
            T = np.eye(4)
            T[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
            T[:3, 3] = xyz
            out.append((link.get("name"), os.path.join(root_dir, mesh.get("filename")), T))
    return out


def _as_mesh(path: str):
    """Load a .glb/.obj/.stl into a single Trimesh (glb files carry a scene)."""
    loaded = trimesh.load(path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate([g for g in loaded.geometry.values()])
    return loaded


def build_scene(joint_q, palm_pos, palm_quat, cube_size, mode="mesh",
                urdf_path=DEFAULT_URDF, robot_yaml=DEFAULT_ROBOT_YAML,
                contact_dist=rk.CONTACT_DIST) -> trimesh.Scene:
    T_palm = palm_matrix(palm_pos, palm_quat)
    links = load_chain(urdf_path).link_transforms(joint_q)
    scene = trimesh.Scene()

    # --- the cube (target object), centered at the origin of the grasp frame ---
    cube = trimesh.creation.box(extents=(cube_size,) * 3)
    cube.visual.face_colors = [220, 60, 60, 170]  # translucent red so fingers show through
    scene.add_geometry(cube, geom_name="cube", node_name="cube")

    if mode in ("mesh", "contacts", "mesh_and_spheres"):
        n = 0
        for link_name, mesh_path, T_vis in _urdf_visuals(urdf_path):
            if link_name not in links or not os.path.isfile(mesh_path):
                continue
            m = _as_mesh(mesh_path)
            m.apply_transform(T_palm @ links[link_name] @ T_vis)
            m.visual.face_colors = [190, 195, 205, 255]  # light metallic grey
            scene.add_geometry(m, geom_name=f"link_{link_name}", node_name=f"link_{link_name}")
            n += 1
        print(f"  placed {n} hand link meshes")

    if mode in ("spheres", "contacts", "mesh_and_spheres"):
        spheres = load_spheres(robot_yaml)
        n_touch = n_far = 0
        for link_name, lst in spheres.items():
            if link_name not in links:
                continue
            for center, radius in lst:
                c_cube = (T_palm @ links[link_name] @ np.append(center, 1.0))[:3]
                touching = (cube_surface_dist(c_cube) - radius) <= contact_dist
                if mode == "contacts" and not touching:
                    continue  # keep the view readable: only what the optimizer counted
                sph = trimesh.creation.icosphere(subdivisions=2, radius=float(radius))
                T = np.eye(4)
                T[:3, 3] = c_cube
                sph.apply_transform(T)
                sph.visual.face_colors = ([60, 220, 120, 210] if touching
                                          else [70, 190, 230, 120])
                sph_name = f"sph_{link_name}_{n_touch + n_far}"
                scene.add_geometry(sph, geom_name=sph_name, node_name=sph_name)
                n_touch += int(touching)
                n_far += int(not touching)
        print(f"  placed {n_touch} contacting sphere(s)" + (f" + {n_far} non-contacting" if n_far else ""))

    # small axis triad at the cube origin for orientation reference
    scene.add_geometry(trimesh.creation.axis(origin_size=0.004, axis_length=0.05), node_name="axes")
    return scene


def _preview_png(scene: trimesh.Scene, path: str, title: str) -> None:
    """Cheap static preview (no pyrender needed): triangles via matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(13, 6))
    for k, (elev, azim) in enumerate([(22, 45), (22, 135)]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        sphere_centers, sphere_colors, sphere_sizes = [], [], []
        for name, geom in scene.geometry.items():
            if not isinstance(geom, trimesh.Trimesh) or len(geom.faces) == 0 or name == "axes":
                continue
            is_cube = name == "cube"
            is_sphere = name.startswith("sph_")
            if is_sphere:
                # mplot3d has no real z-buffer, so overlapping Poly3DCollections (mesh vs.
                # tiny spheres) sort unreliably and the spheres vanish behind the hand.
                # Draw spheres as scatter points instead -- far more reliably visible,
                # and drawn in a final pass so they sit on top.
                fc = np.asarray(geom.visual.face_colors[0], dtype=float) / 255.0
                sphere_centers.append(geom.vertices.mean(axis=0))
                sphere_colors.append(tuple(fc.tolist()))
                sphere_sizes.append(float(geom.bounding_sphere.primitive.radius) * 6000)
                continue
            # Render each part's CONVEX HULL: matplotlib can't depth-sort a dense mesh,
            # and random triangle subsampling turns the hand into confetti. Per-link hulls
            # keep every finger segment as a readable solid.
            tris = geom.triangles if is_cube else geom.convex_hull.triangles
            col = (0.88, 0.22, 0.22, 0.45) if is_cube else (0.55, 0.60, 0.68, 0.55)
            ax.add_collection3d(
                Poly3DCollection(tris, facecolors=col, edgecolors=(0.25, 0.28, 0.32, 0.35), linewidths=0.2)
            )
        if sphere_centers:
            pts = np.array(sphere_centers)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=sphere_colors, s=sphere_sizes,
                      depthshade=False, edgecolors=(0.15, 0.15, 0.15, 0.9), linewidths=0.6, zorder=10)
        ax.set_xlim(-0.13, 0.13); ax.set_ylim(-0.13, 0.13); ax.set_zlim(-0.13, 0.13)
        ax.set_box_aspect([1, 1, 1]); ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(f"view {k + 1}")
    fig.suptitle(title)
    plt.tight_layout(); plt.savefig(path, dpi=95, bbox_inches="tight"); plt.close(fig)


def _caption(html: str, lines: list[str]) -> str:
    """Overlay a small caption on the three.js page (best-effort, skipped if no <body>)."""
    if "<body" not in html:
        return html
    box = ("<div style=\"position:fixed;top:10px;left:10px;z-index:9;font:13px/1.5 "
           "system-ui,sans-serif;background:rgba(20,22,26,.82);color:#e8eaed;padding:10px 14px;"
           "border-radius:8px;max-width:min(92vw,520px)\">"
           + "<br>".join(lines) + "</div>")
    i = html.index(">", html.index("<body")) + 1
    return html[:i] + box + html[i:]


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline Inspire-hand grasp pose viewer (no Isaac Lab).")
    ap.add_argument("--library", default=DEFAULT_LIBRARY, help="goal library npz")
    ap.add_argument("--index", type=int, default=None,
                    help="grasp row to view (default: best_idx from grasp_selection/scores.json)")
    ap.add_argument("--stage", type=int, default=1, choices=[0, 1, 2],
                    help="0=pregrasp, 1=grasp (default), 2=squeeze")
    ap.add_argument("--cube", type=float, default=2 * CUBE_HALF_EDGE, help="cube edge length (m)")
    ap.add_argument("--mode", choices=["mesh", "spheres", "contacts", "mesh_and_spheres"], default="mesh",
                    help="hand representation. mesh_and_spheres = full mesh + ALL collision "
                         "spheres (green=touching the cube, blue=not) -- everything the "
                         "policy's grasp_reach/optimizer actually sees.")
    ap.add_argument("--out", default=None, help="output basename (default: grasp_viz/grasp_<idx>_<mode>)")
    ap.add_argument("--no-png", action="store_true", help="skip the matplotlib preview")
    args = ap.parse_args()

    data = np.load(args.library, allow_pickle=True)
    g = np.asarray(data["grasp_pose"], dtype=float)

    idx = args.index
    if idx is None:
        with open(_SCORES) as f:
            idx = int(json.load(f)["best_idx"])
        print(f"using the optimizer's pick from scores.json: grasp #{idx}")
    if not 0 <= idx < g.shape[0]:
        raise SystemExit(f"--index {idx} out of range (library has {g.shape[0]} grasps)")

    grasp = g[idx, 0, args.stage, :]
    pos, quat, q6 = grasp[:3], grasp[3:7], grasp[7:]

    # stats, so the picture is labelled with what the optimizer actually measured
    info = rk.score_grasp_spheres(load_chain(), load_spheres(), g[idx, 0, 1, :])
    src = data["source_indices"][idx] if "source_indices" in data else None
    Rm = R.from_quat(quat, scalar_first=True).as_matrix()
    lines = [
        f"<b>grasp #{idx}</b>" + (f" (pool #{src})" if src is not None else "")
        + f" &mdash; stage: {_STAGE_NAMES[args.stage]}",
        f"{info['n_fingers']} fingers touching ({', '.join(info['fingers'])}), "
        f"{info['n_spheres']} contacting spheres, FSWO {info['fswo']:.2e}"
        if info["fswo"] is not None else f"rejected: {info['reject']}",
        f"palm {np.linalg.norm(pos[:2]) * 100:.1f} cm lateral, {pos[2] * 100:.1f} cm above the cube, "
        f"{np.degrees(np.arccos(np.clip(-Rm[2, 2], -1, 1))):.1f}&deg; from straight down",
        "red = 5 cm cube &middot; grey = hand" + (" &middot; green = contacting spheres"
                                                 if args.mode in ("spheres", "contacts") else ""),
    ]
    print(f"\ngrasp #{idx}" + (f" (pool #{src})" if src is not None else ""))
    for ln in lines[1:3]:
        print("  " + ln.replace("&mdash;", "-").replace("&deg;", " deg").replace("&middot;", "-"))

    out = args.out or os.path.join(_HERE, "grasp_viz", f"grasp_{idx}_{args.mode}")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    print(f"\nbuilding scene ({args.mode}) ...")
    scene = build_scene(q6, pos, quat, args.cube, mode=args.mode)

    from trimesh.viewer import scene_to_html
    html = _caption(scene_to_html(scene), lines)
    with open(out + ".html", "w") as f:
        f.write(html)
    scene.export(out + ".glb")
    print(f"\nWrote:\n  {out}.html   <-- open this in Firefox/Chrome (orbit with the mouse)\n  {out}.glb")
    if not args.no_png:
        _preview_png(scene, out + ".png", f"grasp #{idx} ({_STAGE_NAMES[args.stage]}) - {args.mode}")
        print(f"  {out}.png")
    print(f"\nfile://{os.path.abspath(out)}.html")


if __name__ == "__main__":
    main()
