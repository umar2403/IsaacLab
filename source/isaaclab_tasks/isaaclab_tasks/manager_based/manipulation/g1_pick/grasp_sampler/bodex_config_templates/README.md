# BODex config templates — v2, structurally verified against the real schema

Now that `third_party/BODex_api` is actually cloned (2026-07-13), these three
files were rewritten against the REAL config schema — read directly from
`ultradex_repo/third_party/BODex_api/src/bodex/content/configs/robot/xhand_right_sim2real.yml`,
`configs/robot/hand_pose_transfer/xhand.yml`, and
`configs/manip/sim_xhand_sim2real/fc_right.yml` — instead of guessed from prose.
The v1 drafts (schema guessed from `Misc./UltraDex.md` S2.2 alone) were
structurally wrong in almost every field name; **do not use them**, they have
been replaced in place.

What's still genuinely unverified, marked TODO in each file: collision sphere
geometry (needs real collision meshes, `gen_spheres.py`), the real dex-urdf
Inspire link names (placeholders like `thumb_tip`/`index_proximal` are inferred
from `tools/kinematic_tree.txt`'s USD names, not confirmed against the actual
URDF), and BODex's internal optimization schedule numbers under `seeder_cfg` /
`grasp_contact_strategy` / `grasp_cfg` (copied from xhand's own file as a
starting point — same order-of-magnitude problem, not verified for Inspire).

`ultradex_repo/util/bodex_util.py` has also been patched with the
`hand_type == 'inspire'` branch `Misc./UltraDex.md` S2.2 describes needing
(`bodex_2_sim_q_idx = [0, 1, 2, 3, 4, 5]`, identity — grep for `inspire` in that
file to see it). That part is no longer a TODO.

Copy the three yml files here into
`ultradex_repo/third_party/BODex_api/src/bodex/content/configs/...` at the
paths named in each file's header comment before running
`synthesize_inspire_grasps.py`.

---

**2026-07-30 update**: applied to their target paths in this checkout (see each
file's own "Target path" comment). `ultradex_repo` itself is gitignored and was
missing from this checkout entirely until then — the URDF, this hand config, and
the collision meshes were pulled in read-only from another user's checkout on
the same server (which has a full `ultradex_repo` clone) rather than recreated
from scratch. The `collision_spheres: {}` placeholder in
`inspire_right_sim2real.yml` is still genuinely unpopulated as of this update —
see `grasp_selection/scores.json`'s generation history for how the cached
optimizer output was produced without it, and check whether
`bodex/geom/sphere_fit.py` can populate it without needing the full CUDA-built
BODex_api pipeline (it looks like pure trimesh/numpy geometry, no GPU
optimization required).
