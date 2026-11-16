"""Two-fighter scene merge: two independent per-person GVHMR passes -> ONE gravity-aligned, floor-grounded GLB.

Pure numpy/scipy (no GPU, no GVHMR import) so it is unit-testable.

Inputs per person (from the worker's per-person pass): GVHMR incam / global SMPL params and `joints24_incam`
(pelvis position in the camera frame, with the true metric depth GVHMR predicts). Both persons share the same
camera, so their camera-frame poses are already in one common frame; merging = (1) estimate world-up in the camera
frame (mean of both persons' gravity-view estimates), (2) rotate the camera frame so up = +Y, (3) put the floor at
y = 0 from both persons' lowest foot vertices, (4) write both skinned figures into one GLB with ONE animation clip.
"""
import json
import struct

import numpy as np
from scipy.spatial.transform import Rotation as R

import smpl_glb as sg
from motion import smooth_motion

P = sg.PARENTS_24
CV2OUT = np.diag([-1.0, -1.0, 1.0])  # OpenCV cam (x right, y down, z fwd) -> Y up, +Z fwd, +X = viewer's left (proper rotation)
FOOT_J = (7, 8, 10, 11)              # ankles + feet
DEFAULT_COLORS = ["#8C1D18", "#1F4FC4"]  # red, blue shorts
PREFIXES = ["fighter0_", "fighter1_"]
CLIP_NAME = "scene"


def rest_shape(model, betas):
    v = model["v_template"] + np.einsum("vcd,d->vc", model["shapedirs"][:, :, :len(betas)], betas)
    return v, model["J_regressor"] @ v


def rots_from_params(go, bp):
    T = len(go)
    aa = np.concatenate([go, bp, np.zeros((T, 6))], 1).reshape(T, 24, 3)
    return R.from_rotvec(aa.reshape(-1, 3)).as_matrix().reshape(T, 24, 3, 3)


def posed(model, betas, rots, root, vert_idx=None):
    """Linear blend skinning (no pose correctives). rots (T,24,3,3) local (joint 0 = global orient), root (T,3) pelvis
    position. Returns verts (T,V,3), joint positions (T,24,3)."""
    v, J = rest_shape(model, betas)
    W = model["weights"]
    if vert_idx is not None:  # only skin a subset of vertices (saves memory on long clips)
        v, W = v[vert_idx], W[vert_idx]
    T = len(rots)
    pos = np.zeros((T, 24, 3))
    wR = np.zeros_like(rots)
    for j in range(24):
        p = P[j]
        if p < 0:
            wR[:, j] = rots[:, j]
            pos[:, j] = root
        else:
            wR[:, j] = wR[:, p] @ rots[:, j]
            pos[:, j] = pos[:, p] + np.einsum("tij,j->ti", wR[:, p], J[j] - J[p])
    out = np.zeros((T, len(v), 3))
    for j in range(24):
        out += W[:, j][None, :, None] * (np.einsum("tij,vj->tvi", wR[:, j], v - J[j]) + pos[:, j][:, None, :])
    return out, pos


def to_quats(rots):
    T = len(rots)
    q = R.from_matrix(rots.reshape(-1, 3, 3)).as_quat().reshape(T, 24, 4)
    for t in range(1, T):
        flip = (q[t] * q[t - 1]).sum(-1) < 0
        q[t][flip] = -q[t][flip]
    return q


def person_motion(d, smoothing="medium"):
    """Camera-frame (OpenCV) local rotations + pelvis position for one person. d: dict with incam_global_orient (T,3),
    incam_body_pose (T,63), joints24_incam (T,24,3)."""
    go, bp, root = smooth_motion(d["incam_global_orient"], d["incam_body_pose"], d["joints24_incam"][:, 0], smoothing)
    return rots_from_params(go, bp), root


def apply_world(rots, root, Rw, tw):
    """Rigidly transform a figure: only the root rotation / position change (other joints are local)."""
    r2 = rots.copy()
    r2[:, 0] = Rw @ rots[:, 0]
    return r2, root @ Rw.T + tw


def gravity_up_cam(d):
    """World-up expressed in OpenCV camera coordinates, per frame (T,3): GVHMR's camera->gravity-view rotation
    R_c2gv = R_global_orient @ R_incam_global_orient^T; its second row is the up axis."""
    Rg = R.from_rotvec(d["global_global_orient"]).as_matrix()
    Rc = R.from_rotvec(d["incam_global_orient"]).as_matrix()
    return (Rg @ Rc.transpose(0, 2, 1))[:, 1, :]


def rot_between(a, b):
    """Minimal rotation taking direction a to direction b."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    if s < 1e-9:
        return np.eye(3) if a @ b > 0 else R.from_rotvec(np.pi * _any_perp(a)).as_matrix()
    return R.from_rotvec(v / s * np.arctan2(s, a @ b)).as_matrix()


def _any_perp(a):
    p = np.cross(a, [1.0, 0, 0])
    if np.linalg.norm(p) < 1e-6:
        p = np.cross(a, [0, 1.0, 0])
    return p / np.linalg.norm(p)


def gravity_frame(persons):
    """-> (G, Wr, up_cv, angle_deg): up_cv = mean over persons of median per-frame up (unit); G rotates the CV2OUT frame so
    up -> +Y (minimal rotation, no extra yaw); Wr = G @ CV2OUT maps OpenCV camera coords to the world frame."""
    meds = [np.median(gravity_up_cam(d), 0) for d in persons]
    meds = [m / np.linalg.norm(m) for m in meds]
    up_cv = np.mean(meds, 0)
    up_cv /= np.linalg.norm(up_cv)
    angle = float(np.degrees(np.arccos(np.clip(meds[0] @ meds[-1], -1, 1))))
    G = rot_between(CV2OUT @ up_cv, np.array([0, 1.0, 0]))
    return G, G @ CV2OUT, up_cv, angle


def foot_vertex_indices(model):
    return np.where(np.isin(model["weights"].argmax(1), FOOT_J))[0]


def floor_height(model, betas_list, motions, Wr, percentile=10):
    """Floor y in the gravity-aligned frame (before the offset): percentile of the lowest foot vertex per frame, pooled over
    all persons (~ the soles while planted; robust to a few lifted frames). Returns (h_floor, per-person lows)."""
    foot = foot_vertex_indices(model)
    lows = []
    for betas, (rots, root) in zip(betas_list, motions):
        r2, p2 = apply_world(rots, root, Wr, np.zeros(3))
        vv, _ = posed(model, betas, r2, p2, foot)
        lows.append(vv[:, :, 1].min(1))
    return float(np.percentile(np.concatenate(lows), percentile)), lows


# ---------------- GLB merge: N single-character GLBs -> one scene, ONE animation clip ----------------
def read_glb(b):
    ln = struct.unpack("<I", b[12:16])[0]
    js = json.loads(b[20:20 + ln])
    off = 20 + ln
    bl = struct.unpack("<I", b[off:off + 4])[0]
    return js, b[off + 8:off + 8 + bl]


def merge_glbs(glbs, prefixes, clip_name=CLIP_NAME, scene_extras=None):
    """Concatenate single-character GLBs (smpl_glb.build_glb output) into one scene. Every node / mesh / material / skin
    name gets the person's prefix; all animation channels go into ONE clip `clip_name`."""
    out = {"asset": {"version": "2.0", "generator": "pt-mocap-worker scene"}, "scene": 0, "scenes": [{"name": "sparring", "nodes": []}],
           "nodes": [], "meshes": [], "materials": [], "skins": [], "animations": [], "accessors": [], "bufferViews": []}
    if scene_extras:
        out["scenes"][0]["extras"] = scene_extras
    clip = {"name": clip_name, "samplers": [], "channels": []}
    binary = b""
    for g, pre in zip(glbs, prefixes):
        js, bn = read_glb(g)
        n0, m0, mat0, s0, a0, bv0 = (len(out[k]) for k in ("nodes", "meshes", "materials", "skins", "accessors", "bufferViews"))
        boff = len(binary)
        for bv in js["bufferViews"]:
            bv = dict(bv)
            bv["byteOffset"] += boff
            out["bufferViews"].append(bv)
        for a in js["accessors"]:
            a = dict(a)
            a["bufferView"] += bv0
            out["accessors"].append(a)
        for n in js["nodes"]:
            n = dict(n)
            n["name"] = pre + n["name"]
            if "children" in n:
                n["children"] = [c + n0 for c in n["children"]]
            if "mesh" in n:
                n["mesh"] += m0
            if "skin" in n:
                n["skin"] += s0
            out["nodes"].append(n)
        for m in js["meshes"]:
            m = json.loads(json.dumps(m))
            m["name"] = pre + m["name"]
            for p in m["primitives"]:
                p["attributes"] = {k: v + a0 for k, v in p["attributes"].items()}
                p["indices"] += a0
                p["material"] += mat0
            out["meshes"].append(m)
        for m in js["materials"]:
            m = dict(m)
            m["name"] = pre + m["name"]
            out["materials"].append(m)
        for s in js["skins"]:
            s = dict(s)
            s["name"] = pre + "skin"
            s["joints"] = [j + n0 for j in s["joints"]]
            s["inverseBindMatrices"] += a0
            s["skeleton"] += n0
            out["skins"].append(s)
        for an in js["animations"]:
            k0 = len(clip["samplers"])
            for sm in an["samplers"]:
                sm = dict(sm)
                sm["input"] += a0
                sm["output"] += a0
                clip["samplers"].append(sm)
            for ch in an["channels"]:
                clip["channels"].append({"sampler": ch["sampler"] + k0, "target": {"node": ch["target"]["node"] + n0, "path": ch["target"]["path"]}})
        out["scenes"][0]["nodes"] += [n + n0 for n in js["scenes"][0]["nodes"]]
        binary += bn
    out["animations"].append(clip)
    out["buffers"] = [{"byteLength": len(binary)}]
    jb = json.dumps(out, separators=(",", ":")).encode()
    jb += b" " * (-len(jb) % 4)
    binary += b"\0" * (-len(binary) % 4)
    total = 12 + 8 + len(jb) + 8 + len(binary)
    return (struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<I4s", len(jb), b"JSON") + jb
            + struct.pack("<I4s", len(binary), b"BIN\0") + binary)


def build_person_glb(model, betas, rots, root, color, fps):
    v, J = rest_shape(model, betas)
    colors, _ = sg.clothing_colors(v, model["faces"], J, model["weights"], "male", shorts_color=color, top=False)
    return sg.build_glb(v, model["faces"], J, P, model["weights"], to_quats(rots), root, fps, sg.JOINT_NAMES_24, colors)


def finger_rots(left_aa, right_aa):
    """(T,15,3) x2 absolute SMPL-X axis-angle -> (T,30,3,3) local rotations in SMPL-X finger order (left 15, right 15)."""
    T = len(left_aa)
    aa = np.concatenate([np.asarray(left_aa), np.asarray(right_aa)], 1)
    return R.from_rotvec(aa.reshape(-1, 3)).as_matrix().reshape(T, 30, 3, 3)


def build_scene(model, persons, K, fps, smoothing="medium", colors=None, track_ids=None, observed=None, bboxes=None,
                percentile=10, hands=None, smplx_model=None):
    """persons: list of dicts (see person_motion / gravity_up_cam, plus incam_betas). Returns (glb_bytes, meta dict).
    hands: optional list (per person) of (left_aa, right_aa) SMPL-X finger poses (T,15,3). With hands the figures are
    SMPL-X meshes (`smplx_model` from smplx_np.load_smplx_npz) with 30 animated finger joints; body motion, pelvis track,
    gravity and floor are computed exactly as for SMPL (same root / body rotations / floor height)."""
    colors = list(colors or DEFAULT_COLORS)
    n = len(persons)
    T = len(persons[0]["incam_transl"])
    betas = [np.asarray(d["incam_betas"], float).reshape(-1, 10).mean(0) for d in persons]
    mot = [person_motion(d, smoothing) for d in persons]
    G, Wr, up_cv, angle = gravity_frame(persons)
    h_floor, lows = floor_height(model, betas, mot, Wr, percentile)
    tw = np.array([0, -h_floor, 0])
    glbs, pelvis = [], []
    for i, (b, (rots, root)) in enumerate(zip(betas, mot)):
        rw, pw = apply_world(rots, root, Wr, tw)
        if hands is not None:
            import smplx_np as sx

            r52 = np.concatenate([rw[:, :22], finger_rots(*hands[i])], axis=1)
            glbs.append(sx.build_person_glb(smplx_model, b, r52, pw, colors[i % len(colors)], fps))
        else:
            glbs.append(build_person_glb(model, b, rw, pw, colors[i % len(colors)], fps))
        pelvis.append(pw)
    prefixes = [PREFIXES[i] if i < len(PREFIXES) else f"fighter{i}_" for i in range(n)]
    K = np.asarray(K, float)
    cam_h = float(-h_floor)
    fov_y = float(np.degrees(2 * np.arctan(K[1, 2] / K[1, 1])))
    glb = merge_glbs(glbs, prefixes, scene_extras={"cam_height_m": cam_h, "fps": fps, "up": "Y", "units": "m"})
    pers = []
    for i in range(n):
        item = {"index": i, "node_prefix": prefixes[i], "color": colors[i % len(colors)],
                "pelvis_node": prefixes[i] + "Pelvis", "mesh_node": prefixes[i] + "SMPL_mesh",
                "track_ids": [int(x) for x in (track_ids[i] if track_ids else [])],
                "observed_frames": int(observed[i].sum()) if observed is not None else int(T),
                "foot_y_p10_m": round(float(np.percentile(lows[i], percentile)), 4),
                "pelvis_world_m": np.round(pelvis[i], 4).tolist()}
        if bboxes is not None:
            bb = np.asarray(bboxes[i], float)
            item["bbox_xyxy"] = {"first": np.round(bb[0], 1).tolist(), "mean": np.round(bb.mean(0), 1).tolist(),
                                 "last": np.round(bb[-1], 1).tolist()}
        pers.append(item)
    meta = {
        "fps": float(fps), "n_frames": int(T), "duration_s": round(T / fps, 3), "units": "m", "up_axis": "Y",
        "clip": CLIP_NAME, "cam_height_m": cam_h, "floor_y": 0.0,
        "K": K.tolist(), "fov_y_deg": fov_y,
        "camera": {"position": [0.0, cam_h, 0.0],
                   "R_world_from_cv_cam": Wr.tolist(),
                   "R_world_from_three_cam": (Wr @ np.diag([1.0, -1.0, -1.0])).tolist(),
                   "note": "world = R_world_from_cv_cam @ x_cv + [0, cam_height_m, 0]; three.js camera (looks down -Z, Y up) "
                           "uses R_world_from_three_cam as its rotation matrix."},
        "gravity": {"up_in_camera_cv": up_cv.tolist(), "R_g": G.tolist(), "angle_between_person_estimates_deg": angle},
        "persons": pers,
    }
    if hands is not None:
        import smplx_np as sx

        meta["hands"] = {"skeleton": "SMPL-X 22 body + 30 finger joints (jaw / eyes folded into head)", "joint_names": sx.NAMES_52,
                         "finger_joints": sx.NAMES_52[22:]}
    return glb, meta
