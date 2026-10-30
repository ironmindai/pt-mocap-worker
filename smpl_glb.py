"""Skinned SMPL mesh + animation -> GLB (raw glTF 2.0 binary, no external deps beyond numpy/scipy).

SMPL linear blend skinning maps 1:1 onto glTF skinning: SMPL joints are glTF nodes (node index == joint index),
inverse bind matrices are translations by -J (rest joints of the shaped T-pose), weights are pruned to the top 4
influences per vertex, and the animation holds one rotation track per joint plus a root translation track.
Pose-corrective blendshapes are not representable in glTF skinning and are omitted.
Conventions match the BVH export: Y-up, meters, facing +Z at frame 0, feet on y=0.
"""
import json
import pickle
import struct

import numpy as np
from scipy.spatial.transform import Rotation as R

from smpl_bvh import rebase

JOINT_NAMES_24 = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2", "L_Ankle", "R_Ankle", "Spine3",
    "L_Foot", "R_Foot", "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder", "L_Elbow",
    "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand",
]
PARENTS_24 = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
ROUGHNESS = 0.6


def _srgb_to_linear(hex_color):
    c = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


# glTF vertex colors / baseColorFactor are linear.
SKIN = np.array([0.80, 0.62, 0.52])
SHORTS = _srgb_to_linear("#8C1D18")
WAISTBAND = _srgb_to_linear("#C9A227")
TOP = _srgb_to_linear("#2B2D31")  # dark charcoal
SHORTS_JOINTS = (0, 1, 2)
TOP_JOINTS = (3, 6, 9)
SHORTS_HEM_T = 0.6  # shorts end 60% of the way from hip joint to knee joint
WAISTBAND_M = 0.04  # waistband height below the top edge of the shorts


def clothing_colors(verts, faces, joints, weights, gender="neutral", feather=1):
    """Painted-on clothing as per-vertex linear RGBA, derived from rest pose + skinning weights.

    Shorts (all genders): vertices whose dominant joint is pelvis / hips, above 60% of the hip->knee drop, with a
    gold waistband strip at the top edge. Sports top (female and neutral): vertices dominated by spine joints
    above the lower-chest cutoff (arms / shoulders / neck stay skin). Rest is skin.
    Returns (colors (V,4) float32, region labels (V,) uint8: 0 skin, 1 shorts, 2 waistband, 3 top).
    """
    verts, joints, weights = np.asarray(verts), np.asarray(joints), np.asarray(weights)
    V = len(verts)
    dom = weights.argmax(axis=1)
    y = verts[:, 1]
    label = np.zeros(V, dtype=np.uint8)

    # Shorts: hem per leg (pelvis-dominated vertices use the side of the body they are on).
    side_left = verts[:, 0] >= joints[0, 0]  # SMPL left is +X
    hip_y = np.where(side_left, joints[1, 1], joints[2, 1])
    knee_y = np.where(side_left, joints[4, 1], joints[5, 1])
    hem_y = hip_y + SHORTS_HEM_T * (knee_y - hip_y)
    shorts = np.isin(dom, SHORTS_JOINTS) & (y >= hem_y)
    label[shorts] = 1
    if shorts.any():
        label[shorts & (y >= y[shorts].max() - WAISTBAND_M)] = 2

    if gender in ("female", "neutral"):
        cut = joints[3, 1] + 0.6 * (joints[6, 1] - joints[3, 1])
        label[np.isin(dom, TOP_JOINTS) & (y >= cut)] = 3

    palette = np.stack([SKIN, SHORTS, WAISTBAND, TOP])
    rgb = palette[label]
    for _ in range(feather):
        rgb = _feather(rgb, faces)
    rgba = np.concatenate([rgb, np.ones((V, 1))], axis=1).astype(np.float32)
    return rgba, label


def _feather(rgb, faces, amount=0.5):
    """One 1-ring smoothing pass of vertex colors (softens region boundaries)."""
    V = len(rgb)
    faces = np.asarray(faces, dtype=np.int64)
    acc, cnt = np.zeros_like(rgb), np.zeros(V)
    for a, b in ((0, 1), (1, 2), (2, 0)):
        for i, j in ((a, b), (b, a)):
            np.add.at(acc, faces[:, i], rgb[faces[:, j]])
            np.add.at(cnt, faces[:, i], 1)
    mean = np.where(cnt[:, None] > 0, acc / np.maximum(cnt[:, None], 1), rgb)
    return (1 - amount) * rgb + amount * mean


class _Stub:
    """Stand-in for classes that cannot be imported (chumpy.Ch); keeps the pickled state."""

    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        self.state = state


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return _Stub
        return super().find_class(module, name)


def load_smpl_pkl(path, n_betas=10):
    """Load an SMPL .pkl (original chumpy-based or the trimmed numpy version) into plain numpy arrays.

    Does not need chumpy: chumpy arrays are unpickled as stubs and their raw ndarray ("x") is used.
    """
    with open(path, "rb") as f:
        d = _Unpickler(f, encoding="latin1").load()

    def arr(x):
        if isinstance(x, _Stub):
            x = x.state["x"]
        if hasattr(x, "toarray"):  # scipy sparse
            x = x.toarray()
        return np.asarray(x)

    return {
        "v_template": arr(d["v_template"]).astype(np.float64),
        "shapedirs": arr(d["shapedirs"])[:, :, :n_betas].astype(np.float64),
        "J_regressor": arr(d["J_regressor"]).astype(np.float64),
        "weights": arr(d["weights"]).astype(np.float64),
        "faces": arr(d["f"]).astype(np.uint32),
    }


def _normals(v, f):
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def top4(weights):
    """(V,J) -> joints (V,4) uint16, weights (V,4) float32 renormalized to sum 1."""
    idx = np.argsort(-weights, axis=1)[:, :4]
    w = np.take_along_axis(weights, idx, axis=1)
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    return idx.astype(np.uint16), w.astype(np.float32)


def _pad4(b, pad=b"\0"):
    return b + pad * (-len(b) % 4)


def build_glb(verts, faces, joints, parents, weights, quats, root_pos, fps, names=None, colors=None):
    """Assemble a skinned, animated GLB and return its bytes.

    verts (V,3) rest mesh; faces (F,3); joints (J,3) rest joint positions; parents (J,) with -1 for root;
    weights (V,J) LBS weights; colors optional (V,4) linear RGBA -> COLOR_0; quats (T,J,4) local rotations xyzw; root_pos (T,3) root node translation; fps float.
    """
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.uint32)
    joints = np.asarray(joints, dtype=np.float64)
    quats = np.asarray(quats, dtype=np.float32)
    root_pos = np.asarray(root_pos, dtype=np.float32)
    V, J, T = len(verts), len(joints), len(quats)
    names = names or [f"joint_{i}" for i in range(J)]
    jidx, jw = top4(np.asarray(weights, dtype=np.float64))
    ibm = np.tile(np.eye(4, dtype=np.float32), (J, 1, 1))
    ibm[:, 3, :3] = -joints  # column-major translation column
    times = (np.arange(T) / fps).astype(np.float32)

    chunks, views, accessors = [], [], []

    def add(data, ctype, count, atype, target=None, minmax=None):
        raw = np.ascontiguousarray(data).tobytes()
        offset = sum(len(c) for c in chunks)
        chunks.append(_pad4(raw))
        bv = {"buffer": 0, "byteOffset": offset, "byteLength": len(raw)}
        if target:
            bv["target"] = target
        views.append(bv)
        acc = {"bufferView": len(views) - 1, "componentType": ctype, "count": count, "type": atype}
        if minmax:
            acc["min"], acc["max"] = minmax
        accessors.append(acc)
        return len(accessors) - 1

    FLOAT, USHORT, UINT = 5126, 5123, 5125
    a_pos = add(verts, FLOAT, V, "VEC3", 34962, (verts.min(0).tolist(), verts.max(0).tolist()))
    a_nrm = add(_normals(verts.astype(np.float64), faces).astype(np.float32), FLOAT, V, "VEC3", 34962)
    a_jnt = add(jidx, USHORT, V, "VEC4", 34962)
    a_wgt = add(jw, FLOAT, V, "VEC4", 34962)
    a_col = add(np.asarray(colors, dtype=np.float32), FLOAT, V, "VEC4", 34962) if colors is not None else None
    a_idx = add(faces.reshape(-1), UINT, faces.size, "SCALAR", 34963)
    a_ibm = add(ibm, FLOAT, J, "MAT4")
    a_time = add(times, FLOAT, T, "SCALAR", None, ([float(times.min())], [float(times.max())]))
    a_root = add(root_pos, FLOAT, T, "VEC3")
    a_rot = [add(quats[:, j], FLOAT, T, "VEC4") for j in range(J)]

    nodes = []
    for j in range(J):
        node = {"name": names[j]}
        p = parents[j]
        node["translation"] = (joints[j] if p < 0 else joints[j] - joints[p]).astype(float).tolist()
        kids = [k for k in range(J) if parents[k] == j]
        if kids:
            node["children"] = kids
        nodes.append(node)
    nodes.append({"name": "SMPL_mesh", "mesh": 0, "skin": 0})

    samplers, channels = [{"input": a_time, "output": a_root, "interpolation": "LINEAR"}], []
    channels.append({"sampler": 0, "target": {"node": 0, "path": "translation"}})
    for j in range(J):
        samplers.append({"input": a_time, "output": a_rot[j], "interpolation": "LINEAR"})
        channels.append({"sampler": len(samplers) - 1, "target": {"node": j, "path": "rotation"}})

    attrs = {"POSITION": a_pos, "NORMAL": a_nrm, "JOINTS_0": a_jnt, "WEIGHTS_0": a_wgt}
    if a_col is not None:
        attrs["COLOR_0"] = a_col
    # With vertex colors the look is driven by COLOR_0 (final = baseColorFactor * COLOR_0).
    base_color = [1.0, 1.0, 1.0, 1.0] if a_col is not None else list(SKIN) + [1.0]
    gltf = {
        "asset": {"version": "2.0", "generator": "pt-mocap-worker"},
        "scene": 0,
        "scenes": [{"nodes": [J, 0]}],
        "nodes": nodes,
        "meshes": [{"name": "SMPL", "primitives": [{
            "attributes": attrs, "indices": a_idx, "material": 0, "mode": 4}]}],
        "materials": [{"name": "skin", "doubleSided": False, "pbrMetallicRoughness": {
            "baseColorFactor": base_color, "metallicFactor": 0.0, "roughnessFactor": ROUGHNESS}}],
        "skins": [{"joints": list(range(J)), "inverseBindMatrices": a_ibm, "skeleton": 0}],
        "animations": [{"name": "motion", "samplers": samplers, "channels": channels}],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": sum(len(c) for c in chunks)}],
    }
    jb = _pad4(json.dumps(gltf, separators=(",", ":")).encode(), b" ")
    bb = b"".join(chunks)
    total = 12 + 8 + len(jb) + 8 + len(bb)
    return (struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<I4s", len(jb), b"JSON") + jb
            + struct.pack("<I4s", len(bb), b"BIN\0") + bb)


def smpl_to_glb(model, global_orient, body_pose, transl, betas, fps, gender="neutral", clothing=True):
    """SMPL(-X body) pose sequence -> GLB bytes. `model` comes from load_smpl_pkl.

    global_orient (T,3), body_pose (T,63) axis-angle, transl (T,3) meters, betas (10,) or (T,10).
    Hand joints (22, 23) are held at identity.
    """
    go = np.asarray(global_orient, dtype=np.float64).reshape(-1, 3)
    T = len(go)
    bp = np.asarray(body_pose, dtype=np.float64).reshape(T, -1)[:, :63]
    aa = np.concatenate([go, bp, np.zeros((T, 6))], axis=1).reshape(T, 24, 3)
    rots = R.from_rotvec(aa.reshape(-1, 3)).as_matrix().reshape(T, 24, 3, 3)
    betas = np.asarray(betas, dtype=np.float64)
    betas = betas.reshape(-1, betas.shape[-1]).mean(axis=0)[: model["shapedirs"].shape[2]]
    verts = model["v_template"] + np.einsum("vcd,d->vc", model["shapedirs"][:, :, : len(betas)], betas)
    joints = model["J_regressor"] @ verts
    root = joints[0][None] + np.asarray(transl, dtype=np.float64).reshape(T, 3)
    rots, root = rebase(joints, rots, root, PARENTS_24)
    q = R.from_matrix(rots.reshape(-1, 3, 3)).as_quat().reshape(T, 24, 4)
    for t in range(1, T):  # sign continuity so slerp takes the short way
        flip = (q[t] * q[t - 1]).sum(-1) < 0
        q[t][flip] = -q[t][flip]
    colors = None
    if clothing:
        colors, _ = clothing_colors(verts, model["faces"], joints, model["weights"], gender)
        if colors is not None and not clothing_is_visible(colors):
            colors = np.concatenate([np.tile(SKIN, (len(verts), 1)), np.ones((len(verts), 1))], axis=1)
    return build_glb(verts, model["faces"], joints, PARENTS_24, model["weights"], q, root, fps, JOINT_NAMES_24, colors)


def clothing_is_visible(colors):
    return bool(np.ptp(colors[:, :3], axis=0).max() > 1e-3)
