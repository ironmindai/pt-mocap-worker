"""SMPL-X (body + fingers) in plain numpy: model loading, rest shape, linear blend skinning, GLB export.

No torch / smplx dependency, so it is unit-testable on CPU. The GLB skeleton is the 22 SMPL body joints plus the 30
finger joints (jaw / eye joints are folded into the head), 52 joints in total, in SMPL-X order. Joint rotations are
ABSOLUTE local rotations relative to the SMPL-X template (so a "flat hand" = identity; the model's hands_mean is NOT added).

Finger joints use the same local-frame convention as MANO (verified numerically, see tests/test_hands.py):
left-hand pose = right-hand-equivalent pose with the y and z axis-angle components negated.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

import smpl_glb as sg

FINGERS = ["Index", "Middle", "Pinky", "Ring", "Thumb"]  # SMPL-X order, 3 joints each
BODY_22 = list(sg.JOINT_NAMES_24[:22])
NAMES_52 = BODY_22 + [f"{s}_{f}{k}" for s in ("L", "R") for f in FINGERS for k in (1, 2, 3)]
KEEP = list(range(22)) + list(range(25, 55))     # SMPL-X joint ids that become GLB nodes
# parents of the finger joints: first joint of each finger hangs on the wrist (20 / 21), the rest on the previous joint
_fp = []
for side, wrist in ((0, 20), (1, 21)):
    base = 22 + 15 * side
    for f in range(5):
        for k in range(3):
            _fp.append(wrist if k == 0 else base + 3 * f + k - 1)
PARENTS_52 = np.concatenate([np.array(sg.PARENTS_24[:22], dtype=int), np.array(_fp, dtype=int)])
L_FINGERS = slice(22, 37)   # indices into the 52-joint skeleton
R_FINGERS = slice(37, 52)
L_WRIST, R_WRIST = 20, 21


def load_smplx_npz(path, n_betas=10):
    """SMPLX_NEUTRAL.npz -> dict for this module. weights (V,52): jaw / eyes folded into the head."""
    d = np.load(path, allow_pickle=True)
    w = np.asarray(d["weights"], dtype=np.float64)
    w52 = w[:, KEEP].copy()
    w52[:, 15] += w[:, 22:25].sum(1)
    return {
        "v_template": np.asarray(d["v_template"], dtype=np.float64),
        "shapedirs": np.asarray(d["shapedirs"])[:, :, :n_betas].astype(np.float64),
        "J_regressor": np.asarray(d["J_regressor"], dtype=np.float64)[KEEP],
        "weights": w52,
        "faces": np.asarray(d["f"]).astype(np.uint32),
        "hands_mean_l": np.asarray(d["hands_meanl"], dtype=np.float64).reshape(15, 3),
        "hands_mean_r": np.asarray(d["hands_meanr"], dtype=np.float64).reshape(15, 3),
        "hands_components_l": np.asarray(d["hands_componentsl"], dtype=np.float64),
        "hands_components_r": np.asarray(d["hands_componentsr"], dtype=np.float64),
    }


def rest_shape(model, betas):
    betas = np.asarray(betas, dtype=np.float64)
    betas = (betas.reshape(-1, betas.shape[-1]).mean(0) if betas.ndim > 1 else betas)[: model["shapedirs"].shape[2]]
    v = model["v_template"] + np.einsum("vcd,d->vc", model["shapedirs"][:, :, : len(betas)], betas)
    J = model["J_regressor"] @ v
    hs = model.get("head_scale")      # set by adapt_model_to_smpl: SMPL-length neck->head bone (head pivot only, mesh untouched)
    if hs is not None:
        J = J.copy()
        J[15] = J[12] + (J[15] - J[12]) * hs
    return v, J


def rots_from_params(go, bp, left_hand=None, right_hand=None):
    """Axis-angle -> (T,52,3,3). go (T,3), bp (T,63), left/right_hand (T,15,3) absolute (None = flat hand)."""
    T = len(go)
    aa = np.zeros((T, 52, 3))
    aa[:, 0] = go
    aa[:, 1:22] = np.asarray(bp).reshape(T, 21, 3)
    if left_hand is not None:
        aa[:, L_FINGERS] = left_hand
    if right_hand is not None:
        aa[:, R_FINGERS] = right_hand
    return R.from_rotvec(aa.reshape(-1, 3)).as_matrix().reshape(T, 52, 3, 3)


def posed(model, betas, rots, root, vert_idx=None):
    """LBS (no pose correctives). rots (T,52,3,3) local, root (T,3) pelvis position. -> verts (T,V,3), joints (T,52,3)."""
    v, J = rest_shape(model, betas)
    W = model["weights"]
    if vert_idx is not None:
        v, W = v[vert_idx], W[vert_idx]
    T, n = len(rots), len(J)
    pos = np.zeros((T, n, 3))
    wR = np.zeros_like(rots)
    for j in range(n):
        p = PARENTS_52[j]
        if p < 0:
            wR[:, j] = rots[:, j]
            pos[:, j] = root
        else:
            wR[:, j] = wR[:, p] @ rots[:, j]
            pos[:, j] = pos[:, p] + np.einsum("tij,j->ti", wR[:, p], J[j] - J[p])
    out = np.zeros((T, len(v), 3))
    for j in range(n):
        out += W[:, j][None, :, None] * (np.einsum("tij,vj->tvi", wR[:, j], v - J[j]) + pos[:, j][:, None, :])
    return out, pos


def to_quats(rots):
    T, n = rots.shape[:2]
    q = R.from_matrix(rots.reshape(-1, 3, 3)).as_quat().reshape(T, n, 4)
    for t in range(1, T):
        flip = (q[t] * q[t - 1]).sum(-1) < 0
        q[t][flip] = -q[t][flip]
    return q


def build_person_glb(model, betas, rots, root, color, fps, top=False, gender="male"):
    """Single-character SMPL-X GLB (52 joints). Same vertex-colour painting as the SMPL path (joint ids 0..21 shared)."""
    v, J = rest_shape(model, betas)
    colors, _ = sg.clothing_colors(v, model["faces"], J, model["weights"], gender, shorts_color=color, top=top)
    return sg.build_glb(v, model["faces"], J, PARENTS_52, model["weights"], to_quats(rots), root, fps, NAMES_52, colors)


def person_glb_ground(model, betas, go, bp, tr, left_hand, right_hand, fps, color=None, top=None, gender="neutral"):
    """Stand-alone, ground-aligned (frame 0 at the origin facing +Z, feet on y=0) SMPL-X GLB from GVHMR GLOBAL params,
    same re-basing as the SMPL path. go (T,3), bp (T,63), tr (T,3), left/right_hand (T,15,3) absolute axis-angle."""
    from smpl_bvh import rebase

    rots = rots_from_params(go, bp, left_hand, right_hand)
    v, J = rest_shape(model, betas)
    root = J[0][None] + np.asarray(tr, dtype=np.float64).reshape(-1, 3)
    rots, root = rebase(J, rots, root, PARENTS_52)
    colors, _ = sg.clothing_colors(v, model["faces"], J, model["weights"], gender, shorts_color=color, top=top)
    if not sg.clothing_is_visible(colors):
        colors = np.concatenate([np.tile(sg.SKIN, (len(v), 1)), np.ones((len(v), 1))], axis=1)
    return sg.build_glb(v, model["faces"], J, PARENTS_52, model["weights"], to_quats(rots), root, fps, NAMES_52, colors)


# ---------------------------------------------------------------- SMPL -> SMPL-X shape transfer
SPINE_HEAD_JOINTS = (3, 6, 9, 12, 15)   # extra weight in the beta fit so torso height is matched tightly
BETA_RIDGE = 1e-3                       # keeps the 10 fitted betas small / well conditioned


def _smpl_rest_joints(smpl_model, betas, n=22):
    b = np.asarray(betas, dtype=np.float64)
    b = (b.reshape(-1, b.shape[-1]).mean(0) if b.ndim > 1 else b)[: smpl_model["shapedirs"].shape[2]]
    v = smpl_model["v_template"] + np.einsum("vcd,d->vc", smpl_model["shapedirs"][:, :, : len(b)], b)
    return (smpl_model["J_regressor"] @ v)[:n]


def adapt_model_to_smpl(model, smpl_model, smpl_betas):
    """SMPL betas must NOT be fed to the SMPL-X shape basis (different basis: avatar came out ~7% too large).
    Instead fit SMPL-X betas so that the *shape-induced change* of the 22 body joints (pelvis-relative, vs the neutral
    template of each model) equals the one SMPL predicts (least squares through the linear map J_regressor @ shapedirs).
    Working on the change keeps each model's own joint conventions (the SMPL-X head joint sits ~4 cm higher than SMPL's).
    That residual convention gap is removed by sliding the SMPL-X head joint along the neck->head bone until pelvis-to-head equals SMPL's (`head_scale`, applied in
    rest_shape; the mesh is untouched, only the head pivot moves down like in SMPL).
    Returns (model2 (shallow copy carrying head_scale), betas_x (nb,), info)."""
    nb = model["shapedirs"].shape[2]
    t_smpl = _smpl_rest_joints(smpl_model, smpl_betas)
    t0_smpl = _smpl_rest_joints(smpl_model, np.zeros(1))
    target = (t_smpl - t_smpl[0]) - (t0_smpl - t0_smpl[0])
    JD = np.einsum("jv,vcd->jcd", model["J_regressor"][:22], model["shapedirs"])
    JD = JD - JD[0][None]
    w = np.ones(22)
    w[list(SPINE_HEAD_JOINTS)] = 4.0
    A = (JD * w[:, None, None]).reshape(-1, nb)
    b = (target * w[:, None]).reshape(-1)
    x = np.linalg.solve(A.T @ A + BETA_RIDGE * np.eye(nb), A.T @ b)
    m0 = dict(model)
    _, Jx = rest_shape(m0, x)
    # head pivot: slide J15 along the SMPL-X neck->head direction until pelvis-to-head equals the SMPL pelvis-to-head
    d = Jx[15] - Jx[12]
    c = Jx[12] - Jx[0]
    L = float(np.linalg.norm(t_smpl[15] - t_smpl[0]))
    # |c + s d| = L  ->  (d.d) s^2 + 2 (c.d) s + (c.c - L^2) = 0, positive root
    qa, qb, qc = float(d @ d), 2 * float(c @ d), float(c @ c) - L * L
    m0["head_scale"] = (-qb + np.sqrt(max(qb * qb - 4 * qa * qc, 0.0))) / (2 * qa)
    _, Jf = rest_shape(m0, x)
    ph = lambda j: float(np.linalg.norm(j[15] - j[0]))  # noqa: E731
    info = {"pelvis_head_smpl": ph(t_smpl), "pelvis_head_smplx": ph(Jf), "head_scale": m0["head_scale"],
            "pelvis_head_smplx_raw_betas": ph(rest_shape(model, np.pad(np.asarray(smpl_betas, dtype=np.float64).reshape(-1, 10).mean(0), (0, max(nb - 10, 0))))[1]),
            "rms_joint_delta_err_m": float(np.sqrt(((JD @ x - target) ** 2).sum(1).mean()))}
    return m0, x, info
