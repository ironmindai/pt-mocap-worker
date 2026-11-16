"""SMPL-X body (22 joints) -> skinned, animated GLB, reusing smpl_glb.build_glb / clothing_colors.

Finger / jaw / eye joints are folded into their parents (wrists / head), so the GLB has the same 22-joint body
skeleton as SMPL (joints 0..21 share names and order). Used for Human3R output, which predicts SMPL-X.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

import smpl_glb as sg
from motion import smooth_motion

PARENTS_22 = np.array(sg.PARENTS_24[:22])
NAMES_22 = list(sg.JOINT_NAMES_24[:22])


def rest_body(bm, betas):
    """bm: smplx SMPLX model (neutral, num_betas >= len(betas)). -> verts (V,3), faces, joints22 (22,3), weights (V,22)."""
    import torch

    nb = int(bm.num_betas)
    b = np.zeros(nb, dtype=np.float32)
    b[: min(nb, len(betas))] = np.asarray(betas, dtype=np.float32)[:nb]
    z = lambda d: torch.zeros(1, d)  # noqa: E731
    with torch.no_grad():
        o = bm(betas=torch.from_numpy(b)[None], global_orient=z(3), body_pose=z(63), left_hand_pose=z(45),
               right_hand_pose=z(45), jaw_pose=z(3), leye_pose=z(3), reye_pose=z(3), expression=z(10))
    verts = o.vertices[0].numpy().astype(np.float64)
    joints = o.joints[0, :22].numpy().astype(np.float64)
    w = bm.lbs_weights.numpy().astype(np.float64)  # (V,55)
    w22 = w[:, :22].copy()
    w22[:, 15] += w[:, 22:25].sum(1)   # jaw, eyes -> head
    w22[:, 20] += w[:, 25:40].sum(1)   # left fingers -> left wrist
    w22[:, 21] += w[:, 40:55].sum(1)   # right fingers -> right wrist
    return verts, np.asarray(bm.faces, dtype=np.int64), joints, w22


def fill_gaps(x, observed):
    """Linear interpolation (nearest at the ends) of unobserved frames along axis 0."""
    x = np.array(x, dtype=np.float64, copy=True)
    idx = np.where(observed)[0]
    if len(idx) == 0 or len(idx) == len(x):
        return x
    flat = x.reshape(len(x), -1)
    for c in range(flat.shape[1]):
        flat[:, c] = np.interp(np.arange(len(x)), idx, flat[idx, c])
    return flat.reshape(x.shape)


def person_glb(bm, rotvec, betas, root, observed, fps, color, smoothing="medium", to_out=None):
    """rotvec (T,53,3) SMPL-X axis-angle, betas (nb,) mean shape, root (T,3) pelvis position (metres) in the frame of
    `rotvec`. to_out: optional 3x3 rotation applied to the whole figure (e.g. OpenCV camera -> Y-up viewer frame)."""
    T = len(rotvec)
    rv = fill_gaps(rotvec, observed)
    root = fill_gaps(root, observed)
    go, bp = rv[:, 0], rv[:, 1:22].reshape(T, 63)
    go, bp, root = smooth_motion(go, bp, root, smoothing)
    aa = np.concatenate([go.reshape(T, 1, 3), bp.reshape(T, 21, 3)], axis=1)
    rots = R.from_rotvec(aa.reshape(-1, 3)).as_matrix().reshape(T, 22, 3, 3)
    if to_out is not None:
        rots[:, 0] = to_out @ rots[:, 0]
        root = root @ to_out.T
    verts, faces, joints, w = rest_body(bm, betas)
    colors, _ = sg.clothing_colors(verts, faces, joints, w, "male", shorts_color=color, top=False)
    q = R.from_matrix(rots.reshape(-1, 3, 3)).as_quat().reshape(T, 22, 4)
    for t in range(1, T):
        flip = (q[t] * q[t - 1]).sum(-1) < 0
        q[t][flip] = -q[t][flip]
    return sg.build_glb(verts, faces, joints, PARENTS_22, w, q, root, fps, NAMES_22, colors)
