"""SMPL / SMPL-X body pose sequence -> BVH.

Uses the first 22 joints (pelvis .. wrists), which are identical in SMPL, SMPL-H and SMPL-X.
Rest-pose joint offsets come from the neutral SMPL-X model (J_regressor @ shaped template),
so the BVH skeleton has the subject's body proportions (from GVHMR's predicted betas).

Conventions:
  * Y-up, right-handed, subject faces +Z, SMPL "left" is +X (SMPL native convention).
  * Units: meters multiplied by `scale` (default 1.0).
  * Root has 6 channels (Xposition Yposition Zposition Zrotation Yrotation Xrotation), every other
    joint has 3 (Zrotation Yrotation Xrotation), i.e. R = Rz * Ry * Rx (intrinsic ZYX).
  * The sequence is re-based so frame 0 starts at XZ origin, facing +Z, with the feet on y=0.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2", "L_Ankle", "R_Ankle",
    "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder",
    "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
]
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
N_JOINTS = 22
SOLE_OFFSET_M = 0.03  # foot joint -> sole of the foot, used to put the feet on the floor


def rest_joints(smplx_npz_path, betas, n_betas=10):
    """(22, 3) rest-pose joint positions for the given shape coefficients."""
    m = np.load(smplx_npz_path, allow_pickle=True)
    betas = np.asarray(betas, dtype=np.float64).reshape(-1)[:n_betas]
    v = m["v_template"] + np.einsum("vcd,d->vc", m["shapedirs"][:, :, : len(betas)], betas)
    return m["J_regressor"][:N_JOINTS] @ v


def _fk(rest, rots, root_pos, parents=PARENTS):
    """Forward kinematics. rest (J,3), rots (T,J,3,3) local, root_pos (T,3) -> world joints (T,J,3)."""
    T, nj = rots.shape[0], rots.shape[1]
    world_R = np.zeros_like(rots)
    pos = np.zeros((T, nj, 3))
    for j in range(nj):
        p = parents[j]
        if p < 0:
            world_R[:, j] = rots[:, j]
            pos[:, j] = root_pos
        else:
            world_R[:, j] = world_R[:, p] @ rots[:, j]
            pos[:, j] = pos[:, p] + np.einsum("tij,j->ti", world_R[:, p], rest[j] - rest[p])
    return pos


def rebase(rest, rots, root, parents=PARENTS):
    """Re-base a motion: XZ of frame 0 to origin, heading of frame 0 to +Z, lowest foot over the clip on y=0.

    rest (J,3), rots (T,J,3,3) local rotations, root (T,3) posed pelvis positions. Returns new (rots, root).
    Joint indices 1/2 are the hips and 7,8,10,11 the ankles/feet (same in SMPL, SMPL-H, SMPL-X).
    """
    rots, root = rots.copy(), root.copy()
    joints0 = _fk(rest, rots[:1], root[:1], parents)[0]
    across = joints0[1] - joints0[2]  # left hip - right hip, +X when facing +Z
    phi = np.arctan2(across[2], across[0])
    c, s = np.cos(phi), np.sin(phi)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    root = (root - np.array([root[0, 0], 0.0, root[0, 2]])) @ Ry.T
    rots[:, 0] = Ry[None] @ rots[:, 0]
    joints = _fk(rest, rots, root, parents)
    ground = joints[:, [7, 8, 10, 11], 1].min() - SOLE_OFFSET_M
    root[:, 1] -= ground
    return rots, root


def smpl_to_bvh(global_orient, body_pose, transl, betas, smplx_npz_path, fps, scale=1.0):
    """Build a BVH string.

    global_orient (T,3), body_pose (T,63), transl (T,3) axis-angle / meters, betas (T,10) or (10,).
    """
    global_orient = np.asarray(global_orient, dtype=np.float64).reshape(-1, 3)
    T = global_orient.shape[0]
    aa = np.concatenate([global_orient, np.asarray(body_pose, dtype=np.float64).reshape(T, -1)[:, :63]], axis=1)
    rots = R.from_rotvec(aa.reshape(-1, 3)).as_matrix().reshape(T, N_JOINTS, 3, 3)
    betas = np.asarray(betas, dtype=np.float64)
    betas = betas.reshape(-1, betas.shape[-1]).mean(axis=0)
    rest = rest_joints(smplx_npz_path, betas)
    root = rest[0][None] + np.asarray(transl, dtype=np.float64).reshape(T, 3)  # posed pelvis position

    rots, root = rebase(rest, rots, root)

    eul = R.from_matrix(rots.reshape(-1, 3, 3)).as_euler("ZYX", degrees=True).reshape(T, N_JOINTS, 3)

    children = {j: [k for k in range(N_JOINTS) if PARENTS[k] == j] for j in range(N_JOINTS)}
    offs = np.zeros_like(rest)
    for j in range(1, N_JOINTS):
        offs[j] = rest[j] - rest[PARENTS[j]]
    offs *= scale

    lines = ["HIERARCHY"]

    def emit(j, depth):
        pad = "\t" * depth
        lines.append(f"{pad}{'ROOT' if j == 0 else 'JOINT'} {JOINT_NAMES[j]}")
        lines.append(pad + "{")
        o = offs[j]
        lines.append(f"{pad}\tOFFSET {o[0]:.6f} {o[1]:.6f} {o[2]:.6f}")
        if j == 0:
            lines.append(f"{pad}\tCHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation")
        else:
            lines.append(f"{pad}\tCHANNELS 3 Zrotation Yrotation Xrotation")
        if children[j]:
            for k in children[j]:
                emit(k, depth + 1)
        else:
            d = offs[j] / (np.linalg.norm(offs[j]) + 1e-9) * 0.1 * scale
            lines.append(f"{pad}\tEnd Site")
            lines.append(pad + "\t{")
            lines.append(f"{pad}\t\tOFFSET {d[0]:.6f} {d[1]:.6f} {d[2]:.6f}")
            lines.append(pad + "\t}")
        lines.append(pad + "}")

    order = []

    def dfs(j):
        order.append(j)
        for k in children[j]:
            dfs(k)

    dfs(0)
    emit(0, 0)
    lines += ["MOTION", f"Frames: {T}", f"Frame Time: {1.0 / fps:.8f}"]
    for t in range(T):
        vals = list(root[t] * scale) + list(eul[t][order].reshape(-1))
        lines.append(" ".join(f"{v:.5f}" for v in vals))
    return "\n".join(lines) + "\n"
