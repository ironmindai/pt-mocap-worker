"""Temporal smoothing of an SMPL pose sequence (quaternion Savitzky-Golay)."""
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

POLYORDER = 2
# window sizes in frames at 30 fps: (rotations, root translation)
LEVELS = {"light": (5, 5), "medium": (7, 7), "strong": (11, 9)}


def _window(w, n):
    """Largest odd window <= min(w, n); None if too short to filter."""
    w = min(w, n)
    if w % 2 == 0:
        w -= 1
    return w if w > POLYORDER else None


def smooth_quats(q):
    """q (T,4) xyzw -> sign-continuous (not smoothed) copy."""
    q = q.copy()
    for t in range(1, len(q)):
        if np.dot(q[t], q[t - 1]) < 0:
            q[t] = -q[t]
    return q


def smooth_motion(global_orient, body_pose, transl, level="medium"):
    """Smooth axis-angle poses (T,3)+(T,63) and translation (T,3). Returns arrays of the same shapes.

    Per joint: axis-angle -> quaternion, sign continuity (flip when dot < 0), Savitzky-Golay per component
    (polyorder 2), renormalize, back to axis-angle. Translation gets a gentler SavGol.
    """
    go = np.asarray(global_orient, dtype=np.float64).reshape(-1, 3)
    T = go.shape[0]
    bp = np.asarray(body_pose, dtype=np.float64).reshape(T, -1)
    tr = np.asarray(transl, dtype=np.float64).reshape(T, 3)
    if level == "off":
        return go, bp, tr
    if level not in LEVELS:
        raise ValueError(f"smoothing must be one of {sorted(LEVELS) + ['off']}")
    w_rot, w_tr = (_window(w, T) for w in LEVELS[level])

    aa = np.concatenate([go, bp], axis=1).reshape(T, -1, 3)  # (T, J, 3)
    out = aa.copy()
    if w_rot:
        for j in range(aa.shape[1]):
            q = smooth_quats(R.from_rotvec(aa[:, j]).as_quat())
            q = savgol_filter(q, w_rot, POLYORDER, axis=0)
            q /= np.linalg.norm(q, axis=1, keepdims=True)
            out[:, j] = R.from_quat(q).as_rotvec()
    tr_out = savgol_filter(tr, w_tr, POLYORDER, axis=0) if w_tr else tr
    out = out.reshape(T, -1)
    return out[:, :3], out[:, 3:], tr_out
