"""Hand-pose pipeline pieces that need no GPU: crop geometry, MANO<->SMPL-X conventions, confidence, gap filling,
anatomical clamp and temporal smoothing.

Conventions (all hand poses inside this module are "right-hand-equivalent" until `to_smplx_pose`):
  * HaMeR / MANO right-hand pose (15 joints, absolute local rotations, flat hand = identity) is applied 1:1 to the
    SMPL-X right hand (identical rest frames, verified numerically).
  * A LEFT hand is processed as a horizontally flipped crop, which yields a right-hand-equivalent pose; the SMPL-X left
    hand pose is that pose with the y and z axis-angle components negated (x-mirror of a rotation vector).
  * Fallback / neutral hand = SMPL-X `hands_mean` (relaxed, slightly curled), never a wild pose.
"""
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

MIRROR_AA = np.array([1.0, -1.0, -1.0])
MIRROR_M = np.diag([-1.0, 1.0, 1.0])
PATCH = 256                  # HaMeR input size; the model sees the central 192 columns
RESCALE = 2.5                # HaMeR demo default: crop = 2.5 x tight hand box
ASPECT = 256.0 / 192.0       # BBOX_SHAPE [192, 256]
HAND_BOX_M = 0.18            # side of the "tight hand box" in metres (fixed, so crops keep a stable scale)


# ---------------------------------------------------------------- hands input / fist mode
HANDS_MODES = ("estimate", "fist")


def parse_hands_mode(value):
    """`hands` input -> None (off: false / null) | "estimate" (true / "estimate") | "fist". Anything else -> ValueError."""
    if value is None or value is False:
        return None
    if value is True:
        return "estimate"
    if isinstance(value, str) and value.strip().lower() in HANDS_MODES:
        return value.strip().lower()
    raise ValueError(f'hands must be false|null, true, "estimate" or "fist" (got {value!r})')


# Closed boxer's fist, right-hand-equivalent absolute MANO/SMPL-X axis-angle (15 joints: index, middle, pinky, ring, thumb x 3).
# Taken from a real MANO training pose (MANO_RIGHT.pkl `hands_mean + hands_coeffs @ hands_components`, pose #515: the pose with
# tightly curled fingers - ~85-100 deg per joint about the local +z axis - whose thumb lies over the index / middle fingers'
# middle phalanges, i.e. a boxer's fist with the thumb outside), rounded to 3 decimals. Verified by rendering the SMPL-X mesh
# (see README / playground). The left hand is its x-mirror via `to_smplx_pose` (y, z negated).
FIST_RIGHT = np.array([
    [0.009, -0.346, 0.928], [0.176, 0.127, 1.749], [-0.053, 0.069, 0.757],     # index
    [-0.499, -0.276, 1.524], [-0.166, -0.170, 1.291], [0.130, -0.043, 0.822],  # middle
    [-1.420, 0.239, 1.505], [-0.128, 0.384, 0.903], [-0.270, 0.373, 2.050],    # pinky
    [-0.400, 0.414, 1.624], [-0.560, 0.298, 1.120], [-0.252, -0.115, 1.049],   # ring
    [0.847, -0.589, -0.196], [-0.637, 0.034, 0.006], [0.522, -0.063, 0.094],   # thumb
])


def fist_pose(T):
    """(left, right) SMPL-X absolute axis-angle fist poses, each (T,15,3), constant over time."""
    right = np.tile(FIST_RIGHT, (T, 1, 1))
    return to_smplx_pose(right, True), right


# ---------------------------------------------------------------- conventions
def to_smplx_pose(aa_right_equiv, is_left):
    """(…,15,3) right-hand-equivalent axis-angle -> SMPL-X (left_hand_pose | right_hand_pose) axis-angle."""
    return np.asarray(aa_right_equiv) * (MIRROR_AA if is_left else 1.0)


def mirror_rotmat(Rm):
    """Rotation matrices mirrored through the x = 0 plane (left <-> right equivalent): M R M."""
    return MIRROR_M @ Rm @ MIRROR_M


def rotmat_to_aa(Rm):
    s = Rm.shape
    return R.from_matrix(Rm.reshape(-1, 3, 3)).as_rotvec().reshape(s[:-2] + (3,))


# ---------------------------------------------------------------- crop geometry
def project(K, X):
    X = np.asarray(X, dtype=np.float64)
    z = np.maximum(X[..., 2], 1e-3)
    return np.stack([K[0, 0] * X[..., 0] / z + K[0, 2], K[1, 1] * X[..., 1] / z + K[1, 2]], axis=-1)


def crop_size_px(K, depth_m, hand_box_m=HAND_BOX_M):
    """HaMeR crop side (px, in the unflipped image) for a tight hand box of `hand_box_m` at `depth_m`."""
    s = K[0, 0] * hand_box_m / np.maximum(depth_m, 0.2)
    return RESCALE * s * ASPECT


def hand_centers(joints52, K, width, height):
    """joints52 (T,52,3) camera-frame (OpenCV) SMPL-X joints. -> centers_px (T,2 hands [L,R],2), size_px (T,2),
    visible (T,2) = fraction (0..1) of the central 0.75 of the crop that lies inside the image (0 when the hand is
    behind the camera)."""
    from smplx_np import L_FINGERS, R_FINGERS, L_WRIST, R_WRIST

    T = len(joints52)
    cen = np.zeros((T, 2, 2))
    size = np.zeros((T, 2))
    vis = np.zeros((T, 2))
    for h, (wr, fi) in enumerate(((L_WRIST, L_FINGERS), (R_WRIST, R_FINGERS))):
        pts = np.concatenate([joints52[:, wr:wr + 1], joints52[:, fi]], axis=1)  # (T,16,3)
        c3 = pts.mean(1)
        cen[:, h] = project(K, c3)
        size[:, h] = crop_size_px(K, c3[:, 2])
        half = 0.75 * size[:, h] / 2
        x0, x1 = np.clip(cen[:, h, 0] - half, 0, width), np.clip(cen[:, h, 0] + half, 0, width)
        y0, y1 = np.clip(cen[:, h, 1] - half, 0, height), np.clip(cen[:, h, 1] + half, 0, height)
        vis[:, h] = ((x1 - x0) * (y1 - y0)) / np.maximum((2 * half) ** 2, 1e-6)
        vis[c3[:, 2] < 0.3, h] = 0.0
    return cen, size, vis


def extract_patch(img, center, side, flip):
    """Square crop of `side` px (>=8) around `center` (x,y) resized to PATCH x PATCH (uint8, same channel order as img),
    zero-padded outside the image, mirrored horizontally when `flip` (left hands).
    Returns (patch, (cx_int, cy_int, n)) where n = integer ROI side, for mapping crop keypoints back."""
    import cv2

    H, W = img.shape[:2]
    n = int(max(8, round(side)))
    cx, cy = int(round(center[0])), int(round(center[1]))
    x0, y0 = cx - n // 2, cy - n // 2
    roi = np.zeros((n, n, img.shape[2]), dtype=img.dtype)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + n, W), min(y0 + n, H)
    if sx1 > sx0 and sy1 > sy0:
        roi[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    interp = cv2.INTER_AREA if n > PATCH else cv2.INTER_LINEAR
    patch = cv2.resize(roi, (PATCH, PATCH), interpolation=interp)
    if flip:
        patch = patch[:, ::-1]
    return np.ascontiguousarray(patch), (x0, y0, n)


def patch_to_image(pts_patch, geom, flip):
    """(…,2) patch-pixel coordinates (as seen by the network, i.e. after the flip) -> image pixels."""
    x0, y0, n = geom
    p = np.array(pts_patch, dtype=np.float64)
    if flip:
        p[..., 0] = PATCH - 1 - p[..., 0]
    return p * (n / PATCH) + np.array([x0, y0])


def sharpness(patch_rgb):
    """Variance of the Laplacian of the central region (blur / low-detail measure; higher = sharper)."""
    import cv2

    g = cv2.cvtColor(np.ascontiguousarray(patch_rgb), cv2.COLOR_RGB2GRAY) if patch_rgb.shape[2] == 3 else patch_rgb[..., 0]
    g = g[32:-32, 32:-32]
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


# ---------------------------------------------------------------- confidence
# Quality thresholds (HaMeR gives stiff, splayed fingers when the hand is occluded, e.g. resting on the head).
AGREE_LO_DEG = 45.0        # hand-model vs body-model wrist disagreement: fully trusted below this ...
AGREE_HI_DEG = 90.0        # ... and rejected (confidence 0) above this. Was 60/120: too lenient for occluded contact.
CONF_BLEND_LO = 0.35       # frame confidence at/below which the estimate gets zero weight (relaxed neutral hand)
CONF_BLEND_HI = 0.80       # frame confidence at/above which the estimate is fully trusted; in between = partial blend
FUSE_THRESH = 0.35         # frames below this are "missing" (bridged / held / faded to neutral), see fuse_hand
FINGER_SMOOTH_WINDOW = 9   # quaternion SavGol window (frames @30 fps) for finger poses; was 5
MAX_EST_WEIGHT = 0.9       # never trust HaMeR 100%: keeps a little relaxed-hand curl, damps splay


def wrist_agreement_deg(R_body_wrist, R_hand_wrist):
    """Angle (deg) between the body model's wrist orientation and the hand model's, both in the camera frame."""
    rel = np.swapaxes(R_body_wrist, -1, -2) @ R_hand_wrist
    return np.degrees(np.linalg.norm(R.from_matrix(rel.reshape(-1, 3, 3)).as_rotvec(), axis=-1)).reshape(rel.shape[:-2])


def _ramp(x, lo, hi):
    """1 for x <= lo, 0 for x >= hi, smooth in between."""
    t = np.clip((np.asarray(x, dtype=np.float64) - lo) / (hi - lo), 0, 1)
    return 1 - t * t * (3 - 2 * t)


def frame_confidence(vis, sharp, agree_deg, observed=None):
    """Per-frame confidence (T,) for one hand in [0,1]: visibility x sharpness (relative to this hand's own median) x
    agreement of the hand model's wrist with the body model's wrist. `observed` False -> 0."""
    sharp = np.asarray(sharp, dtype=np.float64)
    med = np.median(sharp[sharp > 0]) if (sharp > 0).any() else 1.0
    s = np.clip(sharp / (0.30 * med), 0, 1)
    c = np.clip(np.asarray(vis) / 0.6, 0, 1) * s * _ramp(agree_deg, AGREE_LO_DEG, AGREE_HI_DEG)
    if observed is not None:
        c = c * np.asarray(observed, dtype=bool)
    return c


# ---------------------------------------------------------------- anatomical prior
class HandPrior:
    """Range of plausible right-hand poses in MANO PCA space (from the MANO training poses shipped inside MANO_RIGHT.pkl).
    Poses are clamped per PCA coefficient to the observed range (+margin), which removes impossible finger bends while
    keeping every pose the real data contains."""

    def __init__(self, components, mean, coeffs, margin=0.25, pct=(0.5, 99.5)):
        self.components = np.asarray(components, dtype=np.float64)       # (45,45): aa = mean + c @ components
        self.mean = np.asarray(mean, dtype=np.float64).reshape(45)
        lo, hi = np.percentile(coeffs, pct[0], axis=0), np.percentile(coeffs, pct[1], axis=0)
        pad = margin * (hi - lo)
        self.lo, self.hi = lo - pad, hi + pad
        self.inv = np.linalg.pinv(self.components)

    @classmethod
    def from_mano_pkl(cls, path, **kw):
        import smpl_glb as sg

        with open(path, "rb") as f:
            d = sg._Unpickler(f, encoding="latin1").load()
        return cls(d["hands_components"], d["hands_mean"], d["hands_coeffs"], **kw)

    def clamp(self, aa):
        """aa (…,15,3) right-hand-equivalent absolute pose -> clamped copy."""
        flat = np.asarray(aa, dtype=np.float64).reshape(-1, 45)
        c = (flat - self.mean) @ self.inv
        c = np.clip(c, self.lo, self.hi)
        return (self.mean + c @ self.components).reshape(np.shape(aa))


# ---------------------------------------------------------------- temporal fusion
def _quat_continuous(q):
    q = q.copy()
    for t in range(1, len(q)):
        flip = (q[t] * q[t - 1]).sum(-1) < 0
        q[t][flip] = -q[t][flip]
    return q


def fuse_hand(aa, conf, neutral, prior=None, thresh=FUSE_THRESH, max_gap=15, hold_tau=10.0, window=FINGER_SMOOTH_WINDOW,
              outlier_rad=1.0):
    """One hand over time. aa (T,15,3) right-hand-equivalent absolute poses, conf (T,) in [0,1], neutral (15,3) relaxed
    pose. Returns (pose (T,15,3), weight (T,)) where weight is the final blend weight of the estimate vs the neutral hand.

    1) clamp to the anatomical prior; 2) frames below `thresh` (and temporal outliers vs a running median) are treated
    as missing; 3) gaps up to `max_gap` frames are bridged by quaternion interpolation, longer gaps (and the clip ends)
    hold the last good pose with exponentially decaying weight (tau = `hold_tau` frames) towards the neutral hand;
    4) quaternion Savitzky-Golay (sign-continuous); 5) final = nlerp(neutral, estimate, smoothed weight).
    Per-frame confidence is first mapped to a blend weight (CONF_BLEND_LO..HI ramp, capped at MAX_EST_WEIGHT), so an unreliable
    frame is pulled towards the relaxed neutral hand instead of being trusted as is."""
    T = len(aa)
    aa = np.asarray(aa, dtype=np.float64).copy()
    conf = np.clip(np.asarray(conf, dtype=np.float64), 0, 1)
    conf_raw = conf
    conf = MAX_EST_WEIGHT * np.clip((conf - CONF_BLEND_LO) / (CONF_BLEND_HI - CONF_BLEND_LO), 0, 1)
    if prior is not None:
        aa = prior.clamp(aa)
    q = R.from_rotvec(aa.reshape(-1, 3)).as_quat().reshape(T, 15, 4)
    good = conf_raw >= thresh
    # temporal outliers: pose far (mean joint angle) from the median of its neighbours
    if good.sum() >= 5:
        idx = np.where(good)[0]
        for k in idx:
            nb = idx[(np.abs(idx - k) <= 3) & (idx != k)]
            if len(nb) >= 3:
                med = np.median(aa[nb], axis=0)
                dq = R.from_rotvec(aa[k]).inv() * R.from_rotvec(med)
                if np.linalg.norm(dq.as_rotvec(), axis=1).mean() > outlier_rad:
                    good[k] = False
    w = np.where(good, conf, 0.0)
    if not good.any():
        return np.tile(neutral, (T, 1, 1)), np.zeros(T)
    gi = np.where(good)[0]
    qc = np.zeros_like(q)
    wf = np.zeros(T)
    for t in range(T):
        if good[t]:
            qc[t], wf[t] = q[t], w[t]
            continue
        j = np.searchsorted(gi, t)
        prev = gi[j - 1] if j > 0 else None
        nxt = gi[j] if j < len(gi) else None
        if prev is not None and nxt is not None and nxt - prev <= max_gap + 1:
            a = (t - prev) / (nxt - prev)                       # bridge: interpolate pose and weight
            qa, qb = q[prev], q[nxt]
            qb = np.where(((qa * qb).sum(-1) < 0)[:, None], -qb, qb)
            qi = (1 - a) * qa + a * qb
            qc[t] = qi / np.linalg.norm(qi, axis=1, keepdims=True)
            wf[t] = (1 - a) * w[prev] + a * w[nxt]
        else:                                                    # hold the nearest good pose, fading to neutral
            src, dist = (prev, t - prev) if (nxt is None or (prev is not None and t - prev <= nxt - t)) else (nxt, nxt - t)
            qc[t] = q[src]
            wf[t] = w[src] * np.exp(-dist / hold_tau)
    qc = _quat_continuous(qc)
    wn = _window(window, T)
    if wn:
        qc = savgol_filter(qc, wn, 2, axis=0)
        qc /= np.linalg.norm(qc, axis=-1, keepdims=True)
        wf = np.clip(savgol_filter(wf, wn, 2), 0, 1)
    qn = R.from_rotvec(np.asarray(neutral, dtype=np.float64)).as_quat()
    qn = np.where(((qc * qn[None]).sum(-1) < 0)[..., None], -qn[None], qn[None])
    out = (1 - wf)[:, None, None] * qn + wf[:, None, None] * qc
    out /= np.linalg.norm(out, axis=-1, keepdims=True)
    return R.from_quat(out.reshape(-1, 4)).as_rotvec().reshape(T, 15, 3), wf


def _window(w, n):
    w = min(w, n)
    if w % 2 == 0:
        w -= 1
    return w if w > 2 else None
