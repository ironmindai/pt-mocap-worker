"""Video + per-person camera-frame SMPL-X body motion -> per-frame finger poses (HaMeR) -> fused / smoothed hand poses.

Crops come from the projected SMPL-X hand joints of each fighter (no extra detector). A second pass re-centres every
crop on HaMeR's own 2D keypoints. Confidence = visibility x sharpness x wrist agreement; see hands.fuse_hand.
"""
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

import hands as H
import smplx_np as sx


def wrist_world_rots(rots):
    """Camera-frame orientation of both wrists from local rotations (T,52,3,3) -> (T,2 [L,R],3,3)."""
    T = len(rots)
    out = np.zeros((T, 2, 3, 3))
    for h, wr in enumerate((sx.L_WRIST, sx.R_WRIST)):
        chain = []
        j = wr
        while j >= 0:
            chain.append(j)
            j = sx.PARENTS_52[j]
        M = np.tile(np.eye(3), (T, 1, 1))
        for j in reversed(chain):
            M = M @ rots[:, j]
        out[:, h] = M
    return out


def estimate_hands(frames, persons, estimator, prior, K, width, height, passes=2, chunk=16, log=print):
    """frames: iterable of RGB uint8 (H,W,3) frames, one per motion frame (length T).
    persons: list of dicts with 'joints52' (T,52,3) camera frame, 'rots' (T,52,3,3) local body rotations (camera frame),
    'observed' (T,) bool. Returns list of dicts (see below) and timing info.
    Per person: raw_aa (T,2,15,3) right-hand-equivalent poses [L,R], conf (T,2), agree_deg, sharp, vis, kp_img (T,2,21,2)."""
    P = len(persons)
    T = len(persons[0]["joints52"])
    cen, size, vis, wrot = [], [], [], []
    for p in persons:
        c, s, v = H.hand_centers(p["joints52"], K, width, height)
        cen.append(c), size.append(s), vis.append(v), wrot.append(wrist_world_rots(p["rots"]))
    raw = np.zeros((P, T, 2, 15, 3))
    kp_img = np.zeros((P, T, 2, 21, 2))
    sharp = np.zeros((P, T, 2))
    agree = np.full((P, T, 2), 180.0)
    t0 = time.time()
    n_pred = 0

    def run_batch(imgs, ts):
        nonlocal n_pred
        items = [(i, t, p, h) for i, (img, t) in enumerate(zip(imgs, ts)) for p in range(P) for h in (0, 1)
                 if vis[p][t, h] > 0.05]
        if not items:
            return
        cur = {(t, p, h): cen[p][t, h].copy() for _, t, p, h in items}
        res = None
        for ps in range(passes):
            patches, geoms = [], []
            for i, t, p, h in items:
                pt, g = H.extract_patch(imgs[i], cur[(t, p, h)], size[p][t, h], flip=(h == 0))
                patches.append(pt), geoms.append(g)
            res = estimator.predict(np.stack(patches))
            n_pred += len(items)
            if ps + 1 < passes:  # re-centre on the predicted keypoints (limited shift)
                for k, (i, t, p, h) in enumerate(items):
                    kp = H.patch_to_image(res["kp2d"][k], geoms[k], flip=(h == 0))
                    shift = kp.mean(0) - cur[(t, p, h)]
                    lim = 0.3 * geoms[k][2]
                    cur[(t, p, h)] = cur[(t, p, h)] + np.clip(shift, -lim, lim)
        for k, (i, t, p, h) in enumerate(items):
            hp = res["hand_pose"][k]
            if h == 0:  # left hand: network saw the mirrored crop -> right-hand-equivalent pose already
                G = H.mirror_rotmat(res["global_orient"][k])
            else:
                G = res["global_orient"][k]
            raw[p, t, h] = H.rotmat_to_aa(hp)
            kp_img[p, t, h] = H.patch_to_image(res["kp2d"][k], geoms[k], flip=(h == 0))
            sharp[p, t, h] = H.sharpness(patches[k])
            agree[p, t, h] = H.wrist_agreement_deg(wrot[p][t, h], G)

    buf, ts = [], []
    for t, img in enumerate(frames):
        if t >= T:
            break
        buf.append(img), ts.append(t)
        if len(buf) == chunk:
            run_batch(buf, ts)
            buf, ts = [], []
    if buf:
        run_batch(buf, ts)
    info = {"hand_crops": int(n_pred), "hamer_s": round(time.time() - t0, 2)}

    out = []
    for p in range(P):
        conf = np.zeros((T, 2))
        for h in (0, 1):
            conf[:, h] = H.frame_confidence(vis[p][:, h], sharp[p][:, h], agree[p][:, h], persons[p].get("observed"))
        out.append({"raw_aa": raw[p], "conf": conf, "agree_deg": agree[p], "sharp": sharp[p], "vis": vis[p],
                    "kp_img": kp_img[p], "centers_px": cen[p]})
    return out, info


def fuse_person(est, neutral_r, prior=None, window=H.FINGER_SMOOTH_WINDOW, **kw):
    """est: one dict from estimate_hands. -> left_aa, right_aa (T,15,3) in SMPL-X convention (absolute), w (T,2) blend weights."""
    T = len(est["conf"])
    poses, ws = [], []
    for h in (0, 1):
        pose, w = H.fuse_hand(est["raw_aa"][:, h], est["conf"][:, h], neutral_r, prior=prior, window=window, **kw)
        poses.append(H.to_smplx_pose(pose, is_left=(h == 0)))
        ws.append(w)
    return poses[0], poses[1], np.stack(ws, 1)
