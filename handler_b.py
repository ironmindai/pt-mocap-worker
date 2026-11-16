"""RunPod serverless handler (LAB, option B): multi-person video -> SMPL-X world reconstruction with Human3R.

One feed-forward pass reconstructs EVERY person in the clip in one shared frame (no per-person tracking/cropping
pipeline, no per-person GVHMR passes). License: Human3R = CC BY-NC-SA 4.0 + NAVER Multi-HMR non-commercial -> RESEARCH
EVALUATION ONLY, not shippable in the paid product.

Input (same style as the dual handler):
  video_url | video_b64   the video
  static_cam   bool, default true. True: camera poses are forced to identity (world == camera frame of frame 0).
               False: Human3R's own per-frame camera poses (c2w) are applied to build the world frame.
  size         int, default 512 (network input long side; Human3R's default)
  reset_interval  int, default 100 (Human3R state reset interval, frames)
  min_frames   int, default 8: persons seen in fewer frames are dropped
  max_persons  int, default 4: keep the N most-observed persons
  smoothing    "medium" (default) | "light" | "strong" | "off": applied to the GLB only (npz params are raw)
  person_colors  list of "#rrggbb" (shorts colours of the per-person GLBs)
  model        "896L" (default) | other checkpoint name baked in the image (only 896L is baked)

Output: {"persons": [{rank, smpl_id, observed_frames, params_npz_b64, glb_b64, ...}], "K_fullimg", "fps": 30, "c2w",
         "timing": {...}, ...}. npz keys follow the dual handler: incam_{global_orient,body_pose,betas,transl},
         global_{...}, K_fullimg, observed, fps, plus joints22_incam (T,22,3), pelvis_incam (T,3), c2w (T,4,4).
         SMPL-X axis-angle (smplx.SMPLX(num_betas=nb)); unobserved frames are NaN (see `observed`).
         incam = per-frame camera frame (OpenCV: x right, y down, z forward, metres). GLBs are in that camera frame
         converted with diag(-1,-1,1) (Y up, +Z forward) exactly like experiment A's scene_incam.glb.
"""
import base64
import glob
import io
import os
import shutil
import tempfile
import time
import traceback

import numpy as np
import runpod

import human3r_infer as H
from handler import MODEL_FPS, fetch_video, normalize_video  # shared video I/O (30 fps normalisation) with the dual handler

PERSON_COLORS = ["#C0201A", "#1F4FC4", "#1E8E3E", "#C98A00"]
CKPT_DIR = os.environ.get("H3R_CKPT_DIR", "/workspace/Human3R/src")
CV2OUT = np.diag([-1.0, -1.0, 1.0])
_state = {}


def _load(device):
    if "model" in _state:
        return 0.0
    t = time.time()
    H.setup_path()
    _state["model"] = H.load_model(os.path.join(CKPT_DIR, "human3r_896L.pth"), device)
    return round(time.time() - t, 1)


def _bm(nb):
    key = ("bm", nb)
    if key not in _state:
        import smplx

        _state[key] = smplx.create(os.path.join(H.H3R_DIR, "src", "models"), "smplx", gender="neutral",
                                   use_pca=False, flat_hand_mean=True, num_betas=nb)
    return _state[key]


def _c2w_stats(c2w):
    from scipy.spatial.transform import Rotation as R

    t = c2w[:, :3, 3]
    ang = np.degrees(np.linalg.norm(R.from_matrix(c2w[:, :3, :3]).as_rotvec(), axis=1))
    return {"max_translation_m": float(np.linalg.norm(t - t[0], axis=1).max()), "max_rotation_deg": float(ang.max()),
            "c2w_first": c2w[0].tolist(), "c2w_last": c2w[-1].tolist()}


def _to_world(rotvec, pelvis_cam, transl_std, c2w):
    """Camera-frame SMPL-X (global_orient, transl) -> world via per-frame c2w. Unobserved (NaN) frames stay NaN."""
    from scipy.spatial.transform import Rotation as R

    T = len(rotvec)
    ok = ~np.isnan(rotvec[:, 0, 0])
    go_w = np.full((T, 3), np.nan, dtype=np.float32)
    tr_w = np.full((T, 3), np.nan, dtype=np.float32)
    if ok.any():
        Rg = R.from_rotvec(rotvec[ok, 0].astype(np.float64)).as_matrix()
        go_w[ok] = R.from_matrix(c2w[ok, :3, :3] @ Rg).as_rotvec()
        pel_w = np.einsum("tij,tj->ti", c2w[ok, :3, :3], pelvis_cam[ok]) + c2w[ok, :3, 3]
        p0 = pelvis_cam[ok] - transl_std[ok]
        tr_w[ok] = pel_w - p0
    return go_w, tr_w


def handler(job):
    t0 = time.time()
    inp = job.get("input") or {}
    work = tempfile.mkdtemp(prefix="h3r_")
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        static_cam = bool(inp.get("static_cam", True))
        size = int(inp.get("size", 512))
        reset_interval = int(inp.get("reset_interval", 100))
        min_frames = int(inp.get("min_frames", 8))
        max_persons = int(inp.get("max_persons", 4))
        smoothing = str(inp.get("smoothing", "medium")).lower()
        colors = inp.get("person_colors") or PERSON_COLORS
        timing = {}

        raw, norm = os.path.join(work, "raw_video"), os.path.join(work, "norm.mp4")
        fetch_video(inp, raw)
        normalize_video(raw, norm)
        os.remove(raw)
        fdir = os.path.join(work, "frames")
        os.makedirs(fdir)
        import subprocess

        p = subprocess.run([os.environ.get("FFMPEG_BIN", "/usr/bin/ffmpeg"), "-loglevel", "error", "-i", norm,
                            "-q:v", "2", os.path.join(fdir, "f%05d.jpg")], capture_output=True, text=True)
        paths = sorted(glob.glob(os.path.join(fdir, "*.jpg")))
        if p.returncode != 0 or len(paths) < 8:
            raise ValueError(f"could not extract frames ({len(paths)}): {p.stderr[-200:]}")
        from PIL import Image

        W_full, H_full = Image.open(paths[0]).size
        timing["prep_s"] = round(time.time() - t0, 1)

        timing["model_load_s"] = _load(device)
        try:
            runpod.serverless.progress_update(job, "running Human3R")
        except Exception:
            pass
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        outputs, _ = H.run_inference(_state["model"], paths, device, size=size, reset_interval=reset_interval,
                                     use_ttt3r=True)
        if device == "cuda":
            torch.cuda.synchronize()
        timing["infer_s"] = round(time.time() - t1, 1)
        timing["gpu_peak_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 2**20) if device == "cuda" else None
        t2 = time.time()
        res = H.postprocess(outputs, device=device, min_frames=min_frames)
        del outputs
        T = res["n_frames"]
        nh, nw = res["net_hw"]
        sc = W_full / nw
        f_full = float(np.median(res["focal_net"]) * sc)
        K = np.array([[f_full, 0, W_full / 2], [0, f_full, H_full / 2], [0, 0, 1]], dtype=np.float64)
        c2w = res["c2w"].astype(np.float64)
        cam_stats = _c2w_stats(c2w)
        c2w_used = np.tile(np.eye(4), (T, 1, 1)) if static_cam else c2w

        from smplx_glb import person_glb

        ranked = sorted(res["persons"].items(), key=lambda kv: -int(kv[1]["observed"].sum()))[:max_persons]
        persons = []
        for rank, (pid, d) in enumerate(ranked):
            rv, obs = d["rotvec"], d["observed"]
            nb = d["betas"].shape[-1]
            go_w, tr_w = _to_world(rv, d["pelvis_cam"], d["transl_std"], c2w_used)
            betas_mean = np.nanmean(d["betas"], axis=0)
            arrs = {
                "K_fullimg": K, "observed": obs, "fps": np.float64(MODEL_FPS), "smpl_id": np.int64(pid),
                "c2w": c2w, "c2w_used": c2w_used, "focal_net_per_frame": res["focal_net"], "net_hw": np.array(res["net_hw"]),
                "incam_global_orient": rv[:, 0], "incam_body_pose": rv[:, 1:22].reshape(T, 63),
                "incam_lhand": rv[:, 22:37].reshape(T, 45), "incam_rhand": rv[:, 37:52].reshape(T, 45),
                "incam_jaw": rv[:, 52], "incam_betas": d["betas"], "incam_transl": d["transl_std"],
                "global_global_orient": go_w, "global_body_pose": rv[:, 1:22].reshape(T, 63), "global_betas": d["betas"],
                "global_transl": tr_w, "joints22_incam": d["joints22_cam"], "pelvis_incam": d["pelvis_cam"],
            }
            buf = io.BytesIO()
            np.savez_compressed(buf, **arrs)
            out = {"rank": rank, "smpl_id": int(pid), "observed_frames": int(obs.sum()), "color": colors[rank % len(colors)],
                   "mean_pelvis_depth_m": float(np.nanmean(d["pelvis_cam"][:, 2])),
                   "params_npz_b64": base64.b64encode(buf.getvalue()).decode()}
            try:
                glb = person_glb(_bm(nb), rv, betas_mean, d["pelvis_cam"], obs, MODEL_FPS, colors[rank % len(colors)],
                                 smoothing, to_out=CV2OUT)
                out["glb_b64"] = base64.b64encode(glb).decode()
            except Exception as e:  # noqa: BLE001
                out["glb_error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"[-1500:]
            persons.append(out)
        timing["post_s"] = round(time.time() - t2, 1)
        timing["handler_total_s"] = round(time.time() - t0, 1)
        return {"persons": persons, "K_fullimg": K.tolist(), "fps": MODEL_FPS, "n_frames": T, "width": W_full,
                "height": H_full, "static_cam": static_cam, "camera_pose_stats": cam_stats, "timing": timing,
                "n_persons_detected": len(res["persons"]), "elapsed_s": timing["handler_total_s"],
                "model": "Human3R human3r_896L (CC BY-NC-SA 4.0 + NAVER Multi-HMR NC: research only)"}
    except (ValueError, KeyError, TypeError) as e:
        return {"error": f"bad request: {e}\n{traceback.format_exc()}"[-3000:]}
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb, flush=True)
        return {"error": f"{e}\n{tb}"[-6000:]}
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
