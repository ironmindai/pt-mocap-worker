"""RunPod serverless handler: monocular video -> 3D human motion (GVHMR) -> BVH.

Input:
  video_url     https URL of the video            (or)
  video_b64     base64 of the video file
  frame_stride  int >= 1, default 1. Output keeps every Nth frame of the 30 fps motion (fps = 30 / N).
                GVHMR itself always runs at 30 fps (the model is trained for it).
  static_cam    bool, default false. True skips visual odometry (tripod / fixed camera).
  f_mm          optional full-frame-equivalent focal length in mm (GVHMR --f_mm).
  bvh_scale     float, default 1.0. BVH units = meters * bvh_scale (use 100 for centimeters).
  return_smpl   bool, default false. Also return the raw SMPL-X params as base64 npz ("smpl_npz_b64").

Output:
  {"bvh_b64", "fps", "n_frames", "duration_s", "elapsed_s", ...}   or   {"error": "..."}
"""
import base64
import importlib.util
import io
import os
import shutil
import subprocess
import tempfile
import traceback
import time
import urllib.parse
import urllib.request
import uuid

import numpy as np
import runpod

GVHMR_DIR = os.environ.get("GVHMR_DIR", "/workspace/GVHMR")
SMPLX_NPZ = os.path.join(GVHMR_DIR, "inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz")
MAX_VIDEO_BYTES = int(os.environ.get("MAX_VIDEO_BYTES", 500_000_000))
MAX_SECONDS = float(os.environ.get("MAX_SECONDS", 120))  # longer videos are truncated
MODEL_FPS = 30
# Debian build with libx264; conda's minimal ffmpeg in the base image may shadow the one on PATH.
FFMPEG = os.environ.get("FFMPEG_BIN", "/usr/bin/ffmpeg")

_state = {}


def _load():
    """Lazy one-time init: cwd, GVHMR demo module, model weights on GPU."""
    if _state:
        return _state
    import hydra
    import torch
    from hydra import compose, initialize_config_module
    from hmr4d.configs import register_store_gvhmr

    os.chdir(GVHMR_DIR)  # GVHMR uses cwd-relative "inputs/checkpoints/..." paths
    spec = importlib.util.spec_from_file_location("gvhmr_demo", os.path.join(GVHMR_DIR, "tools/demo/demo.py"))
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    register_store_gvhmr()
    _patch_simple_vo()

    cfg = _compose(compose, initialize_config_module, "warmup", "/tmp/gvhmr_warmup", False, None)
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.load_pretrained_model(cfg.ckpt_path)
    model = model.eval().cuda()
    _state.update(demo=demo, model=model, compose=compose, init=initialize_config_module, torch=torch)
    return _state


def _patch_simple_vo():
    """Make SimpleVO tolerate pairs where pycolmap cannot recover a relative pose.

    pycolmap returns cam2_from_cam1=None when two-view geometry fails (too few matches/inliers, e.g. a
    static or low-texture scene), and GVHMR then crashes with "'NoneType' object has no attribute 'matrix'".
    Treat such a pair as "no camera motion" (identity), which is the right default for a mostly fixed camera.
    """
    from hmr4d.utils.preproc.relpose import solver_two_view as stv

    cls = stv.PycolmapRansacTwoViewGeometrySolver
    orig = cls.solve

    def safe_solve(self, pts0, pts1):
        try:
            return orig(self, pts0, pts1)
        except Exception as e:  # noqa: BLE001
            print(f"[SimpleVO] pair failed ({type(e).__name__}: {e}); assuming identity", flush=True)
            return np.eye(4)

    cls.solve = safe_solve


def _compose(compose, init, name, root, static_cam, f_mm):
    with init(version_base="1.3", config_module="hmr4d.configs"):
        ov = [f"video_name={name}", f"output_root={root}", f"static_cam={static_cam}",
              "verbose=False", "use_dpvo=False"]
        if f_mm is not None:
            ov.append(f"f_mm={f_mm}")
        return compose(config_name="demo", overrides=ov)


def fetch_video(inp, dest):
    if inp.get("video_url"):
        url = str(inp["video_url"])
        if urllib.parse.urlparse(url).scheme != "https":
            raise ValueError("video_url must be https")
        req = urllib.request.Request(url, headers={"User-Agent": "pt-mocap-worker"})
        size = 0
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_VIDEO_BYTES:
                    raise ValueError(f"video too large (> {MAX_VIDEO_BYTES} bytes)")
                f.write(chunk)
    elif inp.get("video_b64"):
        data = base64.b64decode(inp["video_b64"], validate=False)
        if len(data) > MAX_VIDEO_BYTES:
            raise ValueError(f"video too large (> {MAX_VIDEO_BYTES} bytes)")
        with open(dest, "wb") as f:
            f.write(data)
    else:
        raise ValueError("video_url or video_b64 is required")
    if os.path.getsize(dest) == 0:
        raise ValueError("empty video")


def normalize_video(src, dst):
    """Re-encode to constant 30 fps, even dimensions, no audio, capped length (GVHMR assumes 30 fps)."""
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", src, "-t", str(MAX_SECONDS), "-an",
           "-vf", f"fps={MODEL_FPS},scale=trunc(iw/2)*2:trunc(ih/2)*2",
           "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", dst]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not os.path.exists(dst):
        raise ValueError("could not decode video: " + p.stderr[-300:])


def run_gvhmr(video_path, workdir, name, static_cam, f_mm):
    st = _load()
    demo, torch = st["demo"], st["torch"]
    cfg = _compose(st["compose"], st["init"], name, workdir, static_cam, f_mm)
    os.makedirs(cfg.preprocess_dir, exist_ok=True)
    shutil.move(video_path, cfg.video_path)
    length, _, _ = demo.get_video_lwh(cfg.video_path)
    if length < 8:
        raise ValueError(f"video too short ({length} frames at {MODEL_FPS} fps)")
    with torch.no_grad():
        demo.run_preprocess(cfg)
        data = demo.load_data_dict(cfg)
        pred = st["model"].predict(data, static_cam=cfg.static_cam)
    pred = demo.detach_to_cpu(pred)
    return {k: v.numpy() for k, v in pred["smpl_params_global"].items()}


def handler(job):
    t0 = time.time()
    inp = job.get("input") or {}
    work = tempfile.mkdtemp(prefix="mocap_")
    try:
        stride = int(inp.get("frame_stride", 1))
        if stride < 1:
            raise ValueError("frame_stride must be >= 1")
        static_cam = bool(inp.get("static_cam", False))
        f_mm = int(inp["f_mm"]) if inp.get("f_mm") is not None else None
        scale = float(inp.get("bvh_scale", 1.0))
        name = "job_" + (str(job.get("id") or uuid.uuid4()).replace("-", "")[:16])

        raw, norm = os.path.join(work, "raw_video"), os.path.join(work, "norm.mp4")
        fetch_video(inp, raw)
        normalize_video(raw, norm)
        os.remove(raw)
        try:
            runpod.serverless.progress_update(job, "running GVHMR")
        except Exception:
            pass
        try:
            params = run_gvhmr(norm, work, name, static_cam, f_mm)
        except ValueError:
            raise
        except Exception as e:
            # Typical: no person detected / tracker returned nothing.
            raise RuntimeError(f"GVHMR inference failed: {type(e).__name__}: {e}\n{traceback.format_exc()}")

        from smpl_bvh import smpl_to_bvh

        sel = slice(None, None, stride)
        fps = MODEL_FPS / stride
        go, bp, tr = params["global_orient"][sel], params["body_pose"][sel], params["transl"][sel]
        bvh_text = smpl_to_bvh(go, bp, tr, params["betas"], SMPLX_NPZ, fps, scale)
        n = int(go.shape[0])
        out = {
            "bvh_b64": base64.b64encode(bvh_text.encode()).decode(),
            "fps": fps,
            "n_frames": n,
            "duration_s": round(n / fps, 3),
            "skeleton": "SMPL-22 (Y-up, +Z forward, meters*bvh_scale)",
            "elapsed_s": round(time.time() - t0, 1),
        }
        if inp.get("return_smpl"):
            buf = io.BytesIO()
            np.savez_compressed(buf, global_orient=go, body_pose=bp, transl=tr, betas=params["betas"][sel], fps=fps)
            out["smpl_npz_b64"] = base64.b64encode(buf.getvalue()).decode()
        return out
    except (ValueError, KeyError, TypeError) as e:
        return {"error": f"bad request: {e}"}
    except Exception as e:
        tb = traceback.format_exc()
        print(tb, flush=True)
        return {"error": f"{e}\n{tb}"[-6000:]}
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
