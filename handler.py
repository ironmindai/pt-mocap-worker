"""RunPod serverless handler: monocular video -> 3D human motion (GVHMR) -> BVH.

Input:
  video_url     https URL of the video            (or)
  video_b64     base64 of the video file
  frame_stride  int >= 1, default 1. Output keeps every Nth frame of the 30 fps motion (fps = 30 / N).
                GVHMR itself always runs at 30 fps (the model is trained for it).
  static_cam    bool, default false. True skips visual odometry (tripod / fixed camera).
  f_mm          optional full-frame-equivalent focal length in mm (GVHMR --f_mm).
  bvh_scale     float, default 1.0. BVH units = meters * bvh_scale (use 100 for centimeters).
  gender        "neutral" (default) | "male" | "female": SMPL body model used for the GLB mesh. Falls back to
                neutral if that model file is unavailable; the model used is reported as "gender_used".
  smoothing     "medium" (default) | "light" | "strong" | "off": quaternion Savitzky-Golay smoothing of the
                pose / root translation before export (applies to both BVH and GLB).
  return_smpl   bool, default false. Also return the raw SMPL-X params as base64 npz ("smpl_npz_b64").
  hands         false|null (default, output identical to before) | true or "estimate" | "fist". "estimate" = HaMeR
                finger-pose estimation; "fist" = no HaMeR at all (for gloved hands): a constant closed boxer's fist on both
                hands for every frame, same SMPL-X GLB structure. Anything else is a bad request. With estimation the GLB mesh
                becomes SMPL-X (22 body + 30 animated finger joints, same vertex-colour painting, same ground alignment);
                works for glb_b64, per-person glb_b64 and scene_glb_b64. BVH is unchanged. If hand estimation fails the
                normal (SMPL) output is returned with "hands_error".
  return_hands  bool, default false. With hands: also return "hands_npz_b64" (per-person finger poses, see README).

Multi-person mode (lab branch; single-person behaviour above is unchanged when none of these is given):
  list_tracks   bool. Run only the YOLO+ByteTrack pass and return {"tracks": [...], "n_frames", "width", "height"}
                (every track: id, rank by bbox-area score, frame range, area_score, sampled bboxes).
  track_ids     list; one entry per person, each an int or a list of ints (ids merged into one person).
  person_rank   list of ints indexing the area-sorted tracks (alternative to track_ids).
  bbx_override  list (per person) of per-frame [x1,y1,x2,y2] at 30 fps (all frames), bypasses tracking.
  -> {"persons": [{rank, track_ids, observed_frames, bbox_xyxy, params_npz_b64, glb_b64, ...}], "tracks", "K_fullimg",
      "fps": 30, ...}. params_npz keys: incam_{global_orient,body_pose,betas,transl}, global_{...}, K_fullimg,
      joints24_incam (T,24,3 SMPL joints in the camera frame, via GVHMR's own SMPL-X->SMPL path), bbx_xys, observed.
  person_colors optional list of "#rrggbb" shorts colours for the per-person (ground-aligned) GLBs.

Output:
  {"bvh_b64", "glb_b64", "gender_used", "smoothing", "fps", "n_frames", "duration_s", "elapsed_s", ...}   or   {"error": "..."}
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
SMPL_DIR = os.path.join(GVHMR_DIR, "inputs/checkpoints/body_models/smpl")
GENDERS = ("neutral", "male", "female")
SMOOTHING = ("off", "light", "medium", "strong")
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


def H_parse(value):
    """Validate the `hands` input: None | "estimate" | "fist" (see hands.parse_hands_mode). ValueError -> bad request."""
    from hands import parse_hands_mode

    return parse_hands_mode(value)


def _smplx_model():
    """Lazy numpy SMPL-X (no HaMeR); used by both hand modes."""
    if "smplx" not in _state:
        import smplx_np as sx

        _state["smplx"] = sx.load_smplx_npz(SMPLX_NPZ)
    return _state["smplx"]


def _hands_models():
    """Lazy: HaMeR + MANO prior + numpy SMPL-X (only touched when hands is true / "estimate")."""
    if "hands" in _state:
        return _state["hands"]
    import hamer_infer
    import hands as H

    est = hamer_infer.HandEstimator(device="cuda")
    prior = H.HandPrior.from_mano_pkl(os.path.join(hamer_infer.HAMER_DATA, "_DATA/data/mano/MANO_RIGHT.pkl"))
    _state["hands"] = (est, prior, _smplx_model())
    return _state["hands"]


def compute_fist(people):
    """hands="fist": constant closed fist on both hands, every frame, no HaMeR. people: list of dicts with incam_global_orient
    (T,3). Same result shape as compute_hands (left_aa/right_aa (T,15,3) SMPL-X convention, weight (T,2) = 1)."""
    import hands as H

    res = []
    for d in people:
        T = len(d["incam_global_orient"])
        left, right = H.fist_pose(T)
        res.append({"left_aa": left, "right_aa": right, "weight": np.ones((T, 2))})
    return res, {"mode": "fist"}


def _video_frames_rgb(path):
    import cv2

    cap = cv2.VideoCapture(path)
    try:
        while True:
            ok, fr = cap.read()
            if not ok:
                return
            yield fr[:, :, ::-1]
    finally:
        cap.release()


def compute_hands(norm_video, people, K, width, height, job=None):
    """people: list of dicts with incam_global_orient (T,3), incam_body_pose (T,63), incam_betas, incam_transl, observed (T,).
    Returns (list of dicts left_aa/right_aa (T,15,3) SMPL-X convention, weights (T,2), raw ..., info)."""
    import hands as H
    import hands_pipeline as hp
    import smplx_np as sx

    est, prior, mdl = _hands_models()
    persons = []
    for d in people:
        betas = np.asarray(d["incam_betas"], dtype=np.float64).reshape(-1, 10).mean(0)
        rots = sx.rots_from_params(d["incam_global_orient"], d["incam_body_pose"])
        _, J = sx.rest_shape(mdl, betas)
        root = J[0][None] + np.asarray(d["incam_transl"], dtype=np.float64)
        _, j52 = sx.posed(mdl, betas, rots, root, vert_idx=np.arange(1))
        persons.append({"joints52": j52, "rots": rots, "observed": d.get("observed")})
    ests, info = hp.estimate_hands(_video_frames_rgb(norm_video), persons, est, prior, np.asarray(K, dtype=np.float64),
                                   width, height)
    neutral_r = mdl["hands_mean_r"]
    res = []
    for e in ests:
        left, right, w = hp.fuse_person(e, neutral_r, prior)
        res.append({"left_aa": left, "right_aa": right, "weight": w, "raw_aa": e["raw_aa"], "conf": e["conf"],
                    "agree_deg": e["agree_deg"], "sharp": e["sharp"], "kp_img": e["kp_img"], "vis": e["vis"]})
    info["mean_weight"] = [[round(float(x), 3) for x in r["weight"].mean(0)] for r in res]
    return res, info


def hands_npz(res, fps, stride=1):
    """base64 npz: per person p{i}_{left,right}_hand_pose (T,15,3) SMPL-X axis-angle (fused), p{i}_weight (T,2) final blend
    weight [L,R] (1 = estimate, 0 = relaxed neutral), p{i}_conf_raw (T,2), p{i}_raw_{left,right} (T,15,3) unfused, p{i}_kp_img
    (T,2,21,2) HaMeR keypoints in image px, p{i}_agree_deg (T,2); neutral_hand_{left,right}; fps."""
    import hands as H
    import smplx_np as sx

    mdl = _smplx_model()
    arr = {"fps": np.float64(fps), "neutral_hand_left": H.to_smplx_pose(mdl["hands_mean_r"], True),
           "neutral_hand_right": mdl["hands_mean_r"], "stride": np.int64(stride)}
    for i, r in enumerate(res):
        arr[f"p{i}_left_hand_pose"] = r["left_aa"]
        arr[f"p{i}_right_hand_pose"] = r["right_aa"]
        arr[f"p{i}_weight"] = r["weight"]
        if "raw_aa" not in r:  # fist mode: nothing was estimated
            continue
        arr[f"p{i}_conf_raw"] = r["conf"]
        arr[f"p{i}_raw_left"] = H.to_smplx_pose(r["raw_aa"][:, 0], True)
        arr[f"p{i}_raw_right"] = r["raw_aa"][:, 1]
        arr[f"p{i}_kp_img"] = r["kp_img"]
        arr[f"p{i}_agree_deg"] = r["agree_deg"]
    buf = io.BytesIO()
    np.savez_compressed(buf, **arr)
    return base64.b64encode(buf.getvalue()).decode()


def get_smpl_model(gender):
    """Return (model, gender_used); falls back to neutral if the requested gender file is missing/unreadable."""
    from smpl_glb import load_smpl_pkl

    for g in ([gender, "neutral"] if gender != "neutral" else ["neutral"]):
        key = ("smpl", g)
        if key in _state:
            return _state[key], g
        path = os.path.join(SMPL_DIR, f"SMPL_{g.upper()}.pkl")
        try:
            _state[key] = load_smpl_pkl(path)
            return _state[key], g
        except Exception as e:  # noqa: BLE001
            print(f"[smpl] could not load {path}: {type(e).__name__}: {e}", flush=True)
    raise RuntimeError("no SMPL body model could be loaded")


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


def run_gvhmr(video_path, workdir, name, static_cam, f_mm, with_incam=False):
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
    glob = {k: v.numpy() for k, v in pred["smpl_params_global"].items()}
    if with_incam:
        return glob, {k: v.numpy() for k, v in pred["smpl_params_incam"].items()}, pred["K_fullimg"][0].numpy()
    return glob


PERSON_COLORS = ["#C0201A", "#1F4FC4", "#1E8E3E", "#C98A00"]  # red, blue, green, amber


def _cpu_np(d):
    return {k: v.detach().cpu().numpy() for k, v in d.items()}


def _incam_joints24(pred, st):
    """SMPL(24) joints in the camera frame for one person, GVHMR's own recipe (render_incam): SMPL-X ->
    smplx2smpl vertices -> SMPL J-regressor. Returns (T,24,3) float32 or None on failure."""
    try:
        import torch
        from hmr4d.utils.net_utils import to_cuda
        from hmr4d.utils.smplx_utils import make_smplx

        if "smplx" not in st:
            st["smplx"] = make_smplx("supermotion").cuda()
            st["smplx2smpl"] = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
            st["J_reg"] = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").cuda()
        with torch.no_grad():
            o = st["smplx"](**to_cuda(pred["smpl_params_incam"]))
            verts = torch.stack([torch.matmul(st["smplx2smpl"], v) for v in o.vertices])
            j = torch.einsum("jv,tvi->tji", st["J_reg"], verts)
        return j.cpu().numpy().astype(np.float32)
    except Exception as e:  # noqa: BLE001
        print(f"[joints24_incam] failed: {type(e).__name__}: {e}", flush=True)
        return None


def handle_multi(inp, job, work, norm, name, t0):
    """list_tracks / per-person GVHMR. `norm` is the normalized 30 fps video."""
    import multiperson as mpx

    os.chdir(GVHMR_DIR)
    static_cam = bool(inp.get("static_cam", False))
    f_mm = int(inp["f_mm"]) if inp.get("f_mm") is not None else None
    smoothing = str(inp.get("smoothing", "medium")).lower()
    if smoothing not in SMOOTHING:
        raise ValueError(f"smoothing must be one of {SMOOTHING}")
    gender = str(inp.get("gender", "male")).lower()
    if gender not in GENDERS:
        raise ValueError(f"gender must be one of {GENDERS}")
    colors = inp.get("person_colors") or PERSON_COLORS
    hands_mode = H_parse(inp.get("hands"))
    want_hands = hands_mode is not None
    from hmr4d.utils.video_io_utils import get_video_lwh

    n_frames, width, height = get_video_lwh(norm)
    if n_frames < 8:
        raise ValueError(f"video too short ({n_frames} frames at {MODEL_FPS} fps)")

    t_track = time.time()
    history, summaries = None, []
    if inp.get("bbx_override") is None:
        import torch
        from hmr4d.utils.preproc import Tracker

        with torch.no_grad():
            tracker = Tracker()
            history = tracker.track(norm)
            del tracker
        torch.cuda.empty_cache()
        summaries = mpx.summarize_tracks(history, width, height)
    track_s = round(time.time() - t_track, 1)
    out = {"tracks": summaries, "n_frames": n_frames, "width": width, "height": height, "fps": MODEL_FPS,
           "tracking_s": track_s}
    if inp.get("list_tracks") and inp.get("track_ids") is None and inp.get("person_rank") is None \
            and inp.get("bbx_override") is None:
        out["suggested"] = [s["id"] for s in summaries[:2]]
        out["sample_every"] = 3
        out["elapsed_s"] = round(time.time() - t0, 1)
        return out

    scene = bool(inp.get("scene"))
    if scene and inp.get("track_ids") is None and inp.get("person_rank") is None and inp.get("bbx_override") is None:
        if len(summaries) < 2:
            raise ValueError(f"scene needs two people but only {len(summaries)} track(s) were found; pass track_ids / bbx_override")
        inp = dict(inp, person_rank=[0, 1])  # auto: the two largest tracks

    # ---- one bbox sequence per requested person ----
    if inp.get("bbx_override") is not None:
        persons = []
        for i, seq in enumerate(inp["bbx_override"]):
            seq = np.asarray(seq, dtype=np.float64)
            if seq.shape != (n_frames, 4):
                raise ValueError(f"bbx_override[{i}] must have shape ({n_frames}, 4), got {seq.shape}")
            persons.append(([-1 - i], seq, np.ones(n_frames, dtype=bool)))
    else:
        groups = mpx.select_tracks(summaries, inp.get("track_ids"), inp.get("person_rank"))
        persons = [(g, *mpx.build_person_bbx(history, g, n_frames)) for g in groups]

    st = _load()
    demo, torch, model = st["demo"], st["torch"], st["model"]
    from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy

    from motion import smooth_motion
    from smpl_glb import smpl_to_glb

    results, slam_src, scene_people, hand_people, glob_params = [], None, [], [], []
    for i, (ids, bbx, observed) in enumerate(persons):
        t_p = time.time()
        try:
            runpod.serverless.progress_update(job, f"person {i + 1}/{len(persons)}")
        except Exception:
            pass
        cfg = _compose(st["compose"], st["init"], f"{name}_p{i}", work, static_cam, f_mm)
        os.makedirs(cfg.preprocess_dir, exist_ok=True)
        shutil.copy(norm, cfg.video_path)
        bbx_t = torch.from_numpy(bbx).float()
        torch.save({"bbx_xyxy": bbx_t, "bbx_xys": get_bbx_xys_from_xyxy(bbx_t, base_enlarge=1.2).float()},
                   cfg.paths.bbx)  # run_preprocess then skips tracking and reuses our bbox
        if slam_src and os.path.exists(slam_src):
            shutil.copy(slam_src, cfg.paths.slam)
        with torch.no_grad():
            demo.run_preprocess(cfg)
            data = demo.load_data_dict(cfg)
            pred = model.predict(data, static_cam=cfg.static_cam)
        pred = demo.detach_to_cpu(pred)
        if not static_cam and os.path.exists(cfg.paths.slam):
            slam_src = cfg.paths.slam
        inc, glo = _cpu_np(pred["smpl_params_incam"]), _cpu_np(pred["smpl_params_global"])
        K = pred["K_fullimg"][0].numpy()
        j24 = _incam_joints24(pred, st)
        arrs = {"K_fullimg": K, "bbx_xyxy": bbx, "bbx_xys": torch.load(cfg.paths.bbx)["bbx_xys"].numpy(),
                "observed": observed, "fps": np.float64(MODEL_FPS)}
        arrs.update({f"incam_{k}": v for k, v in inc.items()})
        arrs.update({f"global_{k}": v for k, v in glo.items()})
        if j24 is not None:
            arrs["joints24_incam"] = j24
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrs)
        if want_hands:
            hand_people.append({**{f"incam_{k}": v for k, v in inc.items()}, "observed": observed})
            glob_params.append(glo)
        if scene:
            if j24 is None:
                raise RuntimeError("scene: joints24_incam could not be computed")
            scene_people.append({**{f"incam_{k}": v for k, v in inc.items()}, **{f"global_{k}": v for k, v in glo.items()},
                                 "joints24_incam": j24})
        res = {"rank": i, "track_ids": [int(x) for x in ids], "observed_frames": int(observed.sum()),
               "bbox_xyxy": np.round(bbx, 1).tolist(), "color": colors[i % len(colors)],
               "params_npz_b64": base64.b64encode(buf.getvalue()).decode()}
        try:  # ground-aligned stand-alone preview GLB
            if (scene and not inp.get("return_persons")) or want_hands:  # hands: GLB is built after hand estimation
                raise StopIteration
            go, bp, tr = smooth_motion(glo["global_orient"], glo["body_pose"], glo["transl"], smoothing)
            model_smpl, used = get_smpl_model(gender)
            res["glb_b64"] = base64.b64encode(smpl_to_glb(
                model_smpl, go, bp, tr, glo["betas"], MODEL_FPS, used, shorts_color=colors[i % len(colors)],
                top=False)).decode()
            res["gender_used"] = used
        except StopIteration:
            pass
        except Exception as e:  # noqa: BLE001
            res["glb_error"] = f"{type(e).__name__}: {e}"[-1500:]
        res["person_s"] = round(time.time() - t_p, 1)
        results.append(res)
        torch.cuda.empty_cache()
    out["K_fullimg"] = K.tolist()
    hands_res = None
    if want_hands:
        try:
            try:
                runpod.serverless.progress_update(job, "estimating hands")
            except Exception:
                pass
            if hands_mode == "fist":
                hands_res, hinfo = compute_fist(glob_params)
            else:
                hands_res, hinfo = compute_hands(norm, hand_people, K, width, height, job)
            out["hands"] = hinfo
            if inp.get("return_hands"):
                out["hands_npz_b64"] = hands_npz(hands_res, MODEL_FPS)
            if not scene or inp.get("return_persons"):
                import smplx_np as sx

                mdl = _smplx_model()
                for i, (r, gl, hr) in enumerate(zip(results, glob_params, hands_res)):
                    go, bp, tr = smooth_motion(gl["global_orient"], gl["body_pose"], gl["transl"], smoothing)
                    betas = np.asarray(gl["betas"], dtype=np.float64).reshape(-1, 10).mean(0)
                    r["glb_b64"] = base64.b64encode(sx.person_glb_ground(
                        mdl, betas, go, bp, tr, hr["left_aa"], hr["right_aa"], MODEL_FPS, color=colors[i % len(colors)],
                        top=False, gender="male")).decode()
                    r["gender_used"] = "neutral (SMPL-X)"
        except Exception as e:  # noqa: BLE001  (hand failure must not lose the body result)
            hands_res = None
            out["hands_error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"[-3000:]
            print(out["hands_error"], flush=True)
            for i, (r, gl) in enumerate(zip(results, glob_params)):  # fall back to the plain SMPL preview GLB
                if (not scene or inp.get("return_persons")) and "glb_b64" not in r:
                    try:
                        go, bp, tr = smooth_motion(gl["global_orient"], gl["body_pose"], gl["transl"], smoothing)
                        model_smpl, used = get_smpl_model(gender)
                        r["glb_b64"] = base64.b64encode(smpl_to_glb(
                            model_smpl, go, bp, tr, gl["betas"], MODEL_FPS, used, shorts_color=colors[i % len(colors)],
                            top=False)).decode()
                        r["gender_used"] = used
                    except Exception as e2:  # noqa: BLE001
                        r["glb_error"] = f"{type(e2).__name__}: {e2}"[-1500:]
    if scene:
        import scene as scn

        if len(persons) < 2:
            raise ValueError("scene needs two persons")
        t_s = time.time()
        model_smpl, used = get_smpl_model(str(inp.get("scene_gender", "neutral")).lower())
        glb, meta = scn.build_scene(
            model_smpl, scene_people, K, MODEL_FPS, smoothing, colors=inp.get("person_colors") or scn.DEFAULT_COLORS,
            track_ids=[p[0] for p in persons], observed=[p[2] for p in persons], bboxes=[p[1] for p in persons],
            hands=([(h["left_aa"], h["right_aa"]) for h in hands_res] if hands_res is not None else None),
            smplx_model=(_smplx_model() if hands_res is not None else None))
        meta.update(width=width, height=height, gender_used=used, static_cam=static_cam, scene_s=round(time.time() - t_s, 1))
        out["scene_glb_b64"] = base64.b64encode(glb).decode()
        out["scene_meta"] = meta
        if not inp.get("return_persons"):  # per-person payloads (params npz + stand-alone GLB) only on request
            for r in results:
                for k in ("params_npz_b64", "glb_b64", "bbox_xyxy"):
                    r.pop(k, None)
    out["persons"] = results
    out["static_cam"] = static_cam
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


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
        gender = str(inp.get("gender", "neutral")).lower()
        if gender not in GENDERS:
            raise ValueError(f"gender must be one of {GENDERS}")
        smoothing = str(inp.get("smoothing", "medium")).lower()
        if smoothing not in SMOOTHING:
            raise ValueError(f"smoothing must be one of {SMOOTHING}")
        H_parse(inp.get("hands"))  # fail fast on an invalid value, before any video work
        name = "job_" + (str(job.get("id") or uuid.uuid4()).replace("-", "")[:16])

        raw, norm = os.path.join(work, "raw_video"), os.path.join(work, "norm.mp4")
        fetch_video(inp, raw)
        normalize_video(raw, norm)
        os.remove(raw)
        if (inp.get("list_tracks") or inp.get("scene") or inp.get("track_ids") is not None or inp.get("person_rank") is not None
                or inp.get("bbx_override") is not None):
            return handle_multi(inp, job, work, norm, name, t0)
        try:
            runpod.serverless.progress_update(job, "running GVHMR")
        except Exception:
            pass
        hands_mode = H_parse(inp.get("hands"))
        want_hands = hands_mode is not None
        est_hands = hands_mode == "estimate"
        norm_copy = None
        if est_hands:  # run_gvhmr moves the video away; keep a copy for the hand pass
            norm_copy = os.path.join(work, "norm_hands.mp4")
            shutil.copy(norm, norm_copy)
        try:
            if est_hands:
                params, inc_params, K_img = run_gvhmr(norm, work, name, static_cam, f_mm, with_incam=True)
            else:
                params = run_gvhmr(norm, work, name, static_cam, f_mm)
        except ValueError:
            raise
        except Exception as e:
            # Typical: no person detected / tracker returned nothing.
            raise RuntimeError(f"GVHMR inference failed: {type(e).__name__}: {e}\n{traceback.format_exc()}")

        from motion import smooth_motion
        from smpl_bvh import smpl_to_bvh

        # Smooth at the native 30 fps, then subsample.
        go, bp, tr = smooth_motion(params["global_orient"], params["body_pose"], params["transl"], smoothing)
        sel = slice(None, None, stride)
        fps = MODEL_FPS / stride
        go, bp, tr, betas = go[sel], bp[sel], tr[sel], params["betas"][sel]
        bvh_text = smpl_to_bvh(go, bp, tr, betas, SMPLX_NPZ, fps, scale)
        n = int(go.shape[0])
        out = {
            "bvh_b64": base64.b64encode(bvh_text.encode()).decode(),
            "fps": fps,
            "n_frames": n,
            "duration_s": round(n / fps, 3),
            "skeleton": "SMPL-22 (Y-up, +Z forward, meters*bvh_scale)",
            "smoothing": smoothing,
        }
        hands_done = False
        if want_hands:
            try:
                import smplx_np as sx
                from hmr4d.utils.video_io_utils import get_video_lwh

                try:
                    runpod.serverless.progress_update(job, "estimating hands")
                except Exception:
                    pass
                if hands_mode == "fist":
                    hres, hinfo = compute_fist([{"incam_global_orient": params["global_orient"]}])
                else:
                    _, width, height = get_video_lwh(norm_copy)
                    hp_in = [{**{f"incam_{k}": v for k, v in inc_params.items()}, "observed": np.ones(len(go), dtype=bool)}]
                    hres, hinfo = compute_hands(norm_copy, hp_in, K_img, width, height, job)
                out["hands"] = hinfo
                hr = hres[0]
                if inp.get("return_hands"):
                    out["hands_npz_b64"] = hands_npz(hres, MODEL_FPS, stride)
                mdl = _smplx_model()
                out["glb_b64"] = base64.b64encode(sx.person_glb_ground(
                    mdl, np.asarray(betas, dtype=np.float64).reshape(-1, 10).mean(0), go, bp, tr, hr["left_aa"][sel], hr["right_aa"][sel], fps, top=None, gender=gender)).decode()
                out["gender_used"] = "neutral (SMPL-X body model; gender affects clothing only)"
                hands_done = True
            except Exception as e:  # noqa: BLE001  (hand failure falls back to the plain SMPL GLB)
                out["hands_error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"[-3000:]
                print(out["hands_error"], flush=True)
        try:
            from smpl_glb import smpl_to_glb

            if not hands_done:
                model, used = get_smpl_model(gender)
                out["glb_b64"] = base64.b64encode(smpl_to_glb(model, go, bp, tr, betas, fps, used)).decode()
                out["gender_used"] = used
        except Exception as e:  # noqa: BLE001  (GLB failure must not lose the BVH)
            out["glb_error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"[-3000:]
        out["elapsed_s"] = round(time.time() - t0, 1)
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
