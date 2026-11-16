"""Human3R (CUT3R + Multi-HMR) multi-person world reconstruction: inference + post-processing.

Pure inference/numpy glue shared by handler_b.py and local CPU smoke tests. The Human3R repo checkout must be
importable (`H3R_DIR`, contains demo.py, add_ckpt_path.py, src/). Everything runs from that cwd because Human3R uses
cwd-relative "src/models/..." paths for the SMPL-X files.

Output per person (`smpl_id` tracked by Human3R across frames):
  rotvec (T,53,3)  SMPL-X axis-angle [global, body21, lhand15, rhand15, jaw]   NaN where unobserved
  betas  (T,nb)    per-frame shape (Human3R predicts it per frame; we also give the mean)
  pelvis_cam (T,3) pelvis position in that frame's camera frame (OpenCV: x right, y down, z forward), metres
  joints22_cam (T,22,3)  SMPL-X body joints in the camera frame
  transl_std (T,3) standard smplx `transl` so that smplx(global_orient, body_pose, betas, transl=transl_std) == mesh
  observed (T,)    bool
plus per-frame c2w camera poses (T,4,4) (world = first camera frame, via Human3R's own pose head) and the focal.
"""
import os
import sys
import time

import numpy as np

H3R_DIR = os.environ.get("H3R_DIR", "/workspace/Human3R")


def setup_path(h3r_dir=None):
    d = h3r_dir or H3R_DIR
    os.chdir(d)
    for p in (d, os.path.join(d, "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return d


def load_model(ckpt, device):
    setup_path()
    from src.dust3r.model import ARCroco3DStereo

    model = ARCroco3DStereo.from_pretrained(ckpt).to(device)
    model.eval()
    return model


def run_inference(model, img_paths, device, size=512, reset_interval=100, use_ttt3r=True):
    """Returns raw Human3R outputs (dict with 'views' and 'pred')."""
    import demo  # Human3R repo's demo.py: importing it does not start anything (viser is imported lazily there)
    from src.dust3r.inference import inference_recurrent_lighter

    img_res = getattr(model, "mhmr_img_res", None)
    views = demo.prepare_input(img_paths=img_paths, img_mask=[True] * len(img_paths), size=size, revisit=1,
                               update=True, img_res=img_res, reset_interval=reset_interval)
    outputs, _ = inference_recurrent_lighter(views, model, device, use_ttt3r=use_ttt3r)
    return outputs, img_res


def postprocess(outputs, device="cpu", min_frames=8):
    device = "cpu"  # tiny workload; CPU keeps it independent of GPU state/memory
    """Raw outputs -> numpy dict (see module doc). Mirrors demo.prepare_output without the viewer/rendering."""
    import roma
    import torch
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils import SMPL_Layer
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from src.dust3r.utils.geometry import matrix_cumprod

    pred, views = outputs["pred"], outputs["views"]
    # drop the duplicated overlap views inserted after each reset (same bookkeeping as demo.prepare_output)
    reset_mask = torch.cat([v["reset"] for v in views], 0)
    shifted = torch.cat([torch.tensor(False).unsqueeze(0), reset_mask[:-1]], 0)
    pred = [p for p, m in zip(pred, shifted) if not m]
    reset_mask = reset_mask[~shifted]

    poses = [pose_encoding_to_camera(p["camera_pose"].clone()).cpu() for p in pred]
    if reset_mask.any():
        pr = torch.cat(poses, 0)
        ident = torch.eye(4, device=pr.device)
        base = matrix_cumprod(torch.where(reset_mask.unsqueeze(-1).unsqueeze(-1), pr, ident))
        shifted_base = torch.cat([ident.unsqueeze(0), base[:-1]], 0)
        pr = torch.einsum("bij,bjk->bik", shifted_base, pr)
        poses = list(pr.unsqueeze(1).unbind(0))
    c2w = torch.cat(poses, 0).numpy()  # (T,4,4)

    pts_self = torch.cat([p["pts3d_in_self_view"].detach().cpu() for p in pred], 0)  # (T,H,W,3)
    T, H, W, _ = pts_self.shape
    pp = torch.tensor([W // 2, H // 2], device=pts_self.device).float().repeat(T, 1)
    focal = estimate_focal_knowing_depth(pts_self, pp, focal_mode="weiszfeld").detach().cpu().numpy()  # (T,)
    net_hw = (int(H), int(W))

    _c = lambda x: x.detach().cpu() if torch.is_tensor(x) else x  # noqa: E731  (outputs may live on the GPU)
    shape_l = [_c(p.get("smpl_shape", torch.empty(1, 0, 10))[0]) for p in pred]
    rot_l = [roma.rotmat_to_rotvec(_c(p.get("smpl_rotmat", torch.empty(1, 0, 53, 3, 3))[0])) for p in pred]
    transl_l = [_c(p.get("smpl_transl", torch.empty(1, 0, 3))[0]) for p in pred]
    expr_l = [_c(p.get("smpl_expression", [None])[0]) for p in pred]
    id_l = [_c(p.get("smpl_id", torch.empty(1, 0))[0]) for p in pred]
    nb = next((s.shape[-1] for s in shape_l if s.shape[0] > 0), 10)

    layer = SMPL_Layer(type="smplx", gender="neutral", num_betas=nb, kid=False, person_center="head").to(device)
    bm = layer.bm_x
    tracks = {}
    for f in range(T):
        n = shape_l[f].shape[0]
        if n == 0:
            continue
        K = torch.eye(3).unsqueeze(0).repeat(n, 1, 1)
        K[:, 0, 0] = K[:, 1, 1] = float(focal[f]); K[:, 0, 2] = float(pp[f, 0]); K[:, 1, 2] = float(pp[f, 1])
        with torch.no_grad():
            o = layer(rot_l[f].to(device), shape_l[f].to(device), transl_l[f].to(device), None, None, K=K.to(device),
                      expression=expr_l[f].to(device) if expr_l[f] is not None else None)
            # rest pelvis for the standard-smplx translation (pose-independent joint 0)
            z = lambda d: torch.zeros(n, d, device=device)  # noqa: E731
            p0 = bm(betas=shape_l[f].to(device), global_orient=z(3), body_pose=z(63), left_hand_pose=z(45),
                    right_hand_pose=z(45), jaw_pose=z(3), leye_pose=z(3), reye_pose=z(3), expression=z(10)).joints[:, 0]
        j3 = o["smpl_j3d"][:, :22].cpu().numpy()
        pel = o["smpl_transl_pelvis"][:, 0].cpu().numpy()
        for k in range(n):
            pid = int(id_l[f][k])
            tr = tracks.setdefault(pid, {"frames": [], "rotvec": [], "betas": [], "pelvis": [], "j22": [], "p0": []})
            tr["frames"].append(f); tr["rotvec"].append(rot_l[f][k].numpy()); tr["betas"].append(shape_l[f][k].numpy())
            tr["pelvis"].append(pel[k]); tr["j22"].append(j3[k]); tr["p0"].append(p0[k].cpu().numpy())

    persons = {}
    for pid, tr in tracks.items():
        fr = np.asarray(tr["frames"])
        if len(fr) < min_frames:
            continue
        full = lambda a, shp: _scatter(fr, np.asarray(a, dtype=np.float32), T, shp)  # noqa: E731
        rv = full(tr["rotvec"], (53, 3)); pel = full(tr["pelvis"], (3,))
        persons[pid] = {
            "observed": np.isin(np.arange(T), fr), "rotvec": rv, "betas": full(tr["betas"], (nb,)),
            "pelvis_cam": pel, "joints22_cam": full(tr["j22"], (22, 3)),
            "transl_std": pel - full(tr["p0"], (3,)),
        }
    return {"persons": persons, "c2w": c2w, "focal_net": focal, "net_hw": net_hw, "n_frames": T}


def _scatter(fr, a, T, shp):
    out = np.full((T,) + tuple(shp), np.nan, dtype=np.float32)
    out[fr] = a
    return out


def timing():
    return time.time()
