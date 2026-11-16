"""HaMeR (MANO hand regressor, ViT-H) wrapper: batches of 256x256 hand crops -> right-hand MANO poses.

Weights/layout (all baked into the image, see Dockerfile):
  $HAMER_DATA/_DATA/hamer_ckpts/{checkpoints/hamer.ckpt, model_config.yaml}
  $HAMER_DATA/_DATA/data/{mano/MANO_RIGHT.pkl, mano_mean_params.npz}
Code: github.com/geopavlakos/hamer (MIT), put on sys.path via $HAMER_REPO. Only the model is used (no detector / ViTPose / renderer).
"""
import os
import sys
import types

import numpy as np

HAMER_REPO = os.environ.get("HAMER_REPO", "/workspace/hamer")
HAMER_DATA = os.environ.get("HAMER_DATA", "/workspace/hamer_data")


def _import_hamer():
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    if HAMER_REPO not in sys.path:
        sys.path.insert(0, HAMER_REPO)
    try:
        import hamer.models  # noqa: F401
    except Exception as e:  # noqa: BLE001  (renderer deps are optional: we never render)
        if "pyrender" in str(e) or "OpenGL" in str(e) or "trimesh" in str(e):
            for m in [k for k in sys.modules if k.startswith("hamer")]:
                del sys.modules[m]
            for name in ("pyrender", "trimesh", "OpenGL", "OpenGL.GL"):
                sys.modules[name] = types.ModuleType(name)
            import hamer.models  # noqa: F401
        else:
            raise


class HandEstimator:
    def __init__(self, data_dir=HAMER_DATA, device="cuda", mano_dir=None, half=False):
        import torch

        _import_hamer()
        import hamer.configs as hc
        from hamer.configs import get_config
        from hamer.models import HAMER

        hc.CACHE_DIR_HAMER = data_dir
        ckpt = os.path.join(data_dir, "_DATA/hamer_ckpts/checkpoints/hamer.ckpt")
        cfg = get_config(os.path.join(data_dir, "_DATA/hamer_ckpts/model_config.yaml"), update_cachedir=True)
        cfg.defrost()
        if mano_dir:
            cfg.MANO.MODEL_PATH = mano_dir
        cfg.MODEL.BBOX_SHAPE = [192, 256]
        if "PRETRAINED_WEIGHTS" in cfg.MODEL.BACKBONE:
            cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
        cfg.freeze()
        model = HAMER(cfg, init_renderer=False)
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        bad = [k for k in missing if not k.startswith(("discriminator", "mano."))]
        if bad:
            raise RuntimeError(f"HaMeR checkpoint is missing weights: {bad[:5]}")
        self.torch, self.device, self.half = torch, torch.device(device), half
        self.model = model.to(self.device).eval()
        if half:
            self.model.backbone.half()
        self.mean = torch.tensor(cfg.MODEL.IMAGE_MEAN, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg.MODEL.IMAGE_STD, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.cfg = cfg

    def predict(self, patches_rgb, batch_size=32):
        """patches_rgb (N,256,256,3) uint8 RGB (left hands already mirrored).
        Returns dict: hand_pose (N,15,3,3), global_orient (N,3,3), kp2d (N,21,2) in patch pixels, cam (N,3)."""
        torch = self.torch
        out = {"hand_pose": [], "global_orient": [], "kp2d": [], "cam": []}
        with torch.no_grad():
            for i in range(0, len(patches_rgb), batch_size):
                x = torch.from_numpy(np.ascontiguousarray(patches_rgb[i:i + batch_size])).to(self.device)
                x = x.permute(0, 3, 1, 2).float() / 255.0
                x = (x - self.mean) / self.std
                if self.half:
                    x = x.half()
                o = self.model.forward_step({"img": x}, train=False)
                n = x.shape[0]
                out["hand_pose"].append(o["pred_mano_params"]["hand_pose"].float().reshape(n, 15, 3, 3).cpu().numpy())
                out["global_orient"].append(o["pred_mano_params"]["global_orient"].float().reshape(n, 3, 3).cpu().numpy())
                out["kp2d"].append((o["pred_keypoints_2d"].float().cpu().numpy() + 0.5) * x.shape[-1])
                out["cam"].append(o["pred_cam"].float().cpu().numpy())
        return {k: (np.concatenate(v) if v else np.zeros((0,))) for k, v in out.items()}
