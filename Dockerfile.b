# LAB option B: RunPod serverless worker, Human3R (CUT3R + Multi-HMR) multi-person video -> SMPL-X world reconstruction.
# Weights baked in at build time (no network needed at cold start). Built only in GitHub Actions (no GPU needed).
# LICENSE WARNING: Human3R is CC BY-NC-SA 4.0 (CUT3R) + NAVER Multi-HMR non-commercial. Research evaluation only.
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    H3R_DIR=/workspace/Human3R H3R_CKPT_DIR=/workspace/Human3R/src \
    TORCH_HOME=/workspace/torch_cache FFMPEG_BIN=/usr/bin/ffmpeg
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ffmpeg build-essential libgl1 libglib2.0-0 libegl1 \
    && rm -rf /var/lib/apt/lists/*

# ---- Human3R code (pinned) ----
ARG H3R_COMMIT=402f2b2c7f20514e99cb42e4126c46b4ff75593f
RUN git clone https://github.com/fanegg/Human3R.git ${H3R_DIR} \
    && cd ${H3R_DIR} && git checkout ${H3R_COMMIT} && rm -rf .git

# Python deps (Human3R requirements.txt minus viewer/training-only packages; transformers pinned for torch 2.3).
RUN pip install "numpy==1.26.4" roma matplotlib tqdm opencv-python-headless scipy einops trimesh "pillow==10.3.0" \
        h5py accelerate "transformers==4.46.3" "huggingface-hub<1" scikit-learn hydra-core smplx \
        pyrender==0.1.45 "pyglet<2" imageio gsplat runpod requests

# DINOv2 hub code (Multi-HMR backbone is built via torch.hub with pretrained=False; weights come from the checkpoint)
RUN python -c "import torch; torch.hub.load('facebookresearch/dinov2','dinov2_vitl14',pretrained=False)"

COPY patch_human3r.py /tmp/patch_human3r.py
RUN python /tmp/patch_human3r.py ${H3R_DIR}

# ---- Weights / body models ----
# SMPL-X neutral + Multi-HMR mean params: ungated community mirror (official source needs registration), pinned sha.
# Human3R checkpoint: official HuggingFace repo faneggg/human3r (pinned revision).
ARG SMPL_MIRROR=https://huggingface.co/lithiumice/models_hub/resolve/363972831b553e4f5b62d11d509e98830d930c97
ARG H3R_HF_REV=1902f6b702547870994f15413ed4c61b19dca6b6
RUN mkdir -p ${H3R_DIR}/src/models/smplx \
    && DL="curl -fL --retry 5 --retry-delay 5 --retry-all-errors -o" \
    && $DL ${H3R_DIR}/src/models/smplx/SMPLX_NEUTRAL.npz ${SMPL_MIRROR}/4_SMPLhub/SMPLX/X_npz/SMPLX_NEUTRAL.npz \
    && $DL ${H3R_DIR}/src/models/smpl_mean_params.npz ${SMPL_MIRROR}/3_4DHumans/data/smpl_mean_params.npz
RUN curl -fL --retry 5 --retry-delay 5 --retry-all-errors -o ${H3R_DIR}/src/human3r_896L.pth \
        https://huggingface.co/faneggg/human3r/resolve/${H3R_HF_REV}/human3r_896L.pth

# Build-time sanity checks: file sizes + CPU-only import of everything the handler touches.
RUN ls -l ${H3R_DIR}/src ${H3R_DIR}/src/models/smplx \
    && test $(stat -c %s ${H3R_DIR}/src/human3r_896L.pth) -gt 4000000000 \
    && test $(stat -c %s ${H3R_DIR}/src/models/smplx/SMPLX_NEUTRAL.npz) -gt 100000000 \
    && /usr/bin/ffmpeg -hide_banner -encoders 2>/dev/null | grep -q libx264 \
    && cd ${H3R_DIR} && python -c "import sys; sys.path[:0]=['.','src']; import demo; from src.dust3r.model import ARCroco3DStereo; from src.dust3r.utils import SMPL_Layer; SMPL_Layer(type='smplx', gender='neutral', num_betas=10, kid=False, person_center='head'); import roma, smplx, runpod, gsplat; print('imports ok')"

COPY handler.py handler_b.py human3r_infer.py smplx_glb.py smpl_glb.py motion.py /workspace/
ENV PYTHONPATH=/workspace
CMD ["python", "-u", "/workspace/handler_b.py"]
