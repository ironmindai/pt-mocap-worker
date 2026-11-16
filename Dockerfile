# RunPod serverless worker: GVHMR monocular video -> SMPL-X motion -> BVH.
# All weights are baked in at build time (no network needed at cold start).
# Build runs in GitHub Actions (see .github/workflows/build.yml); needs no GPU.
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    GVHMR_DIR=/workspace/GVHMR
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ffmpeg build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---- GVHMR code (pinned) + python deps ----
ARG GVHMR_COMMIT=ee960bb6e2ea2d381aa97f08e9b71ef320b624b1
RUN git clone https://github.com/zju3dv/GVHMR.git ${GVHMR_DIR} \
    && cd ${GVHMR_DIR} && git checkout ${GVHMR_COMMIT} && rm -rf .git

# Drop dev-only packages and the ones that need special build handling (installed below).
# torch/torchvision (2.3.0 / 0.18.0, cu121) come from the base image, which matches the pytorch3d wheel in requirements.txt.
RUN cd ${GVHMR_DIR} \
    && grep -viE '^(torch|torchvision|jupyter|ipdb|black|chumpy|cython_bbox|lapx)\b' requirements.txt > /tmp/req.txt \
    && pip install -r /tmp/req.txt \
    && pip install "setuptools<70" wheel cython \
    && pip install --no-build-isolation chumpy cython_bbox lapx \
    && pip install -e . \
    && pip install runpod scipy requests \
    && rm -f /tmp/req.txt

# ---- Weights ----
# SMPL / SMPL-X body models: ungated community mirror on HuggingFace (official source needs registration).
# GVHMR / HMR2 / ViTPose / YOLOv8x checkpoints: HuggingFace mirror of the official Google-Drive release.
# Both pinned to a commit sha.
ARG SMPL_MIRROR=https://huggingface.co/lithiumice/models_hub/resolve/363972831b553e4f5b62d11d509e98830d930c97
ARG GVHMR_MIRROR=https://huggingface.co/ryanrudes/gvhmr/resolve/336150c798625df08c2249ff5c1179b737fca5b5
ENV CKPT=${GVHMR_DIR}/inputs/checkpoints
RUN mkdir -p ${CKPT}/body_models/smpl ${CKPT}/body_models/smplx ${CKPT}/gvhmr ${CKPT}/hmr2 ${CKPT}/vitpose ${CKPT}/yolo \
    && DL="curl -fL --retry 5 --retry-delay 5 --retry-all-errors -o" \
    && $DL ${CKPT}/body_models/smpl/SMPL_NEUTRAL.pkl   ${SMPL_MIRROR}/4_SMPLhub/SMPL/X_pkl/SMPL_NEUTRAL.pkl \
    && for g in MALE FEMALE; do \
         $DL ${CKPT}/body_models/smpl/SMPL_$g.pkl ${SMPL_MIRROR}/4_SMPLhub/SMPL/X_pkl/SMPL_$g.pkl \
         || { echo "WARNING: SMPL_$g.pkl unavailable, handler falls back to neutral"; rm -f ${CKPT}/body_models/smpl/SMPL_$g.pkl; }; \
       done \
    && $DL ${CKPT}/body_models/smplx/SMPLX_NEUTRAL.npz ${SMPL_MIRROR}/4_SMPLhub/SMPLX/X_npz/SMPLX_NEUTRAL.npz \
    && $DL ${CKPT}/gvhmr/gvhmr_siga24_release.ckpt     ${GVHMR_MIRROR}/gvhmr/gvhmr_siga24_release.ckpt \
    && $DL ${CKPT}/yolo/yolov8x.pt                     ${GVHMR_MIRROR}/yolo/yolov8x.pt
RUN DL="curl -fL --retry 5 --retry-delay 5 --retry-all-errors -o" \
    && $DL "${CKPT}/hmr2/epoch=10-step=25000.ckpt"     "${GVHMR_MIRROR}/hmr2/epoch=10-step=25000.ckpt" \
    && $DL ${CKPT}/vitpose/vitpose-h-multi-coco.pth    ${GVHMR_MIRROR}/vitpose/vitpose-h-multi-coco.pth

# Sanity checks at build time (file sizes + CPU-only import of the code we call).
RUN ls -l ${CKPT}/*/* && test $(stat -c %s ${CKPT}/body_models/smplx/SMPLX_NEUTRAL.npz) -gt 100000000 \
    && test $(stat -c %s ${CKPT}/vitpose/vitpose-h-multi-coco.pth) -gt 2000000000 \
    && /usr/bin/ffmpeg -hide_banner -encoders 2>/dev/null | grep -q libx264 \
    && python -c "import pytorch3d, smplx, chumpy, ultralytics, hydra, scipy, runpod; print('imports ok')"

# ---- HaMeR hand-pose regressor (`hands: true`): MIT code (pinned) + ungated weights (HF space, pinned) ----
# Only the model is used (crops come from the SMPL-X hand joints); detectron2 / ViTPose / mmcv are NOT needed.
ARG HAMER_COMMIT=3a01849f4148352e9260b69bf28b65d1671a4905
ARG HAMER_MIRROR=https://huggingface.co/spaces/geopavlakos/HaMeR/resolve/aabc78ee4fd1f1ccf1a55fdd4225bf84df0e680a/_DATA
ENV HAMER_REPO=/workspace/hamer HAMER_DATA=/workspace/hamer_data
RUN git clone https://github.com/geopavlakos/hamer.git ${HAMER_REPO} \
    && cd ${HAMER_REPO} && git checkout ${HAMER_COMMIT} && rm -rf .git third-party assets example_data docker \
    && pip install yacs \
    && mkdir -p ${HAMER_DATA}/_DATA/hamer_ckpts/checkpoints ${HAMER_DATA}/_DATA/data/mano \
    && DL="curl -fL --retry 5 --retry-delay 5 --retry-all-errors -o" \
    && $DL ${HAMER_DATA}/_DATA/hamer_ckpts/checkpoints/hamer.ckpt ${HAMER_MIRROR}/hamer_ckpts/checkpoints/hamer.ckpt \
    && $DL ${HAMER_DATA}/_DATA/hamer_ckpts/model_config.yaml      ${HAMER_MIRROR}/hamer_ckpts/model_config.yaml \
    && $DL ${HAMER_DATA}/_DATA/data/mano_mean_params.npz          ${HAMER_MIRROR}/data/mano_mean_params.npz \
    && $DL ${HAMER_DATA}/_DATA/data/mano/MANO_RIGHT.pkl           ${SMPL_MIRROR}/4_SMPLhub/MANO/pkl/MANO_RIGHT.pkl \
    && test $(stat -c %s ${HAMER_DATA}/_DATA/hamer_ckpts/checkpoints/hamer.ckpt) -gt 2000000000 \
    && test $(stat -c %s ${HAMER_DATA}/_DATA/data/mano/MANO_RIGHT.pkl) -gt 1000000
COPY hamer_infer.py /workspace/
RUN cd /workspace && python -c "import hamer_infer; hamer_infer._import_hamer(); import hamer.models; print('hamer import ok')"

COPY handler.py smpl_bvh.py smpl_glb.py motion.py multiperson.py scene.py smplx_np.py hands.py hands_pipeline.py hamer_infer.py /workspace/
CMD ["python", "-u", "/workspace/handler.py"]
