# pt-mocap-worker

RunPod serverless worker: monocular video -> 3D human motion -> BVH, using
[GVHMR](https://github.com/zju3dv/GVHMR) (world-grounded human motion recovery). Built for pt-tracker.

Image: `ghcr.io/ironmindai/pt-mocap-worker:latest` (also tagged with the git sha).

## Endpoint contract

Input (`input` object of a RunPod job):

| field | type | default | notes |
|---|---|---|---|
| `video_url` | string | - | https URL. One of `video_url` / `video_b64` is required. |
| `video_b64` | string | - | base64 of the video file (mind RunPod's request size limit; prefer URLs). |
| `frame_stride` | int >= 1 | 1 | Keep every Nth frame of the output. Output fps = 30 / stride. GVHMR always runs at 30 fps internally. |
| `static_cam` | bool | false | true skips visual odometry (tripod / fixed camera). |
| `f_mm` | int | null | Full-frame-equivalent focal length in mm, if the camera is zoomed in. |
| `bvh_scale` | float | 1.0 | BVH units = meters * scale (100 gives centimeters). |
| `return_smpl` | bool | false | Also return raw params as `smpl_npz_b64`. |

Output:

```json
{"bvh_b64": "...", "fps": 30.0, "n_frames": 150, "duration_s": 5.0,
 "skeleton": "SMPL-22 (Y-up, +Z forward, meters*bvh_scale)", "elapsed_s": 41.2}
```

Errors are returned as `{"error": "..."}` (e.g. bad input, undecodable video, no person detected).

Limits (env overridable): `MAX_VIDEO_BYTES` 500 MB, `MAX_SECONDS` 120 (longer videos are truncated).
The video is re-encoded with ffmpeg to constant 30 fps first, so variable/60 fps sources are handled.

### BVH details

- Skeleton: 22 SMPL joints (Pelvis, hips, knees, ankles, feet, spine1-3, neck, collars, head, shoulders, elbows, wrists).
  Same tree and order as SMPL / SMPL-H / SMPL-X body joints. See `smpl_bvh.py` for names and parents.
- Rest-pose offsets come from the neutral SMPL-X model shaped with GVHMR's predicted betas (real proportions).
- Y-up, right-handed, subject faces +Z at frame 0, SMPL left is +X. Frame 0 starts at the XZ origin; feet are placed
  on y = 0 using the lowest foot joint over the clip (minus 3 cm sole offset).
- Root channels: `Xposition Yposition Zposition Zrotation Yrotation Xrotation`; other joints
  `Zrotation Yrotation Xrotation` (degrees, R = Rz * Ry * Rx). Leaf joints get an `End Site`.
- Root translation is GVHMR's world-grounded trajectory (gravity-aligned). With a moving camera and no `f_mm`,
  translation scale and drift are estimates.
- Fallback: pass `return_smpl: true` to also get `global_orient (T,3)`, `body_pose (T,63)`, `transl (T,3)`,
  `betas (T,10)` (axis-angle, SMPL-X convention, meters) as a compressed npz. Note these are raw GVHMR
  values, before the BVH re-basing (heading/ground).

## Build

GitHub Actions (`.github/workflows/build.yml`) builds on every push to `main` and pushes `latest` + `<git sha>`
to ghcr using the default `GITHUB_TOKEN` (`packages: write`). The build needs no GPU. Do not build locally:
the image is roughly 15 GB (weights included); the workflow frees runner disk before building.
GHA layer cache is used (`mode=min`); the cache limit is 10 GB per repo so the big layers may not all be cached.

The first push creates the ghcr package as private. To let RunPod pull without credentials, set the package to
public (GitHub -> Packages -> pt-mocap-worker -> Package settings), or add a registry credential in RunPod.

Base: `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime` (Python 3.10, torch 2.3.0 + cu121, what GVHMR's pinned
requirements and the prebuilt pytorch3d wheel expect). GVHMR is pinned to commit `ee960bb`.
Visual odometry uses GVHMR's default SimpleVO (SIFT); DPVO is not installed.

## Weights (baked into the image, nothing downloaded at cold start)

| file | source |
|---|---|
| `gvhmr/gvhmr_siga24_release.ckpt`, `hmr2/epoch=10-step=25000.ckpt`, `vitpose/vitpose-h-multi-coco.pth`, `yolo/yolov8x.pt` | HuggingFace mirror [`ryanrudes/gvhmr`](https://huggingface.co/ryanrudes/gvhmr) (pinned commit), mirror of the official Google-Drive release |
| `body_models/smplx/SMPLX_NEUTRAL.npz`, `body_models/smpl/SMPL_NEUTRAL.pkl` | HuggingFace [`lithiumice/models_hub`](https://huggingface.co/lithiumice/models_hub) (`4_SMPLhub/...`, pinned commit). Ungated community mirror of the SMPL/SMPL-X files |

Licensing: SMPL / SMPL-X are distributed by the Max Planck Institute under a registration-gated license
(non-commercial research unless a commercial license is obtained); GVHMR and the other checkpoints are
research-licensed as well. The mirrors are unofficial. Check licensing before commercial use of this worker's output
pipeline, and prefer swapping in files from your own registered download (change the two `*_MIRROR` build args or the
`curl` lines in the Dockerfile).

## Test

Deploy the image as a RunPod serverless endpoint (GPU with >= 24 GB recommended; container disk ~25 GB), then:

```bash
curl -s -X POST "https://api.runpod.ai/v2/$ENDPOINT_ID/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"input": {"video_url": "https://example.com/clip.mp4", "frame_stride": 2}}'
# -> {"id": "<job id>", "status": "IN_QUEUE"}
curl -s "https://api.runpod.ai/v2/$ENDPOINT_ID/status/<job id>" -H "Authorization: Bearer $RUNPOD_API_KEY"
```

Or end to end (submit, poll, write the BVH):

```bash
RUNPOD_API_KEY=... ENDPOINT_ID=... python3 test_local.py https://example.com/clip.mp4 -o out.bvh --stride 2
```

Open `out.bvh` in Blender (File -> Import -> Motion Capture) to eyeball it.

## Status

The image and handler have not been run end to end yet (build happens in CI, there is no GPU on the dev box).
`smpl_bvh.py` was unit-checked on a synthetic skeleton only. First deployed run should be checked visually.
