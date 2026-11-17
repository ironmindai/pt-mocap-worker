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
| `static_cam` | bool | false | true skips visual odometry (tripod / fixed camera): faster and recommended for tripod footage. |
| `f_mm` | int | null | Full-frame-equivalent focal length in mm, if the camera is zoomed in. |
| `bvh_scale` | float | 1.0 | BVH units = meters * scale (100 gives centimeters). |
| `gender` | `neutral` / `male` / `female` | neutral | SMPL body model used for the GLB mesh and its skinning. Falls back to neutral if that model file is unavailable. |
| `smoothing` | `off` / `light` / `medium` / `strong` | medium | Temporal smoothing of pose and root translation before export (see below). |
| `return_smpl` | bool | false | Also return the (smoothed, un-rebased) params as `smpl_npz_b64`. |

Output:

```json
{"bvh_b64": "...", "glb_b64": "...", "gender_used": "neutral", "smoothing": "medium",
 "fps": 30.0, "n_frames": 150, "duration_s": 5.0,
 "skeleton": "SMPL-22 (Y-up, +Z forward, meters*bvh_scale)", "elapsed_s": 41.2}
```

If the GLB step fails the response still carries the BVH and a `glb_error` string instead of `glb_b64`.

### Smoothing

Applied at the native 30 fps (before `frame_stride` subsampling). Each joint rotation becomes a quaternion, sign
continuity is enforced (flip when dot < 0), a Savitzky-Golay filter (polyorder 2) runs per component, and the
result is renormalized. Windows: light 5, medium 7, strong 11 frames; root translation gets a gentler window
(5 / 7 / 9). Windows shrink automatically for very short clips. `off` returns the raw GVHMR motion (use it to get
unsmoothed output; BVH and GLB are both built from the same smoothed motion).

### Clothing (painted-on vertex colors)

Regions come from the model's own rest pose and skinning weights (dominant joint per vertex), so they follow body
shape. Colors are softened with one 1-ring smoothing pass.

| region | rule | color |
|---|---|---|
| shorts (all genders) | dominant joint pelvis / left hip / right hip (0, 1, 2), above 60% of the way from hip to knee joint (mid-thigh hem) | deep red `#8C1D18` |
| waistband | top 4 cm of the shorts | gold `#C9A227` |
| sports top (female and neutral, not male) | dominant joint spine1/2/3 (3, 6, 9) above the lower-chest cutoff (40% of the way from spine1 to spine2 joint upward); arms, shoulders, neck excluded | dark charcoal `#2B2D31` |
| everything else | skin | skin tone (linear 0.80, 0.62, 0.52) |

Vertex counts on the real neutral SMPL model (before smoothing): skin 5328, shorts 646, waistband 70, top 846.
Male: skin 6188, shorts 645, waistband 57, top 0.

### GLB details

A skinned, animated glTF 2.0 binary (built in `smpl_glb.py`, no Blender):

- Mesh: SMPL body, 6890 vertices, one primitive, shaped with GVHMR's betas. Look is driven by a `COLOR_0`
  vertex attribute (float RGBA, linear); the PBR material is white (baseColorFactor 1,1,1,1), metallic 0,
  roughness 0.6, single sided. three.js GLTFLoader applies COLOR_0 automatically.
- Skeleton: 24 SMPL joints as glTF nodes (node index = SMPL joint index, pelvis is the root; hand joints stay at
  identity). Inverse bind matrices are translations by minus the shaped rest-pose joints.
- Skinning: SMPL LBS weights pruned to the top 4 influences per vertex (renormalized) in `JOINTS_0` / `WEIGHTS_0`.
  SMPL pose-corrective blendshapes cannot be expressed in glTF skinning and are omitted.
- Animation: one LINEAR rotation track per joint (quaternions, local) plus a root translation track, at output fps.
- Conventions as BVH: Y-up, meters, facing +Z at frame 0, start at XZ origin, lowest foot over the clip on y = 0.
- Shape caveat: GVHMR predicts SMPL-X (neutral) betas; they are applied to the chosen SMPL model's shape space,
  so proportions are close but not identical to the SMPL-X body. `gender` selects the model, it does not estimate gender.
- BVH skeleton rest offsets still come from the neutral SMPL-X model.

Errors are returned as `{"error": "..."}` (e.g. bad input, undecodable video, no person detected); unexpected failures include the full Python traceback.

Camera motion: with `static_cam` false, GVHMR's SimpleVO (SIFT + pycolmap) estimates camera rotation. Frame pairs where pycolmap cannot recover a pose (static / low-texture scenes) are treated as no camera motion instead of crashing.

Limits (env overridable): `MAX_VIDEO_BYTES` 500 MB, `MAX_SECONDS` 120 (longer videos are truncated).
The video is re-encoded with ffmpeg to constant 30 fps first, so variable/60 fps sources are handled.

## Sparring scene mode (`scene: true`, branch `sparring-scene`, image `ghcr.io/ironmindai/pt-mocap-lab:scene-<sha>`)

Returns one ready-to-use GLB with two animated fighters; no SMPL weights or merge scripts needed on the app side.

Input (all the usual ones plus): `scene: true`; optional `track_ids` / `person_rank` / `bbx_override` (default: the two
largest tracks, `person_rank [0,1]`; error if fewer than two); `person_colors` (default `["#8C1D18", "#1F4FC4"]`, red / blue
shorts); `smoothing`; `static_cam`; `scene_gender` (default `neutral`); `return_persons: true` to also get the per-person
npz / stand-alone GLBs (omitted by default in scene mode). `list_tracks` additionally returns `suggested` (ids of the two
largest tracks) and `sample_every` (3; `tracks[].bboxes` = `[frame, x1, y1, x2, y2]` every 3rd frame, in video pixels, 30 fps timeline).

Output: `scene_glb_b64`, `scene_meta`, plus `tracks`, `n_frames`, `width`, `height`, `K_fullimg`, `persons` (summary only), `elapsed_s`.

GLB: glTF 2.0, Y up, metres, floor at y = 0 (10th percentile of the lowest foot vertex over both fighters, pooled), 30 fps.
Two skinned SMPL meshes, nodes prefixed `fighter0_` / `fighter1_` (`fighter0_Pelvis` ... `fighter0_R_Hand`, `fighter0_SMPL_mesh`;
skins / meshes / materials prefixed too), 25 nodes per fighter (24 joints + mesh), root nodes of the scene = each fighter's pelvis
and mesh node. ONE animation clip named `scene` holding all 50 channels (per fighter: 24 rotation tracks + pelvis translation),
so `mixer.clipAction(gltf.animations[0]).play()` animates both. Gravity: world-up in the camera frame = mean of both persons'
GVHMR gravity-view estimates, minimal rotation to +Y (no extra yaw: the world frame is the camera heading, camera looks along +Z,
+X is the viewer's left). Camera is at `(0, cam_height_m, 0)`; `scene_meta.camera.R_world_from_three_cam` is the rotation matrix for a
three.js camera that reproduces the video view (fov = `fov_y_deg`, aspect = width/height). Scale is monocular (GVHMR depth),
`cam_height_m` is derived, not measured. Frame a free viewer on the midpoint of both `pelvis_world_m` tracks.

`scene_meta`: `fps, n_frames, duration_s, units, up_axis, clip, cam_height_m, floor_y, K, fov_y_deg, width, height, camera, gravity,
persons[{index, node_prefix, pelvis_node, mesh_node, color, track_ids, observed_frames, foot_y_p10_m, bbox_xyxy{first,mean,last},
pelvis_world_m (T x 3)}]`. Merge maths lives in `scene.py` (tests: `tests/test_scene.py`).

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

## Hands (`hands: true`, branch `hands`, image `ghcr.io/ironmindai/pt-mocap-lab:hands-<sha>`)

Adds per-finger pose estimation (HaMeR, MANO ViT-H) on top of GVHMR. Default `hands: false` = behaviour identical to before.

Inputs (all modes: single-person `glb_b64`, multi-person per-person `glb_b64`, sparring `scene_glb_b64`):
- `hands` - `false`/`null`/absent = off (default, behaviour identical to before); `true` or `"estimate"` = estimate fingers with HaMeR;
  `"fist"` = **no HaMeR** (weights not loaded, no crops, much cheaper): both hands get a constant closed boxer's fist (thumb outside
  over the fingers) on every frame - meant for gloved clips where HaMeR is unreliable. Anything else -> `{"error": "bad request: hands must be ..."}`
  (validated before any video work). In all non-off modes the GLB mesh becomes **SMPL-X** (neutral; `gender` only affects the painted
  clothing), same 52-node structure, painted clothing, scene/single/per-person paths and positions. `fist` returns `hands: {"mode": "fist"}`.
- `return_hands` bool - also return `hands_npz_b64`.

How it works: each fighter's camera-frame SMPL-X joints are projected with K to get a hand crop per hand (no detector); left hands
are mirrored; HaMeR runs in two passes (second pass re-centres on its own 2D keypoints); per-frame confidence = visibility x
sharpness (motion blur) x agreement of HaMeR's wrist with the body wrist; poses are clamped to the MANO training-pose range,
low-confidence frames are bridged (<= 0.5 s) or held and faded to the relaxed SMPL-X `hands_mean` pose, then quaternion
Savitzky-Golay (window 5, sign-continuous). Body motion, pelvis track, gravity and floor are unchanged (same root, same body rotations,
floor from the SMPL feet). Wrist orientation always comes from the body model.

GLB skeleton (per figure, prefixed `fighter0_` / `fighter1_` in scenes): 52 nodes = the 22 SMPL body nodes (`Pelvis`, `L_Hip`, ..., `L_Wrist`,
`R_Wrist`; no `L_Hand`/`R_Hand`) + 30 finger nodes `{L,R}_{Index,Middle,Pinky,Ring,Thumb}{1,2,3}` (first joint parented to the wrist),
plus `SMPL_mesh`. One rotation track per node, ONE clip. Rotations are absolute local rotations from the SMPL-X template (flat hand =
identity). Extra response keys: `hands` (estimate: {hand_crops, hamer_s, mean_weight}; fist: {mode: "fist"}), `hands_error` (then the plain SMPL output is returned).

`hands_npz_b64` keys (np.savez): per person `p{i}_left_hand_pose`, `p{i}_right_hand_pose` (T,15,3) SMPL-X axis-angle (fused; MANO joint order
index, middle, pinky, ring, thumb x 3; left = right-equivalent with y,z negated), `p{i}_weight` (T,2) final blend weight [L,R]
(1 = estimate, 0 = relaxed neutral), `p{i}_conf_raw`, `p{i}_raw_left`, `p{i}_raw_right` (unfused), `p{i}_kp_img` (T,2,21,2) HaMeR keypoints in
image px, `p{i}_agree_deg`; `neutral_hand_left/right`, `fps`. For Mixamo retargeting use the (T,15,3) tracks + `weight`.

Fist mode image: `ghcr.io/ironmindai/pt-mocap-lab:hands-02d71a92dc7d615870eb1c438c6ff1ad9bb1d6df` (branch `hands-fist`); both RunPod templates
(`q06rkli9cz` sparring endpoint `n3emnmhsycwdzn`, `20k942w10w` production endpoint `gk1lt0rkaqdto9`) were pinned to it on 2026-11-17.
Rollback tag (previous, estimate-only): `ghcr.io/ironmindai/pt-mocap-lab:hands-dc034f07717fed3431b287eb4e303a42045be3d5`. Measured on a
real GPU: sparring clip (121 frames, scene) 52.6 s exec, single clip (147 frames) 40.5 s exec, with `hands: "fist"` (no HaMeR load).

Fist constants (`hands.FIST_RIGHT`): MANO/SMPL-X right-hand absolute axis-angle of a real MANO training pose (`MANO_RIGHT.pkl`
`hands_mean + hands_coeffs @ hands_components`, pose #515: tightly curled fingers, thumb folded over index/middle), rounded to 3
decimals; the left hand is the x-mirror (`to_smplx_pose`: y, z negated). Checked by rendering the SMPL-X mesh (both hands) and by unit
tests (`tests/test_hands.py`). With `return_hands` and fist, `hands_npz_b64` only has `p{i}_left/right_hand_pose`, `p{i}_weight` (=1),
`neutral_hand_*`, `fps`.

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
| `body_models/smplx/SMPLX_NEUTRAL.npz`, `body_models/smpl/SMPL_{NEUTRAL,MALE,FEMALE}.pkl` (all three verified present on the mirror; male/female are the original 247 MB v1.1.0 files) | HuggingFace [`lithiumice/models_hub`](https://huggingface.co/lithiumice/models_hub) (`4_SMPLhub/...`, pinned commit). Ungated community mirror of the SMPL/SMPL-X files |

Licensing: SMPL / SMPL-X are distributed by the Max Planck Institute under a registration-gated license
(non-commercial research unless a commercial license is obtained); GVHMR and the other checkpoints are
research-licensed as well. The mirrors are unofficial. Check licensing before commercial use of this worker's output
pipeline, and prefer swapping in files from your own registered download (change the two `*_MIRROR` build args or the
`curl` lines in the Dockerfile).

## Test

Unit tests (GLB builder, smoothing, BVH; no GPU or weights needed):
`pip install numpy scipy pygltflib pytest && python -m pytest tests -q`
(set `SMPL_PKL=/path/to/SMPL_NEUTRAL.pkl` to also run against the real SMPL model).

Endpoint test:

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
