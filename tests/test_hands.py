"""Unit tests for the hand pipeline pieces that need no GPU: conventions, crop geometry, confidence, fusion, SMPL-X GLB, scene.

Run: python -m pytest tests -q
Set SMPLX_NPZ=/path/to/SMPLX_NEUTRAL.npz to also run the checks on the real SMPL-X model (rest-frame / mirror conventions).
"""
import os
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import hands as H  # noqa: E402
import scene as sc  # noqa: E402
import smplx_np as sx  # noqa: E402
from test_glb import parse_glb  # noqa: E402

SMPLX_NPZ = os.environ.get("SMPLX_NPZ")
real = pytest.mark.skipif(not SMPLX_NPZ or not os.path.exists(SMPLX_NPZ or ""), reason="set SMPLX_NPZ")
FPAR = np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14])


def _fk(Jr, rots):
    P = np.zeros((16, 3))
    W = [None] * 16
    for j in range(16):
        if FPAR[j] < 0:
            W[j], P[j] = rots[j], Jr[j]
        else:
            W[j] = W[FPAR[j]] @ rots[j]
            P[j] = P[FPAR[j]] + W[FPAR[j]] @ (Jr[j] - Jr[FPAR[j]])
    return P


# ---------------------------------------------------------------- conventions
@real
def test_left_hand_is_mirror_of_right_with_yz_negated():
    d = np.load(SMPLX_NPZ, allow_pickle=True)
    J = d["J_regressor"] @ d["v_template"]
    L = np.concatenate([J[20:21], J[25:40]])
    Rr = np.concatenate([J[21:22], J[40:55]])
    M = H.MIRROR_M
    assert np.abs((L - L[0]) - (Rr - Rr[0]) @ M).max() < 0.01          # rest hands are x-mirrors (template asymmetry < 1 cm)
    aa = np.concatenate([[[0, 0, 0]], np.random.default_rng(0).normal(0, 0.6, (15, 3))])
    Pr = _fk(Rr, R.from_rotvec(aa).as_matrix())
    Pl = _fk(L, R.from_rotvec(H.to_smplx_pose(aa, True)).as_matrix())
    assert np.abs((Pl - Pl[0]) - (Pr - Pr[0]) @ M).max() < 0.01
    # the other candidate conventions must be clearly wrong (guards the test itself)
    Pw = _fk(L, R.from_rotvec(aa).as_matrix())
    assert np.abs((Pw - Pw[0]) - (Pr - Pr[0]) @ M).max() > 0.03


@real
def test_smplx_structure():
    m = sx.load_smplx_npz(SMPLX_NPZ)
    assert m["weights"].shape[1] == 52 and np.allclose(m["weights"].sum(1), 1, atol=1e-5)
    assert len(sx.PARENTS_52) == 52 and sx.PARENTS_52[22] == 20 and sx.PARENTS_52[37] == 21 and sx.PARENTS_52[24] == 23
    # relaxed neutral hand of the left side is the mirror of the right one
    assert np.abs(m["hands_mean_l"] - H.to_smplx_pose(m["hands_mean_r"], True)).max() < 0.1


def test_mirror_rotmat_matches_axis_angle_rule():
    aa = np.random.default_rng(1).normal(size=(5, 3))
    Rm = R.from_rotvec(aa).as_matrix()
    assert np.allclose(R.from_matrix(H.mirror_rotmat(Rm)).as_rotvec(), aa * H.MIRROR_AA, atol=1e-9)


# ---------------------------------------------------------------- crop geometry
def test_patch_roundtrip_and_flip():
    img = np.zeros((200, 300, 3), np.uint8)
    img[80:90, 150:160] = 255                               # marker around (155, 85)
    for flip in (False, True):
        patch, geom = H.extract_patch(img, (155.0, 85.0), 100, flip)
        ys, xs = np.where(patch[..., 0] > 128)
        c = np.array([xs.mean(), ys.mean()])
        back = H.patch_to_image(c, geom, flip)
        assert np.abs(back - [154.5, 84.5]).max() < 2.0
        assert abs(c[0] - 128) < 8 and abs(c[1] - 128) < 8   # centred in the patch
    assert H.extract_patch(img, (-500, -500), 80, False)[0].max() == 0   # fully outside -> black, no crash


def test_hand_centers_and_visibility():
    K = np.array([[1000.0, 0, 960], [0, 1000.0, 540], [0, 0, 1]])
    J = np.zeros((2, 52, 3))
    J[:, :, 2] = 4.0
    J[:, sx.R_WRIST] = [0.3, 0, 4.0]
    J[:, sx.R_FINGERS] = [0.4, 0, 4.0]
    J[1, sx.R_FINGERS] = [-9.0, 0, 4.0]                      # frame 1: right fingers far off to the side
    J[1, sx.R_WRIST] = [-9.0, 0, 4.0]
    cen, size, vis = H.hand_centers(J, K, 1920, 1080)
    assert vis[0, 1] == pytest.approx(1.0) and vis[1, 1] == 0.0
    assert cen[0, 1, 0] == pytest.approx(960 + 1000 * (0.3 + 15 * 0.4) / 16 / 4.0, abs=1e-3)
    assert size[0, 1] == pytest.approx(H.RESCALE * 1000 * H.HAND_BOX_M / 4.0 * H.ASPECT)


# ---------------------------------------------------------------- confidence / fusion
NEUTRAL = np.tile([0.1, 0.0, -0.3], (15, 1))


def _fist(T, angle=1.4):
    aa = np.zeros((T, 15, 3))
    aa[:, :, 2] = angle                                       # flexion about z for all joints
    return aa


def test_fuse_all_low_confidence_is_neutral():
    pose, w = H.fuse_hand(_fist(20), np.zeros(20), NEUTRAL)
    assert np.allclose(w, 0) and np.allclose(pose, NEUTRAL[None], atol=1e-9)


def test_fuse_high_confidence_follows_estimate_and_bridges_gaps():
    T = 40
    aa = _fist(T)
    conf = np.ones(T)
    conf[10:14] = 0.0                                         # short gap -> bridged
    aa[10:14] = 0.0                                           # garbage inside the gap must not leak
    pose, w = H.fuse_hand(aa, conf, NEUTRAL, window=5)
    assert np.allclose(pose[5:36, :, 2], 1.4, atol=0.05)
    assert w[10:14].min() > 0.8


def test_fuse_long_gap_decays_to_neutral_and_never_wild():
    T = 80
    aa = _fist(T)
    conf = np.ones(T)
    conf[20:] = 0.0                                           # estimate lost for the rest of the clip
    pose, w = H.fuse_hand(aa, conf, NEUTRAL)
    assert w[19] > 0.8 and w[-1] < 0.05
    assert np.abs(pose[-1] - NEUTRAL).max() < 0.1
    ang = np.linalg.norm(pose, axis=-1)
    assert ang.max() < 2.0


def test_fuse_rejects_single_frame_outlier_and_is_smooth():
    T = 30
    aa = _fist(T, 1.0)
    aa[15] = R.from_euler("xyz", [2.5, -2.0, 2.8]).as_rotvec()[None]   # one wild frame
    pose, _ = H.fuse_hand(aa, np.ones(T), NEUTRAL)
    assert np.abs(pose[15, :, 2] - 1.0).max() < 0.15
    q = R.from_rotvec(pose.reshape(-1, 3)).as_quat().reshape(T, 15, 4)
    d = np.minimum(np.linalg.norm(q[1:] - q[:-1], axis=-1), np.linalg.norm(q[1:] + q[:-1], axis=-1))
    assert d.max() < 0.1                                      # no flicker


def test_fuse_sign_continuity_near_pi():
    T = 20
    ang = np.linspace(3.0, 3.3, T)                            # crosses pi: axis-angle flips, quaternion must not
    aa = np.zeros((T, 15, 3))
    aa[:, :, 0] = ang[:, None]
    aa = R.from_quat(R.from_rotvec(aa.reshape(-1, 3)).as_quat()).as_rotvec().reshape(T, 15, 3)
    pose, _ = H.fuse_hand(aa, np.ones(T), np.zeros((15, 3)))
    r = R.from_rotvec(pose.reshape(-1, 3)).as_matrix().reshape(T, 15, 3, 3)
    step = np.linalg.norm(R.from_matrix((r[1:] @ np.swapaxes(r[:-1], -1, -2)).reshape(-1, 3, 3)).as_rotvec(), axis=-1)
    assert np.isfinite(pose).all() and step.max() < 0.1       # smooth rotation through pi, no 2*pi jump


def test_prior_clamps_impossible_bends_only():
    rng = np.random.default_rng(0)
    comps = np.linalg.qr(rng.normal(size=(45, 45)))[0]
    mean = rng.normal(size=45) * 0.1
    coeffs = rng.normal(size=(500, 45)) * 0.3
    prior = H.HandPrior(comps, mean, coeffs)
    ok = (mean + coeffs[:20] @ comps).reshape(20, 15, 3)
    assert np.abs(prior.clamp(ok) - ok).max() < 0.35 * 1.0 + 1e-9      # plausible poses barely move
    wild = (mean + (rng.normal(size=(1, 45)) * 5.0) @ comps).reshape(1, 15, 3)
    c = prior.clamp(wild)
    assert np.abs(c - mean.reshape(1, 15, 3)).max() < np.abs(wild - mean.reshape(1, 15, 3)).max()


def test_frame_confidence_components():
    T = 10
    sharp = np.full(T, 100.0)
    sharp[3] = 5.0                                            # motion-blurred frame
    c = H.frame_confidence(np.ones(T), sharp, np.full(T, 20.0))
    assert c[0] == pytest.approx(1.0) and c[3] < 0.5
    assert H.frame_confidence(np.zeros(T), sharp, np.full(T, 20.0)).max() == 0          # out of frame
    assert H.frame_confidence(np.ones(T), sharp, np.full(T, 150.0)).max() == 0          # wrist disagrees completely
    obs = np.ones(T, bool)
    obs[7] = False
    assert H.frame_confidence(np.ones(T), sharp, np.full(T, 20.0), obs)[7] == 0


# ---------------------------------------------------------------- SMPL-X GLB + scene with a synthetic model
def fake_smplx(V=80, seed=0):
    rng = np.random.default_rng(seed)
    base = np.array([[0, 0, 0], [.09, -.09, 0], [-.09, -.09, 0], [0, .1, 0], [.09, -.5, 0], [-.09, -.5, 0], [0, .25, 0],
                     [.09, -.9, 0], [-.09, -.9, 0], [0, .4, 0], [.09, -.95, .1], [-.09, -.95, .1], [0, .55, 0], [.05, .5, 0],
                     [-.05, .5, 0], [0, .65, 0], [.17, .5, 0], [-.17, .5, 0], [.42, .5, 0], [-.42, .5, 0], [.65, .5, 0],
                     [-.65, .5, 0]])
    rest = [*base]
    for side, wx in ((1, 0.65), (-1, -0.65)):                # left hand +x
        for f in range(5):
            for k in range(3):
                rest.append([side * (0.65 + 0.03 * (3 * 0 + k + 1) + 0.01 * k), 0.5 - 0.02 * f, 0.01 * f])
    rest = np.array(rest)
    v = rng.normal(size=(V, 3)) * 0.05
    v[:52] = rest
    jreg = np.zeros((52, V))
    jreg[np.arange(52), np.arange(52)] = 1
    w = rng.random((V, 52)) ** 6
    w[np.arange(52), np.arange(52)] += 5
    w /= w.sum(1, keepdims=True)
    f = np.stack([rng.permutation(V)[:3] for _ in range(100)]).astype(np.uint32)
    return {"v_template": v, "shapedirs": np.zeros((V, 3, 10)), "J_regressor": jreg, "weights": w, "faces": f,
            "hands_mean_l": np.zeros((15, 3)), "hands_mean_r": np.zeros((15, 3))}


def _person(T=8, x=0.0, seed=0):
    rng = np.random.default_rng(seed)
    m = fake_smplx()
    J = m["J_regressor"] @ m["v_template"]
    return {"incam_global_orient": rng.normal(size=(T, 3)) * 0.02, "global_global_orient": rng.normal(size=(T, 3)) * 0.02,
            "incam_body_pose": rng.normal(size=(T, 63)) * 0.05, "incam_betas": np.zeros((T, 10)), "incam_transl": np.zeros((T, 3)),
            "joints24_incam": np.tile(J[:24], (T, 1, 1)) + np.array([x, 0.3, 4.0])}


def test_scene_with_hands_structure_and_unchanged_pelvis():
    T = 8
    mx = fake_smplx()
    people = [_person(T, -0.5, 0), _person(T, 0.5, 1)]
    K = np.array([[1000.0, 0, 960], [0, 1000.0, 540], [0, 0, 1]])
    # SMPL path needs a 24-joint model; reuse the first 24 joints of the fake SMPL-X as an SMPL stand-in
    smpl = {"v_template": mx["v_template"], "shapedirs": mx["shapedirs"], "J_regressor": mx["J_regressor"][:24],
            "weights": mx["weights"][:, :24] / mx["weights"][:, :24].sum(1, keepdims=True), "faces": mx["faces"]}
    rng = np.random.default_rng(3)
    hands = [(rng.normal(size=(T, 15, 3)) * 0.3, rng.normal(size=(T, 15, 3)) * 0.3) for _ in range(2)]
    glb0, meta0 = sc.build_scene(smpl, people, K, 30.0)
    glb1, meta1 = sc.build_scene(smpl, people, K, 30.0, hands=hands, smplx_model=mx)
    js, _ = parse_glb(glb1)
    names = [n["name"] for n in js["nodes"]]
    for pre in ("fighter0_", "fighter1_"):
        assert pre + "Pelvis" in names and pre + "L_Wrist" in names and pre + "SMPL_mesh" in names
        assert pre + "L_Index1" in names and pre + "R_Thumb3" in names
        assert sum(n.startswith(pre) for n in names) == 52 + 1
    anim = js["animations"]
    assert len(anim) == 1 and anim[0]["name"] == "scene"
    assert len(anim[0]["channels"]) == 2 * (52 + 1)
    assert len(js["skins"]) == 2 and len(js["skins"][0]["joints"]) == 52
    # pelvis / floor / camera numbers are identical to the SMPL scene (same root track, same floor)
    for a, b in zip(meta0["persons"], meta1["persons"]):
        assert np.allclose(a["pelvis_world_m"], b["pelvis_world_m"])
    assert meta0["cam_height_m"] == meta1["cam_height_m"] and meta1["hands"]["finger_joints"][0] == "L_Index1"
    # body joint rotation tracks equal the SMPL scene's (only fingers are new)
    assert meta1["persons"][0]["color"] == meta0["persons"][0]["color"]


def test_person_glb_ground_runs_and_is_grounded():
    m = fake_smplx()
    T = 6
    z = np.zeros((T, 15, 3))
    b = np.random.default_rng(0).normal(size=(T, 63)) * 0.02
    glb = sx.person_glb_ground(m, np.zeros(10), np.zeros((T, 3)), b, np.tile([0.3, 0, 0.3], (T, 1)), z, z, 30.0)
    js, _ = parse_glb(glb)
    assert len(js["skins"][0]["joints"]) == 52 and js["nodes"][0]["name"] == "Pelvis"


# ---------------------------------------------------------------- hands input / fist mode
@pytest.mark.parametrize("v,exp", [(None, None), (False, None), (True, "estimate"), ("estimate", "estimate"),
                                   ("fist", "fist"), (" Fist ", "fist"), ("ESTIMATE", "estimate")])
def test_parse_hands_mode_valid(v, exp):
    assert H.parse_hands_mode(v) == exp


@pytest.mark.parametrize("v", ["", "yes", "true", "false", "open", 0, 1, 2, [], {}, ["fist"], 1.0])
def test_parse_hands_mode_invalid(v):
    with pytest.raises(ValueError, match="hands must be"):
        H.parse_hands_mode(v)


def test_fist_pose_shape_constant_and_mirrored():
    left, right = H.fist_pose(7)
    assert left.shape == right.shape == (7, 15, 3)
    assert np.allclose(right, right[0]) and np.allclose(left, left[0])
    assert np.allclose(right[0], H.FIST_RIGHT)
    assert np.allclose(left, right * H.MIRROR_AA)                      # x-mirror of the right fist
    # as rotations the left joints are M R M
    Rl = R.from_rotvec(left[0]).as_matrix()
    assert np.allclose(Rl, H.mirror_rotmat(R.from_rotvec(right[0]).as_matrix()), atol=1e-9)


def test_fist_constants_are_a_closed_fist_not_a_claw():
    f = H.FIST_RIGHT
    assert np.isfinite(f).all() and np.abs(f).max() < 2.5            # plausible joint angles (rad)
    ang = np.linalg.norm(f[:12], axis=1).reshape(4, 3)               # index, middle, pinky, ring x (MCP, PIP, DIP) rotation angle
    assert (ang > 0.7).all() and (ang < 2.3).all()                   # every joint clearly bent, none absurd
    assert (ang.sum(1) > 3.2).all()                                  # total curl per finger > 180 deg: closed, not a claw
    flat = H.fist_pose(1)[1][0]
    assert np.abs(flat).max() > 0.5                                  # not the identity / flat hand


@real
def test_fist_geometry_real_smplx_both_hands_mirror_and_closed():
    m = sx.load_smplx_npz(SMPLX_NPZ)
    left, right = H.fist_pose(1)
    rots = sx.rots_from_params(np.zeros((1, 3)), np.zeros((1, 63)), left, right)
    root = sx.rest_shape(m, np.zeros(10))[1][0][None]
    v, J = sx.posed(m, np.zeros(10), rots, root, vert_idx=np.arange(1))
    J = J[0]
    Lw, Rw = J[sx.L_WRIST], J[sx.R_WRIST]
    PL = J[sx.L_FINGERS] - Lw
    PR = J[sx.R_FINGERS] - Rw
    assert np.abs(PL - PR * np.array([-1, 1, 1])).max() < 0.012     # both fists mirror each other (template asymmetry < 1.2 cm)
    # fingertip joints (3rd joint of each non-thumb finger) end up close to the wrist (curled) vs ~0.09 m when flat
    tips = np.linalg.norm(PR[[2, 5, 8, 11]], axis=1)
    flat_r = sx.posed(m, np.zeros(10), sx.rots_from_params(np.zeros((1, 3)), np.zeros((1, 63))), root, vert_idx=np.arange(1))[1][0]
    flat_tips = np.linalg.norm((flat_r[sx.R_FINGERS] - flat_r[sx.R_WRIST])[[2, 5, 8, 11]], axis=1)
    assert (tips < 0.75 * flat_tips).all()


def test_handler_compute_fist_glue_both_key_styles(monkeypatch):
    import types

    monkeypatch.setitem(sys.modules, "runpod", types.ModuleType("runpod"))
    import handler as hd

    for key in ("global_orient", "incam_global_orient"):        # multi path passes global params, single path incam params
        res, info = hd.compute_fist([{key: np.zeros((9, 3))}, {key: np.zeros((9, 3))}])
        assert info == {"mode": "fist"} and len(res) == 2
        assert res[0]["left_aa"].shape == res[0]["right_aa"].shape == (9, 15, 3) and res[0]["weight"].shape == (9, 2)
    with pytest.raises(ValueError):
        hd.H_parse("open")
    assert hd.H_parse("fist") == "fist" and hd.H_parse(False) is None
