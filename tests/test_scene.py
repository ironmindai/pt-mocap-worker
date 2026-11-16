"""Unit tests for the two-fighter scene merge (no GPU, no weights). Run: python -m pytest tests -q"""
import json
import os
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import scene as sc  # noqa: E402
from test_glb import fake_model_fixed, parse_glb  # noqa: E402

T = 6
PITCH = np.radians(20)
K = np.array([[1000.0, 0, 960], [0, 1000.0, 540], [0, 0, 1]])


def make_person(x, seed):
    """Standing person in the OpenCV camera frame; camera pitched down by PITCH, so world-up in camera coords is
    (0, -cos, -sin) ... built via a known R_c2gv."""
    rng = np.random.default_rng(seed)
    R_c2gv = R.from_euler("x", PITCH).as_matrix()
    inc = R.from_rotvec(rng.normal(size=(T, 3)) * 0.02).as_matrix()
    glo = R_c2gv @ inc
    model = fake_model_fixed()
    J = model["J_regressor"] @ model["v_template"]
    j24 = np.tile(J, (T, 1, 1)) + np.array([x, 0.3, 4.0])
    return {"incam_global_orient": R.from_matrix(inc).as_rotvec(), "global_global_orient": R.from_matrix(glo).as_rotvec(),
            "incam_body_pose": np.zeros((T, 63)), "incam_betas": np.zeros((T, 10)), "incam_transl": np.zeros((T, 3)),
            "joints24_incam": j24}


@pytest.fixture
def people():
    return [make_person(-0.5, 0), make_person(0.5, 1)]


def test_rot_between():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b = rng.normal(size=3), rng.normal(size=3)
        M = sc.rot_between(a, b)
        assert np.allclose(M @ M.T, np.eye(3)) and np.isclose(np.linalg.det(M), 1)
        assert np.allclose(M @ (a / np.linalg.norm(a)), b / np.linalg.norm(b))
    a = np.array([0, 1.0, 0])
    assert np.allclose(sc.rot_between(a, -a) @ a, -a)
    assert np.allclose(sc.rot_between(a, a), np.eye(3))


def test_gravity_frame_maps_up_to_plus_y(people):
    G, Wr, up_cv, ang = sc.gravity_frame(people)
    assert ang < 1e-6
    expect_up = R.from_euler("x", PITCH).as_matrix()[1]
    # median over frames of the (noisy) per-frame up is within a degree of the constructed one
    assert np.degrees(np.arccos(np.clip(up_cv @ expect_up, -1, 1))) < 2
    assert np.allclose(Wr @ up_cv, [0, 1, 0], atol=1e-6)
    assert np.isclose(np.linalg.det(Wr), 1) and np.allclose(Wr @ Wr.T, np.eye(3))


def test_gravity_average_of_two_estimates():
    a, b = make_person(0, 0), make_person(0, 1)
    # tilt person b's estimate by 4 degrees about x -> merged up is halfway
    b["global_global_orient"] = R.from_matrix(R.from_euler("x", np.radians(4)).as_matrix()
                                              @ R.from_rotvec(b["global_global_orient"]).as_matrix()).as_rotvec()
    _, _, up, ang = sc.gravity_frame([a, b])
    ua, ub = (np.median(sc.gravity_up_cam(d), 0) for d in (a, b))
    assert np.isclose(ang, 4, atol=0.3)
    assert np.allclose(up, (ua + ub) / np.linalg.norm(ua + ub), atol=1e-3)


def test_posed_identity_matches_rest_plus_root():
    m = fake_model_fixed()
    betas = np.zeros(10)
    _, J = sc.rest_shape(m, betas)
    rots = np.tile(np.eye(3), (T, 24, 1, 1))
    root = np.tile([1.0, 2.0, 3.0], (T, 1))
    v, pos = sc.posed(m, betas, rots, root)
    assert np.allclose(pos[:, 1] - pos[:, 0], J[1] - J[0])
    assert np.allclose(pos[:, 0], root)
    sub, _ = sc.posed(m, betas, rots, root, np.arange(5))
    assert np.allclose(sub, v[:, :5])


def test_floor_is_lowest_foot_and_scene_grounded(people):
    model = fake_model_fixed()
    glb, meta = sc.build_scene(model, people, K, 30.0, smoothing="off", track_ids=[[1], [2]],
                               observed=[np.ones(T, bool)] * 2, bboxes=[np.tile([0, 0, 10, 20.0], (T, 1))] * 2, percentile=0)
    # fake foot vertices: joints 7,8,10,11 dominate some vertices; floor is placed under the lowest of them, y = 0
    assert meta["cam_height_m"] > 0 and meta["floor_y"] == 0
    # every pelvis is above the floor; both fighters have the same foot floor offset in the shared frame
    for p in meta["persons"]:
        assert min(y for _, y, _ in p["pelvis_world_m"]) > 0
    # recompute min foot height from the GLB-driving motion: must be ~0
    G, Wr, _, _ = sc.gravity_frame(people)
    betas = [np.zeros(10)] * 2
    mot = [sc.person_motion(d, "off") for d in people]
    tw = np.array([0, meta["cam_height_m"], 0])
    foot = sc.foot_vertex_indices(model)
    lows = []
    for b, (r, rt) in zip(betas, mot):
        r2, p2 = sc.apply_world(r, rt, Wr, tw)
        lows.append(sc.posed(model, b, r2, p2, foot)[0][:, :, 1].min())
    assert np.isclose(min(lows), 0, atol=1e-6)


def test_scene_glb_structure(people):
    model = fake_model_fixed()
    glb, meta = sc.build_scene(model, people, K, 30.0, smoothing="off", colors=None, track_ids=[[1], [2, 3]],
                               observed=[np.ones(T, bool)] * 2, bboxes=[np.tile([0, 0, 10, 20.0], (T, 1))] * 2)
    js, binchunk = parse_glb(glb)
    names = [n["name"] for n in js["nodes"]]
    assert len(names) == len(set(names)) == 2 * 25
    assert {n.split("_")[0] for n in names} == {"fighter0", "fighter1"}
    assert len(js["animations"]) == 1 and js["animations"][0]["name"] == "scene"
    ch = js["animations"][0]["channels"]
    assert len(ch) == 2 * 25
    assert len({(c["target"]["node"], c["target"]["path"]) for c in ch}) == len(ch)  # no collisions
    assert len(js["skins"]) == 2 and len(js["meshes"]) == 2
    assert sorted(js["scenes"][0]["nodes"]) == sorted([24, 0, 25 + 24, 25])  # mesh + pelvis per fighter
    for a in js["accessors"]:
        bv = js["bufferViews"][a["bufferView"]]
        assert bv["byteOffset"] + bv["byteLength"] <= len(binchunk)
    for sk in js["skins"]:
        assert len(sk["joints"]) == 24
    # sampler / accessor indices in range, durations cover T frames
    for s in js["animations"][0]["samplers"]:
        assert s["input"] < len(js["accessors"]) and s["output"] < len(js["accessors"])
        assert js["accessors"][s["input"]]["count"] == T
    # meta
    assert meta["n_frames"] == T and meta["fps"] == 30.0 and meta["clip"] == "scene"
    assert meta["persons"][1]["track_ids"] == [2, 3] and meta["persons"][0]["node_prefix"] == "fighter0_"
    json.dumps(meta)
    # distinct colours: the two COLOR_0 accessors differ
    cols = [js["accessors"][m["primitives"][0]["attributes"]["COLOR_0"]] for m in js["meshes"]]
    assert cols[0]["bufferView"] != cols[1]["bufferView"]
    # pelvis root translation track of fighter0 equals meta pelvis_world
    root_ch = [c for c in ch if c["target"]["path"] == "translation" and c["target"]["node"] == 0][0]
    acc = js["accessors"][js["animations"][0]["samplers"][root_ch["sampler"]]["output"]]
    bv = js["bufferViews"][acc["bufferView"]]
    tr = np.frombuffer(binchunk, "<f4", count=T * 3, offset=bv["byteOffset"]).reshape(T, 3)
    assert np.allclose(tr, np.array(meta["persons"][0]["pelvis_world_m"]), atol=1e-3)
