"""Unit tests for the GLB builder, smoothing and BVH conversion on tiny synthetic inputs (no GPU, no weights).

Run: python -m pytest tests -q    (needs numpy, scipy, pygltflib, pytest)
Set SMPL_PKL=/path/to/SMPL_NEUTRAL.pkl to additionally run a test on the real SMPL model.
"""
import json
import os
import pickle
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import smpl_bvh  # noqa: E402
import smpl_glb  # noqa: E402
from motion import smooth_motion  # noqa: E402

pygltflib = pytest.importorskip("pygltflib")

# Synthetic "SMPL": 24 joints, a tetra-ish mesh of 30 vertices, weights over >4 joints to exercise pruning.
REST24 = np.array([[0, 0, 0], [.09, -.09, 0], [-.09, -.09, 0], [0, .1, 0], [.09, -.5, 0], [-.09, -.5, 0],
                   [0, .25, 0], [.09, -.9, 0], [-.09, -.9, 0], [0, .4, 0], [.09, -.95, .1], [-.09, -.95, .1],
                   [0, .55, 0], [.05, .5, 0], [-.05, .5, 0], [0, .65, 0], [.17, .5, 0], [-.17, .5, 0],
                   [.42, .5, 0], [-.42, .5, 0], [.65, .5, 0], [-.65, .5, 0], [.72, .5, 0], [-.72, .5, 0]])


def fake_model(V=30, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.random((V, 24)) ** 4
    w /= w.sum(1, keepdims=True)
    f = np.stack([rng.permutation(V)[:3] for _ in range(40)]).astype(np.uint32)
    return {
        "v_template": rng.normal(size=(V, 3)) * 0.2 + [0, 0.1, 0],
        "shapedirs": rng.normal(size=(V, 3, 10)) * 0.01,
        "J_regressor": _jreg(V),
        "weights": w,
        "faces": f,
    }


def _jreg(V):
    # joint i = vertex i; fake_model_fixed puts REST24 into vertices 0..23
    j = np.zeros((24, V))
    j[np.arange(24), np.arange(24)] = 1
    return j


def fake_model_fixed():
    m = fake_model(V=30)
    m["v_template"][:24] = REST24
    m["shapedirs"][:24] = 0
    return m


def motion(T=3):
    rng = np.random.default_rng(1)
    go = rng.normal(size=(T, 3)) * 0.1
    bp = rng.normal(size=(T, 63)) * 0.1
    tr = np.cumsum(rng.normal(size=(T, 3)) * 0.01, axis=0) + [0.5, 0, 0.5]
    return go, bp, tr, np.zeros(10)


def parse_glb(b):
    magic, ver, length = struct.unpack("<4sII", b[:12])
    assert magic == b"glTF" and ver == 2 and length == len(b)
    jl, jt = struct.unpack("<I4s", b[12:20])
    assert jt == b"JSON" and jl % 4 == 0
    js = json.loads(b[20:20 + jl])
    bl, bt = struct.unpack("<I4s", b[20 + jl:28 + jl])
    assert bt == b"BIN\0" and bl % 4 == 0 and 28 + jl + bl == length
    return js, b[28 + jl:]


@pytest.mark.parametrize("T", [1, 2, 3])
def test_glb_structure_and_roundtrip(T, tmp_path):
    model = fake_model_fixed()
    go, bp, tr, betas = motion(T)
    glb = smpl_glb.smpl_to_glb(model, go, bp, tr, betas, fps=15.0)
    js, binchunk = parse_glb(glb)
    V, J = 30, 24

    # accessors / bufferViews: counts, alignment, bounds
    for bv in js["bufferViews"]:
        assert bv["byteOffset"] % 4 == 0
        assert bv["byteOffset"] + bv["byteLength"] <= len(binchunk)
    sizes = {5126: 4, 5123: 2, 5125: 4}
    ncomp = {"SCALAR": 1, "VEC3": 3, "VEC4": 4, "MAT4": 16}
    for a in js["accessors"]:
        bv = js["bufferViews"][a["bufferView"]]
        assert a["count"] * sizes[a["componentType"]] * ncomp[a["type"]] == bv["byteLength"]
    prim = js["meshes"][0]["primitives"][0]
    acc = js["accessors"]
    assert acc[prim["attributes"]["POSITION"]]["count"] == V
    assert acc[prim["attributes"]["JOINTS_0"]]["count"] == V
    assert acc[prim["indices"]]["count"] == 40 * 3
    assert len(js["skins"][0]["joints"]) == J
    assert acc[js["skins"][0]["inverseBindMatrices"]]["count"] == J
    assert js["materials"][0]["doubleSided"] is False
    assert js["materials"][0]["pbrMetallicRoughness"]["roughnessFactor"] == 0.6

    def read(i):
        a = acc[i]
        bv = js["bufferViews"][a["bufferView"]]
        dt = {5126: "<f4", 5123: "<u2", 5125: "<u4"}[a["componentType"]]
        x = np.frombuffer(binchunk, dt, count=a["count"] * ncomp[a["type"]], offset=bv["byteOffset"])
        return x.reshape(a["count"], -1)

    w = read(prim["attributes"]["WEIGHTS_0"])
    j = read(prim["attributes"]["JOINTS_0"])
    assert np.allclose(w.sum(1), 1, atol=1e-5) and j.max() < J and (w >= 0).all()
    nrm = read(prim["attributes"]["NORMAL"])
    assert np.allclose(np.linalg.norm(nrm, axis=1), 1, atol=1e-4)
    assert read(prim["indices"]).max() < V

    # animation: one rotation track per joint + root translation, unit quaternions, increasing times
    anim = js["animations"][0]
    assert len(anim["channels"]) == J + 1
    for ch in anim["channels"]:
        s = anim["samplers"][ch["sampler"]]
        assert acc[s["input"]]["count"] == T and acc[s["output"]]["count"] == T
        if ch["target"]["path"] == "rotation":
            q = read(s["output"])
            assert np.allclose(np.linalg.norm(q, axis=1), 1, atol=1e-5)
    t = read(anim["samplers"][0]["input"]).ravel()
    assert np.allclose(t, np.arange(T) / 15.0)

    # root: feet on ground, start at XZ origin
    root = read(anim["samplers"][0]["output"])
    assert np.allclose(root[0, [0, 2]], 0, atol=1e-5)

    # pygltflib round trip
    p = tmp_path / "t.glb"
    p.write_bytes(glb)
    g = pygltflib.GLTF2().load(str(p))
    assert len(g.nodes) == J + 1 and len(g.animations[0].channels) == J + 1
    g.save(str(tmp_path / "t2.glb"))
    g2 = pygltflib.GLTF2().load(str(tmp_path / "t2.glb"))
    assert len(g2.accessors) == len(g.accessors) == len(acc)
    assert g2.skins[0].joints == list(range(J))


def test_ibm_is_negative_rest_translation():
    model = fake_model_fixed()
    glb = smpl_glb.smpl_to_glb(model, *motion(2), fps=30.0)
    js, b = parse_glb(glb)
    a = js["accessors"][js["skins"][0]["inverseBindMatrices"]]
    bv = js["bufferViews"][a["bufferView"]]
    ibm = np.frombuffer(b, "<f4", count=24 * 16, offset=bv["byteOffset"]).reshape(24, 4, 4)
    assert np.allclose(ibm[:, 3, :3], -REST24, atol=1e-5)  # column-major translation column
    assert np.allclose(ibm[:, :3, :3], np.eye(3))


def test_top4_prune():
    w = np.array([[.4, .3, .2, .06, .04]])
    j, ww = smpl_glb.top4(w)
    assert list(j[0]) == [0, 1, 2, 3] and abs(ww.sum() - 1) < 1e-6


def test_pkl_loader_without_chumpy(tmp_path):
    import types
    import scipy.sparse as sp

    m = fake_model_fixed()
    mod = types.ModuleType("chumpy.ch")  # temporary fake chumpy so we can pickle a "Ch" object
    mod.Ch = type("Ch", (), {"__module__": "chumpy.ch"})
    sys.modules["chumpy"], sys.modules["chumpy.ch"] = types.ModuleType("chumpy"), mod
    try:
        ch = mod.Ch()
        ch.x = m["shapedirs"]
        d = {"v_template": m["v_template"], "shapedirs": ch, "J_regressor": sp.csc_matrix(m["J_regressor"]),
             "weights": m["weights"], "f": m["faces"]}
        blob = pickle.dumps(d)
    finally:
        sys.modules.pop("chumpy", None)
        sys.modules.pop("chumpy.ch", None)
    p = tmp_path / "x.pkl"
    p.write_bytes(blob)
    loaded = smpl_glb.load_smpl_pkl(str(p))  # chumpy now unimportable -> stub path
    assert loaded["J_regressor"].shape == (24, 30) and np.allclose(loaded["shapedirs"], m["shapedirs"])


def test_smoothing_properties():
    T = 40
    rng = np.random.default_rng(3)
    go = np.cumsum(rng.normal(size=(T, 3)) * 0.05, 0)
    bp = np.cumsum(rng.normal(size=(T, 63)) * 0.05, 0)
    tr = np.cumsum(rng.normal(size=(T, 3)) * 0.02, 0)
    g0, b0, t0 = smooth_motion(go, bp, tr, "off")
    assert np.array_equal(b0, bp)
    noisy = bp + rng.normal(size=bp.shape) * 0.1
    for lvl in ("light", "medium", "strong"):
        g, b, t = smooth_motion(go, noisy, tr, lvl)
        assert b.shape == bp.shape and g.shape == go.shape and t.shape == tr.shape
        assert np.abs(np.diff(b, axis=0)).mean() < np.abs(np.diff(noisy, axis=0)).mean()
    # sign continuity: a rotation crossing pi must not produce a spike
    aa = np.zeros((T, 3))
    aa[:, 0] = np.linspace(3.0, 3.3, T)  # passes through angle pi (axis-angle flips representation)
    g, _, _ = smooth_motion(aa, np.zeros((T, 63)), np.zeros((T, 3)), "medium")
    from scipy.spatial.transform import Rotation as R
    ang = (R.from_rotvec(g[1:]) * R.from_rotvec(g[:-1]).inv()).magnitude()
    assert ang.max() < 0.05
    # short clips do not crash
    smooth_motion(go[:2], bp[:2], tr[:2], "strong")


def test_bvh_still_valid():
    smplx = np.zeros(0)
    np.savez("/tmp/_fake_smplx.npz", v_template=REST24[:22], shapedirs=np.zeros((22, 3, 10)), J_regressor=np.eye(22))
    go, bp, tr, betas = motion(3)
    out = smpl_bvh.smpl_to_bvh(go, bp, tr, betas, "/tmp/_fake_smplx.npz", 30.0)
    os.remove("/tmp/_fake_smplx.npz")
    assert "Frames: 3" in out and len(out.strip().splitlines()[-1].split()) == 6 + 21 * 3


@pytest.mark.skipif(not os.environ.get("SMPL_PKL"), reason="SMPL_PKL not set")
def test_real_smpl_model():
    model = smpl_glb.load_smpl_pkl(os.environ["SMPL_PKL"])
    assert model["v_template"].shape == (6890, 3) and model["J_regressor"].shape == (24, 6890)
    glb = smpl_glb.smpl_to_glb(model, *motion(3), fps=30.0)
    js, _ = parse_glb(glb)
    assert js["accessors"][js["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]["count"] == 6890


def onehot_model():
    """Vertex i is skinned 100% to joint i and placed at joint i (i < 24); 6 extra vertices on the pelvis."""
    m = fake_model_fixed()
    w = np.zeros((30, 24))
    w[np.arange(24), np.arange(24)] = 1
    w[24:, 0] = 1
    m["v_template"][24:] = [0.0, -0.2, 0.0]
    m["weights"] = w
    return m


def test_clothing_regions_by_gender():
    m = onehot_model()
    J = REST24
    for gender, has_top in (("male", False), ("female", True), ("neutral", True)):
        _, lab = smpl_glb.clothing_colors(m["v_template"], m["faces"], J, m["weights"], gender, feather=0)
        assert lab[1] == 1 and lab[2] == 1       # hips above the 60% hip->knee hem: shorts
        assert lab[0] == 2                        # pelvis at top edge of shorts: waistband
        assert lab[4] == 0 and lab[5] == 0        # knees below the hem: skin
        assert lab[16] == 0 and lab[18] == 0      # shoulders / arms stay skin
        assert lab[3] == 0                        # spine1 below chest cutoff
        assert (lab[[6, 9]] == 3).all() == has_top and (lab[[6, 9]] == 0).all() == (not has_top)


def test_color0_in_glb(tmp_path):
    m = onehot_model()
    glb = smpl_glb.smpl_to_glb(m, *motion(2), fps=30.0, gender="female")
    js, b = parse_glb(glb)
    prim = js["meshes"][0]["primitives"][0]
    a = js["accessors"][prim["attributes"]["COLOR_0"]]
    assert a["count"] == 30 and a["type"] == "VEC4" and a["componentType"] == 5126
    bv = js["bufferViews"][a["bufferView"]]
    assert bv["byteOffset"] % 4 == 0 and bv["byteLength"] == 30 * 16
    col = np.frombuffer(b, "<f4", count=30 * 4, offset=bv["byteOffset"]).reshape(30, 4)
    assert (col >= 0).all() and (col <= 1).all() and np.allclose(col[:, 3], 1)
    skin = np.array(smpl_glb.SKIN)
    is_skin = np.isclose(col[:, :3], skin, atol=0.2).all(1)
    assert is_skin.any() and (~is_skin).any()          # both skin and clothing present
    assert js["materials"][0]["pbrMetallicRoughness"]["baseColorFactor"] == [1.0, 1.0, 1.0, 1.0]
    p = tmp_path / "c.glb"
    p.write_bytes(glb)
    g = pygltflib.GLTF2().load(str(p))
    assert g.meshes[0].primitives[0].attributes.COLOR_0 == prim["attributes"]["COLOR_0"]


def test_feather_keeps_range_and_softens():
    m = onehot_model()
    hard, _ = smpl_glb.clothing_colors(m["v_template"], m["faces"], REST24, m["weights"], "female", feather=0)
    soft, _ = smpl_glb.clothing_colors(m["v_template"], m["faces"], REST24, m["weights"], "female", feather=1)
    assert soft.min() >= 0 and soft.max() <= 1 and not np.allclose(hard, soft)


@pytest.mark.skipif(not os.environ.get("SMPL_PKL"), reason="SMPL_PKL not set")
def test_real_model_clothing_counts():
    m = smpl_glb.load_smpl_pkl(os.environ["SMPL_PKL"])
    J = m["J_regressor"] @ m["v_template"]
    _, lab = smpl_glb.clothing_colors(m["v_template"], m["faces"], J, m["weights"], "neutral", feather=0)
    n = np.bincount(lab, minlength=4)
    assert n[1] > 300 and n[2] > 20 and n[3] > 300 and n[0] > 4000
    _, lab_m = smpl_glb.clothing_colors(m["v_template"], m["faces"], J, m["weights"], "male", feather=0)
    assert (lab_m == 3).sum() == 0
