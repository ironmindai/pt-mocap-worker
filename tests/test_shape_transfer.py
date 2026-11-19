"""SMPL -> SMPL-X shape transfer. Needs the real models: SMPLX_NPZ=/path/SMPLX_NEUTRAL.npz SMPL_PKL=/path/SMPL_NEUTRAL.pkl."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import smpl_glb as sg  # noqa: E402
import smplx_np as sx  # noqa: E402

SMPLX, SMPL = os.environ.get("SMPLX_NPZ"), os.environ.get("SMPL_PKL")
pytestmark = pytest.mark.skipif(not (SMPLX and SMPL), reason="needs SMPLX_NPZ and SMPL_PKL")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_adapted_skeleton_matches_smpl(seed):
    mx, ms = sx.load_smplx_npz(SMPLX), sg.load_smpl_pkl(SMPL)
    betas = np.random.default_rng(seed).normal(size=10) * 0.8
    m2, bx, info = sx.adapt_model_to_smpl(mx, ms, betas)
    _, J = sx.rest_shape(m2, bx)
    Js = sx._smpl_rest_joints(ms, betas)
    ph_x, ph_s = np.linalg.norm(J[15] - J[0]), np.linalg.norm(Js[15] - Js[0])
    assert abs(ph_x - ph_s) < 0.01                                   # pelvis-to-head within 1 cm
    for a, b in ((16, 18), (18, 20), (1, 4), (4, 7)):                # arm / leg bones within 1.5 cm
        assert abs(np.linalg.norm(J[a] - J[b]) - np.linalg.norm(Js[a] - Js[b])) < 0.015
    assert info["rms_joint_delta_err_m"] < 0.01
