"""Unit tests for multiperson.py (no GPU / weights). Run: python -m pytest tests -q"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import multiperson as mp  # noqa: E402


def hist():
    """10 frames: id 1 big in all frames, id 2 smaller and missing frames 4-5, id 3 one-frame ghost."""
    h = []
    for f in range(10):
        fr = [{"id": 1, "bbx_xyxy": np.array([100 + f, 50, 300 + f, 450])}]
        if f not in (4, 5):
            fr.append({"id": 2, "bbx_xyxy": np.array([500, 60, 640, 400])})
        if f == 0:
            fr.append({"id": 3, "bbx_xyxy": np.array([0, 0, 10, 10])})
        h.append(fr)
    return h


def test_summary_sorted_by_area_and_rank():
    s = mp.summarize_tracks(hist(), 1920, 1080)
    assert [d["id"] for d in s] == [1, 2, 3]
    assert [d["rank"] for d in s] == [0, 1, 2]
    assert s[1]["n_frames"] == 8 and s[1]["n_gaps"] == 1 and s[0]["n_gaps"] == 0
    assert s[0]["bboxes"][0][0] == 0 and len(s[0]["bboxes"][0]) == 5


def test_build_person_bbx_interpolates_gap():
    bb, obs = mp.build_person_bbx(hist(), [2], 10)
    assert bb.shape == (10, 4) and obs.sum() == 8 and not obs[4] and not obs[5]
    assert np.allclose(bb[4], [500, 60, 640, 400])  # constant box -> interpolation is exact


def test_edges_held_and_merge_ids():
    h = hist()
    bb, obs = mp.build_person_bbx(h, [3, 2], 10)  # id 3 only frame 0, id 2 elsewhere
    assert obs[0] and obs[1] and not obs[4]
    assert bb.shape == (10, 4) and np.isfinite(bb).all()


def test_select():
    s = mp.summarize_tracks(hist(), 1920, 1080)
    assert mp.select_tracks(s, person_rank=[0, 1]) == [[1], [2]]
    assert mp.select_tracks(s, track_ids=[1, [2, 3]]) == [[1], [2, 3]]
    with pytest.raises(ValueError):
        mp.select_tracks(s, person_rank=[7])
    with pytest.raises(ValueError):
        mp.build_person_bbx(hist(), [99], 10)
