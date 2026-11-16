"""Multi-person helpers: per-track summaries, track selection / merging, per-person bbox sequences.

Pure numpy (no GPU, no GVHMR import) so it is unit-testable. `track_history` is the output of
GVHMR's Tracker.track(): per frame a list of {"id": int, "bbx_xyxy": array(4)}.
"""
import numpy as np


def collect_tracks(track_history):
    """-> {id: (frame_ids int array, bbx (N,4) float array)} (one detection per frame per id)."""
    ids, boxes = {}, {}
    for f, frame in enumerate(track_history):
        for det in frame:
            ids.setdefault(det["id"], []).append(f)
            boxes.setdefault(det["id"], []).append(np.asarray(det["bbx_xyxy"], dtype=np.float64))
    return {k: (np.asarray(ids[k]), np.stack(boxes[k])) for k in ids}


def summarize_tracks(track_history, width, height, sample_every=3, min_frames=1):
    """Tracks sorted like GVHMR's sort_track_length: by sum(bbox_area / frame_area), descending."""
    out = []
    for tid, (fr, bb) in collect_tracks(track_history).items():
        if len(fr) < min_frames:
            continue
        wh = bb[:, 2:] - bb[:, :2]
        area = float((wh[:, 0] * wh[:, 1] / width / height).sum())
        gaps = int(np.sum(np.diff(fr) > 1))
        out.append({
            "id": int(tid), "first_frame": int(fr[0]), "last_frame": int(fr[-1]), "n_frames": int(len(fr)),
            "n_gaps": gaps, "area_score": round(area, 4),
            "mean_bbox": [round(float(x), 1) for x in bb.mean(0)],
            "bboxes": [[int(f)] + [round(float(x), 1) for x in b] for f, b in zip(fr[::sample_every], bb[::sample_every])],
        })
    out.sort(key=lambda d: d["area_score"], reverse=True)
    for rank, d in enumerate(out):
        d["rank"] = rank
    return out


def _moving_average(x, window=5):
    pad = window // 2
    xp = np.concatenate([np.repeat(x[:1], pad, 0), x, np.repeat(x[-1:], pad, 0)], axis=0)
    k = np.ones(window) / window
    return np.stack([np.convolve(xp[:, c], k, mode="valid") for c in range(x.shape[1])], axis=1)


def build_person_bbx(track_history, ids, n_frames):
    """One bbox per frame for one person made of one or more track ids (union; first id wins when several
    are present in the same frame). Gaps are linearly interpolated, leading/trailing frames hold the nearest
    box, then two moving-average passes (window 5), same recipe as GVHMR Tracker.get_one_track.
    Returns (bbx (n_frames,4), observed (n_frames,) bool)."""
    tracks = collect_tracks(track_history)
    per_frame = {}
    for tid in reversed(list(ids)):  # earlier ids overwrite later ones
        if tid not in tracks:
            raise ValueError(f"track id {tid} not found (have {sorted(tracks)})")
        fr, bb = tracks[tid]
        for f, b in zip(fr, bb):
            per_frame[int(f)] = b
    if not per_frame:
        raise ValueError("empty track selection")
    fr = np.array(sorted(per_frame))
    bb = np.stack([per_frame[int(f)] for f in fr])
    allf = np.arange(n_frames)
    full = np.stack([np.interp(allf, fr, bb[:, c]) for c in range(4)], axis=1)  # edge-constant outside range
    observed = np.zeros(n_frames, dtype=bool)
    observed[fr[fr < n_frames]] = True
    full = _moving_average(_moving_average(full))
    return full, observed


def select_tracks(summaries, track_ids=None, person_rank=None):
    """Resolve the request to a list of id-lists (one per person). track_ids entries may be an int or a list
    of ints (merged into one person); person_rank indexes the area-sorted summaries."""
    if track_ids is not None:
        out = [[int(t)] if isinstance(t, (int, np.integer)) else [int(x) for x in t] for t in track_ids]
    elif person_rank is not None:
        out = []
        for r in person_rank:
            if not 0 <= int(r) < len(summaries):
                raise ValueError(f"person_rank {r} out of range (found {len(summaries)} tracks)")
            out.append([summaries[int(r)]["id"]])
    else:
        raise ValueError("track_ids or person_rank required")
    if not out or any(not g for g in out):
        raise ValueError("empty person selection")
    return out
