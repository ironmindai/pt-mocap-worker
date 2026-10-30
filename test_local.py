#!/usr/bin/env python3
"""Invoke the mocap worker contract against a deployed RunPod serverless endpoint.

  RUNPOD_API_KEY=... ENDPOINT_ID=... python3 test_local.py https://example.com/clip.mp4 [-o out.bvh] [--stride 2]
"""
import argparse
import base64
import json
import os
import sys
import time
import urllib.request

API = "https://api.runpod.ai/v2"


def call(method, url, key, body=None):
    req = urllib.request.Request(url, method=method, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        data=json.dumps(body).encode() if body is not None else None)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_url")
    ap.add_argument("-o", "--out", default="out.bvh")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--static-cam", action="store_true")
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()
    key, eid = os.environ.get("RUNPOD_API_KEY"), os.environ.get("ENDPOINT_ID")
    if not key or not eid:
        sys.exit("set RUNPOD_API_KEY and ENDPOINT_ID")

    inp = {"video_url": a.video_url, "frame_stride": a.stride, "static_cam": a.static_cam}
    job = call("POST", f"{API}/{eid}/run", key, {"input": inp})
    jid = job["id"]
    print("submitted", jid)
    t0, last = time.time(), None
    while time.time() - t0 < a.timeout:
        st = call("GET", f"{API}/{eid}/status/{jid}", key)
        if st["status"] != last:
            print(f"[{int(time.time() - t0):>4}s] {st['status']}")
            last = st["status"]
        if st["status"] == "COMPLETED":
            out = st["output"]
            if "error" in out:
                sys.exit("worker error: " + out["error"])
            with open(a.out, "wb") as f:
                f.write(base64.b64decode(out["bvh_b64"]))
            print(f"wrote {a.out}: {out['n_frames']} frames @ {out['fps']} fps ({out['duration_s']} s), "
                  f"worker {out.get('elapsed_s')} s")
            return
        if st["status"] in ("FAILED", "CANCELLED", "TIMED_OUT"):
            sys.exit(f"job {st['status']}: {st.get('error')}")
        time.sleep(5)
    sys.exit("client timeout")


if __name__ == "__main__":
    main()
