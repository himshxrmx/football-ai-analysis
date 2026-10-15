"""Download the demonstration clip into data/samples/.

The clip (08fd33_0.mp4, 30 s, 1920 x 1080, about 19 MB) is one of the Bundesliga
sample videos listed in the Roboflow "sports" repository
(https://github.com/roboflow/sports/blob/main/examples/soccer/setup.sh).

Usage:
    python scripts/download_sample.py
"""
from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data" / "samples" / "08fd33_0.mp4"
URL = "https://drive.usercontent.google.com/download?id=1OG8K6wqUw9t7lp9ms1M48DxRhwTYciK-&export=download&confirm=t"
MANUAL = ("Download a short football clip yourself and put it in data/samples/, or upload a video in the app. "
          "The original clip is listed in https://github.com/roboflow/sports (examples/soccer/setup.sh).")


def main() -> int:
    if TARGET.exists() and TARGET.stat().st_size > 1_000_000:
        print(f"Sample video already present: {TARGET}")
        return 0
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading the sample video (about 19 MB) to {TARGET} ...")
    tmp = TARGET.with_suffix(".part")
    try:
        with urllib.request.urlopen(URL, timeout=120) as response, open(tmp, "wb") as fh:
            if "video" not in response.headers.get("Content-Type", ""):
                print("The download link did not return a video. " + MANUAL)
                return 1
            while chunk := response.read(1 << 20):
                fh.write(chunk)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"The sample video could not be downloaded ({exc}). " + MANUAL)
        return 1
    finally:
        if tmp.exists() and not TARGET.exists() and tmp.stat().st_size < 1_000_000:
            tmp.unlink()
    tmp.replace(TARGET)
    print(f"Done ({TARGET.stat().st_size / 1e6:.1f} MB).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
