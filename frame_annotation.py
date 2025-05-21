"""wave_peak_trough_annotator.py

A small interactive utility that lets you step through *selected* frames of a
wave‐tank video (as indicated in a CSV file) and click any number of **peaks**
(left‑mouse button) or **troughs** (right‑mouse button).  When you press the
**n** key (or SPACE) the tool advances to the next frame listed in the CSV,
preserving every click you made.  When you close the window (or when the last
frame is done) a new CSV file is written that contains one row per click with
frame number, x‑ and y‑pixel coordinate, and the label (peak | trough).

Typical usage (after installing the dependencies listed below)::

    python wave_peak_trough_annotator.py \
        --video  data/wave_movie_1.mp4 \
        --csv    data/peaks_and_trough_frames.csv \
        --out    data/annotated_clicks.csv

Dependencies
------------
* numpy
* pandas
* matplotlib
* opencv‑python (cv2)
* pillow (PIL)

You almost certainly have them already if you managed to run the earlier code.

The input CSV must have a column called ``frame`` with integer frame indices
(starting from 0, as given by OpenCV).  If you generated separate *peak* and
*trough* columns you can still keep them – we only care about the ``frame``
column and ignore the rest.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseEvent, KeyEvent
from PIL import Image


# -----------------------------------------------------------------------------
# Utility ‑‑ video loading ------------------------------------------------------
# -----------------------------------------------------------------------------

def load_camera_video(
    video_file: str | Path,
    convert_to_rgb: bool = True,
) -> np.ndarray:
    """Read *all* frames of *video_file* into an ndarray of shape (n, h, w, 3).

    Parameters
    ----------
    video_file : str or Path
        Path to the video.
    convert_to_rgb : bool, default=True
        If *True* convert each BGR frame yielded by OpenCV into RGB so that
        `matplotlib.pyplot.imshow` shows colours correctly.
    """
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {video_file}")

    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if convert_to_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return np.array(frames, dtype=np.uint8)


# -----------------------------------------------------------------------------
# Core annotation logic --------------------------------------------------------
# -----------------------------------------------------------------------------

def annotate_frames(
    video: np.ndarray,
    frame_numbers: List[int],
    out_csv: str | Path,
) -> None:
    """Interactive loop – present each frame in *frame_numbers* and record clicks.

    Parameters
    ----------
    video : ndarray  (n_frames, H, W, 3)
    frame_numbers : list[int]
        Frame indices to annotate.
    out_csv : str or Path
        Where to write the resulting annotations.  The file will have the
        columns ``frame, x, y, label``.
    """

    # Storage for *all* clicks
    clicks: List[Dict[str, int | str]] = []

    # Mutable frame index wrapped in list so we can modify from the nested fn
    idx = [0]

    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("Wave peak/trough annotator")

    text_instructions = (
        "LEFT‑click → peak,  RIGHT‑click → trough\n"
        "Press 'n' or SPACE to go to the next frame;  \n"
        "'q' to quit and save."
    )

    def draw_current_frame():
        ax.clear()
        frame_no = frame_numbers[idx[0]]
        ax.imshow(video[frame_no])
        ax.set_title(f"Frame {frame_no}  ({idx[0]+1}/{len(frame_numbers)})")
        # draw already existing clicks for this frame, if any
        for c in clicks:
            if c["frame"] == frame_no:
                colour = "red" if c["label"] == "peak" else "blue"
                ax.plot(c["x"], c["y"], marker="x", color=colour, ms=8, mew=2)
        # show the help text below the image (outside axes)
        fig.text(0.02, 0.02, text_instructions, fontsize=9, va="bottom")
        fig.canvas.draw_idle()

    def on_click(event: MouseEvent):
        if not event.inaxes:
            return
        if event.button not in (1, 3):
            return  # ignore middle scroll wheel etc.

        label = "peak" if event.button == 1 else "trough"
        click = {
            "frame": int(frame_numbers[idx[0]]),
            "x": int(event.xdata),
            "y": int(event.ydata),
            "label": label,
        }
        clicks.append(click)

        colour = "red" if label == "peak" else "blue"
        ax.plot(click["x"], click["y"], marker="x", color=colour, ms=8, mew=2)
        fig.canvas.draw_idle()

    def on_key(event: KeyEvent):
        if event.key in ("n", " "):
            idx[0] += 1
            if idx[0] >= len(frame_numbers):
                plt.close(fig)
            else:
                draw_current_frame()
        elif event.key == "q":
            plt.close(fig)

    cid_click = fig.canvas.mpl_connect("button_press_event", on_click)
    cid_key = fig.canvas.mpl_connect("key_press_event", on_key)

    draw_current_frame()
    plt.show()

    # After the GUI window is closed – dump to CSV
    if clicks:
        df = pd.DataFrame(clicks)
        df.to_csv(out_csv, index=False)
        print(f"\nSaved {len(df)} clicks → {out_csv}")
    else:
        print("No clicks recorded – nothing saved.")


# -----------------------------------------------------------------------------
# Command‑line entry‑point ------------------------------------------------------
# -----------------------------------------------------------------------------

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive peak/trough annotator")
    p.add_argument("--video", required=True, type=Path, help="Video file path")
    p.add_argument(
        "--csv",
        required=True,
        type=Path,
        help="CSV with a 'frame' column listing frames to annotate",
    )
    p.add_argument(
        "--out",
        default="annotated_clicks.csv",
        type=Path,
        help="Where to write the annotation CSV (default: annotated_clicks.csv)",
    )
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)

    # --- load resources -------------------------------------------------------
    print(f"Loading video → {args.video}")
    video = load_camera_video(args.video)
    n_frames = video.shape[0]
    print(f"  …loaded {n_frames} frames")

    print(f"Reading frame list → {args.csv}")
    frames_df = pd.read_csv(args.csv)
    if "frame" not in frames_df.columns:
        sys.exit("ERROR: CSV file must contain a 'frame' column with integers")
    frame_numbers = frames_df["frame"].astype(int).tolist()

    # Filter out‑of‑range indices early to avoid surprises later
    frame_numbers = [f for f in frame_numbers if 0 <= f < n_frames]
    if not frame_numbers:
        sys.exit("No valid frames to annotate after bounds‑checking!")

    print(f"Annotating {len(frame_numbers)} frames …")
    annotate_frames(video, frame_numbers, args.out)


if __name__ == "__main__":
    main()
