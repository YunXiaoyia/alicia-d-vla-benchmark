#!/usr/bin/env python3
"""Live preview to re-align the overhead `top` camera to the pose used when
`wipes-to-green-plate-01` was recorded (reference: ep66 frame 0, arm at home).

Overlays on the live frame: reference edges (green), reference landmarks (crosshairs) and an
optional 50% blend. Move/tilt the camera until the green edges sit on the live edges of the
robot base, table left edge, and the base LED. Put the arm at its HOME pose (as at episode start).

Keys: [b] cycle overlay (edges / blend / none)   [s] save snapshot   [q] quit
Usage:
  python align_top_camera.py --device /dev/video8
  python align_top_camera.py --image some_frame.png --snapshot out.png   # headless test
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

REF = Path(__file__).resolve().parents[1] / "assets" / "top_reference_train_ep66_f0.png"
# (x, y, label) landmarks measured on the reference frame (640x480)
LANDMARKS = [(292, 78, "base LED"), (323, 74, "base right"), (135, 275, "table L edge"), (138, 479, "table L bottom")]


def render(live, ref, mode):
    live = cv2.resize(live, (640, 480))
    out = live.copy()
    if mode == 0:
        edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), (5, 5), 0), 60, 140)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
        out[edges > 0] = (0, 255, 0)
    elif mode == 1:
        out = cv2.addWeighted(live, 0.5, ref, 0.5, 0)
    for x, y, name in LANDMARKS:
        cv2.drawMarker(out, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
        cv2.putText(out, name, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    cv2.putText(out, ["edges", "blend", "none"][mode] + "  [b]mode [s]save [q]quit", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="/dev/video8")
    ap.add_argument("--image", default=None, help="use a still image instead of the camera (testing)")
    ap.add_argument("--snapshot", default=None, help="save one composite and exit")
    args = ap.parse_args()
    ref = cv2.imread(str(REF))
    assert ref is not None, f"missing reference {REF}"

    if args.image:
        live = cv2.imread(args.image)
        out = render(live, ref, 0)
        cv2.imwrite(args.snapshot or "align_preview.png", out)
        return

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    assert cap.isOpened(), f"cannot open {args.device}"
    for _ in range(15):  # auto-exposure warmup
        cap.read()
    mode = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        out = render(frame, ref, mode)
        if args.snapshot:
            cv2.imwrite(args.snapshot, out)
            break
        cv2.imshow("align top camera", out)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("b"):
            mode = (mode + 1) % 3
        if k == ord("s"):
            cv2.imwrite("align_snapshot.png", out)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
