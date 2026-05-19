from __future__ import annotations

from pathlib import Path

import cv2


def main() -> None:
    """
    Generate 6-inch printable ArUco markers (300 DPI) for IDs used in anchors.
    Output directory: draft2/aruco_markers_6in
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    dpi = 300
    marker_inches = 6.0
    marker_px = int(marker_inches * dpi)  # 1800
    border_px = 220

    out_dir = Path(__file__).resolve().parent / "aruco_markers_6in"
    out_dir.mkdir(parents=True, exist_ok=True)

    for marker_id in (0, 1, 2):
        marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_px)
        canvas = cv2.copyMakeBorder(
            marker,
            border_px,
            border_px,
            border_px,
            border_px,
            cv2.BORDER_CONSTANT,
            value=255,
        )
        out_path = out_dir / f"aruco_id_{marker_id}_6in_300dpi.png"
        cv2.imwrite(str(out_path), canvas)
        print(f"Generated: {out_path}")


if __name__ == "__main__":
    main()
