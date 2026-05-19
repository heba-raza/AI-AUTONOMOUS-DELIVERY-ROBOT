from __future__ import annotations

import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class DetectionResult:
    blocked: bool
    status: str
    red_ratio: float
    green_ratio: float
    white_ratio: float
    marker_ids: list[int]
    roi_rect: tuple[int, int, int, int]


PROFILE_SETTINGS = {
    "Home": {
        "sat_gain": 1.2,
        "use_white_block": True,
        "white_block_thresh": 0.12,
        # Wider floor-green, stricter red (fewer false reds from WB / noise).
        "green_hsv_lo": (28, 38, 40),
        "green_hsv_hi": (100, 255, 255),
        "red1_hsv_lo": (0, 95, 65),
        "red1_hsv_hi": (11, 255, 255),
        "red2_hsv_lo": (170, 95, 65),
        "red2_hsv_hi": (179, 255, 255),
        "red_green_margin": 0.045,
        "min_red_ratio_for_block": 0.11,
        "green_clears_white_ratio": 0.14,
    },
    "University": {
        "sat_gain": 1.7,
        # Decision: block iff red% > green%; if no red and no green pixels in ROI, pass.
        "simple_red_green_block": True,
        "use_white_block": False,
        "green_hsv_lo": (26, 32, 35),
        "green_hsv_hi": (102, 255, 255),
        "red1_hsv_lo": (0, 100, 70),
        "red1_hsv_hi": (10, 255, 255),
        "red2_hsv_lo": (171, 100, 70),
        "red2_hsv_hi": (179, 255, 255),
    },
}


def open_mjpeg_stream(url: str, timeout_s: float = 12.0):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=timeout_s)


def fetch_snapshot_frame(url: str, timeout_s: float = 2.0):
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0", "Cache-Control": "no-cache"}
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = resp.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


class CameraStreamClient:
    def __init__(self, camera_url: str, mode: str = "snapshot", snapshot_url: str = "") -> None:
        self.camera_url = camera_url
        self.mode = mode
        self.snapshot_url = snapshot_url.strip() or camera_url.replace("/stream", "/jpg")
        self.cap = None
        self.stream = None
        self.byte_buf = b""
        self.use_snapshot = mode == "snapshot"
        self.use_mjpeg_fallback = mode == "mjpeg"
        self.snapshot_fail_count = 0
        self.mjpeg_connect_fail_count = 0
        self.eoi_fail_count = 0
        self.last_ok_frame_ts = time.time()
        self.last_warn_ts = 0.0

    def start(self) -> None:
        if self.mode in ("auto", "opencv"):
            self.cap = cv2.VideoCapture(self.camera_url)
            if self.cap.isOpened():
                ok, _ = self.cap.read()
                if ok:
                    return
                if self.mode == "auto":
                    self.cap.release()
                    self.cap = None
                    self.use_mjpeg_fallback = True
                else:
                    raise RuntimeError("VideoCapture opened but failed to read first frame.")
            elif self.mode == "opencv":
                raise RuntimeError(f"Cannot open stream with VideoCapture: {self.camera_url}")
            else:
                self.use_mjpeg_fallback = True

        if self.use_mjpeg_fallback:
            try:
                self.stream = open_mjpeg_stream(self.camera_url, timeout_s=12.0)
            except (TimeoutError, socket.timeout, OSError, urllib.error.URLError):
                self.use_mjpeg_fallback = False
                self.use_snapshot = True

    def _connect_mjpeg(self) -> bool:
        try:
            self.stream = open_mjpeg_stream(self.camera_url, timeout_s=12.0)
            self.byte_buf = b""
            self.mjpeg_connect_fail_count = 0
            return True
        except (TimeoutError, socket.timeout, OSError, urllib.error.URLError):
            self.mjpeg_connect_fail_count += 1
            if self.mjpeg_connect_fail_count >= 5:
                self.use_mjpeg_fallback = False
                self.use_snapshot = True
            return False

    def read_frame(self):
        while True:
            if self.use_snapshot:
                try:
                    frame = fetch_snapshot_frame(self.snapshot_url, timeout_s=2.0)
                    self.snapshot_fail_count = 0
                except (TimeoutError, socket.timeout, OSError, urllib.error.URLError):
                    self.snapshot_fail_count += 1
                    if self.snapshot_fail_count >= 5:
                        self.use_snapshot = False
                        self.use_mjpeg_fallback = True
                    continue
                if frame is not None:
                    self.last_ok_frame_ts = time.time()
                    return frame
                continue

            if self.cap is not None:
                ok, frame = self.cap.read()
                if ok:
                    self.last_ok_frame_ts = time.time()
                    return frame
                continue

            if self.stream is None and not self._connect_mjpeg():
                time.sleep(0.3)
                continue

            try:
                chunk = self.stream.read(4096)
            except (TimeoutError, socket.timeout, OSError):
                try:
                    self.stream.close()
                except Exception:
                    pass
                self.stream = None
                self.byte_buf = b""
                continue

            if not chunk:
                try:
                    self.stream.close()
                except Exception:
                    pass
                self.stream = None
                self.byte_buf = b""
                continue

            self.byte_buf += chunk
            a = self.byte_buf.find(b"\xff\xd8")
            if a == -1:
                if len(self.byte_buf) > 32768:
                    self.byte_buf = self.byte_buf[-4096:]
                if time.time() - self.last_ok_frame_ts > 8.0:
                    try:
                        self.stream.close()
                    except Exception:
                        pass
                    self.stream = None
                    self.byte_buf = b""
                continue
            if a > 0:
                self.byte_buf = self.byte_buf[a:]

            b = self.byte_buf.find(b"\xff\xd9", 2)
            if b == -1:
                if time.time() - self.last_ok_frame_ts > 8.0:
                    self.eoi_fail_count += 1
                    try:
                        self.stream.close()
                    except Exception:
                        pass
                    self.stream = None
                    self.byte_buf = b""
                    if self.eoi_fail_count >= 6:
                        self.use_mjpeg_fallback = False
                        self.use_snapshot = True
                continue

            latest_start = 0
            search_pos = 2
            while True:
                na = self.byte_buf.find(b"\xff\xd8", search_pos)
                if na == -1 or na > b:
                    break
                latest_start = na
                search_pos = na + 2

            jpg = self.byte_buf[latest_start : b + 2]
            self.byte_buf = self.byte_buf[b + 2 :]
            arr = np.frombuffer(jpg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            self.eoi_fail_count = 0
            self.last_ok_frame_ts = time.time()
            return frame

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
        if self.stream is not None:
            self.stream.close()


def analyze_frame(frame: np.ndarray, profile: str = "Home") -> tuple[np.ndarray, DetectionResult]:
    cfg = PROFILE_SETTINGS.get(profile, PROFILE_SETTINGS["Home"])

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    frame_hsv[:, :, 1] = np.clip(frame_hsv[:, :, 1] * cfg["sat_gain"], 0, 255)
    frame_enhanced = cv2.cvtColor(frame_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    h, w = frame_enhanced.shape[:2]
    y0, y1 = int(0.05 * h), int(0.38 * h)
    x0, x1 = int(0.18 * w), int(0.82 * w)
    roi = frame_enhanced[y0:y1, x0:x1]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    r1_lo = cfg.get("red1_hsv_lo", (0, 80, 60))
    r1_hi = cfg.get("red1_hsv_hi", (12, 255, 255))
    r2_lo = cfg.get("red2_hsv_lo", (165, 80, 60))
    r2_hi = cfg.get("red2_hsv_hi", (179, 255, 255))
    g_lo = cfg.get("green_hsv_lo", (35, 60, 60))
    g_hi = cfg.get("green_hsv_hi", (95, 255, 255))
    red1 = cv2.inRange(hsv, r1_lo, r1_hi)
    red2 = cv2.inRange(hsv, r2_lo, r2_hi)
    red_mask = cv2.bitwise_or(red1, red2)
    green_mask = cv2.inRange(hsv, g_lo, g_hi)
    white_mask = cv2.inRange(hsv, (0, 0, 180), (179, 55, 255))

    total = float(max(1, roi.shape[0] * roi.shape[1]))
    red_ratio = float(np.count_nonzero(red_mask)) / total
    green_ratio = float(np.count_nonzero(green_mask)) / total
    white_ratio = float(np.count_nonzero(white_mask)) / total

    if cfg.get("simple_red_green_block"):
        red_px = int(np.count_nonzero(red_mask))
        green_px = int(np.count_nonzero(green_mask))
        if red_px == 0 and green_px == 0:
            blocked_red = False
        else:
            blocked_red = red_ratio > green_ratio
        blocked_white = False
    else:
        margin = float(cfg.get("red_green_margin", 0.04))
        min_red = float(cfg.get("min_red_ratio_for_block", 0.10))
        blocked_red = (red_ratio > green_ratio + margin) and (red_ratio >= min_red)

        blocked_white = False
        if cfg["use_white_block"] and white_ratio > float(cfg["white_block_thresh"]):
            green_clear = float(cfg.get("green_clears_white_ratio", 0.12))
            blocked_white = green_ratio < green_clear

    blocked = blocked_red or blocked_white

    gray = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, _ = aruco_detector.detectMarkers(gray)
    marker_ids: list[int] = []
    if ids is not None:
        marker_ids = [int(v) for v in ids.flatten().tolist()]
        cv2.aruco.drawDetectedMarkers(frame_enhanced, corners, ids)

    result = DetectionResult(
        blocked=blocked,
        status="BLOCK" if blocked else "GREEN",
        red_ratio=red_ratio,
        green_ratio=green_ratio,
        white_ratio=white_ratio,
        marker_ids=marker_ids,
        roi_rect=(x0, y0, x1, y1),
    )
    return frame_enhanced, result


def overlay_detection(frame: np.ndarray, result: DetectionResult, profile: str) -> None:
    x0, y0, x1, y1 = result.roi_rect
    cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 200, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 0, 255) if result.blocked else (0, 255, 0)
    cv2.putText(frame, f"Detected: {result.status}", (20, 30), font, 0.8, color, 2)
    cv2.putText(
        frame,
        f"profile={profile} red={result.red_ratio:.3f} green={result.green_ratio:.3f} white={result.white_ratio:.3f}",
        (20, 60),
        font,
        0.65,
        (255, 255, 255),
        2,
    )
    cv2.putText(frame, f"markers={result.marker_ids}", (20, 90), font, 0.65, (255, 255, 255), 2)
