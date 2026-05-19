from __future__ import annotations

import argparse
import json
import tkinter as tk
from tkinter import messagebox
import time
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2

from cv_runtime_core import CameraStreamClient, DetectionResult, analyze_frame, overlay_detection


GRID_COLS = 5
GRID_ROWS = 10
CELL_PX = 50
MARGIN = 20
CANVAS_W = GRID_COLS * CELL_PX + 2 * MARGIN
CANVAS_H = GRID_ROWS * CELL_PX + 2 * MARGIN

WALLS = {
    (3, 0),
    (0, 1), (1, 1),
    (1, 2), (4, 2),
    (4, 3),
    (2, 4),
    (1, 5),
    (1, 6), (3, 6),
    (3, 7),
    (0, 8), (2, 8),
}
ANCHORS = {(0, 0): 0, (2, 5): 1, (4, 9): 2}
ANCHOR_BY_ID = {v: k for k, v in ANCHORS.items()}
GOALS = {(4, 9)}

HEADING_N, HEADING_E, HEADING_S, HEADING_W = 0, 1, 2, 3
HEADING_NAMES = {HEADING_N: "N", HEADING_E: "E", HEADING_S: "S", HEADING_W: "W"}
HEADING_DELTA = {
    HEADING_N: (0, -1),
    HEADING_E: (1, 0),
    HEADING_S: (0, 1),
    HEADING_W: (-1, 0),
}

ACTION_FORWARD = "FORWARD"
ACTION_LEFT = "TURN_LEFT"
ACTION_RIGHT = "TURN_RIGHT"
ACTION_BACK = "BACKWARD"
CONFIDENCE_LOW_THRESHOLD = 0.55
CONFIDENCE_DECAY_PER_STEP = 0.05
CONFIDENCE_RESET_ON_ANCHOR = 1.0


@dataclass
class Pose:
    cell: tuple[int, int]
    heading: int


def http_get_text(url: str, timeout: float = 10.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def http_get_json(url: str, timeout: float = 10.0) -> dict:
    txt = http_get_text(url, timeout=timeout)
    return json.loads(txt)


def clamp_cell(cell: tuple[int, int]) -> tuple[int, int]:
    x, y = cell
    return (max(0, min(GRID_COLS - 1, x)), max(0, min(GRID_ROWS - 1, y)))


def parse_cell(text: str) -> tuple[int, int]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Bad cell format '{text}'. Use x,y.")
    x, y = int(parts[0]), int(parts[1])
    if not (0 <= x < GRID_COLS and 0 <= y < GRID_ROWS):
        raise ValueError(f"Cell '{text}' out of bounds for {GRID_COLS}x{GRID_ROWS} grid.")
    return (x, y)


def parse_cells_list(text: str) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for raw in text.replace(";", " ").split():
        out.add(parse_cell(raw))
    return out


def plan_to_targets(
    start: tuple[int, int],
    heading: int,
    blocked: set[tuple[int, int]],
    targets: set[tuple[int, int]],
) -> Optional[str]:
    if start in targets:
        return None
    parents: dict[tuple[int, int], tuple[Optional[tuple[int, int]], Optional[int]]] = {
        start: (None, None)
    }
    q = deque([start])
    found: Optional[tuple[int, int]] = None
    while q:
        cell = q.popleft()
        if cell in targets and cell != start:
            found = cell
            break
        for dh in (HEADING_N, HEADING_E, HEADING_S, HEADING_W):
            dx, dy = HEADING_DELTA[dh]
            nb = (cell[0] + dx, cell[1] + dy)
            if not (0 <= nb[0] < GRID_COLS and 0 <= nb[1] < GRID_ROWS):
                continue
            if nb in WALLS or nb in blocked or nb in parents:
                continue
            parents[nb] = (cell, dh)
            q.append(nb)
    if found is None:
        return None

    cur = found
    first_heading: Optional[int] = None
    while parents[cur][0] is not None:
        prev, taken_heading = parents[cur]
        if prev == start:
            first_heading = taken_heading
            break
        cur = prev  # type: ignore[assignment]
    if first_heading is None:
        return None
    diff = (first_heading - heading) % 4
    if diff == 0:
        return ACTION_FORWARD
    if diff == 1:
        return ACTION_RIGHT
    if diff == 3:
        return ACTION_LEFT
    return ACTION_RIGHT


class DeliverySimulationWindow:
    """Simple planner walkthrough simulation: each press of Step (or each
    auto tick) advances the car by one decided action. Used to validate
    routing / goal / anchor-seek logic without the real car or camera.
    """

    def __init__(
        self,
        root: tk.Tk,
        start_cell: tuple[int, int],
        goals: set[tuple[int, int]],
        step_delay_ms: int = 550,
        on_close: Optional[callable] = None,
    ) -> None:
        self.root = root
        self.on_close = on_close
        self.win = tk.Toplevel(root)
        self.win.title("Planner Simulation")
        self.win.configure(bg="#151515")
        self.win.protocol("WM_DELETE_WINDOW", self.close)

        self.start_cell = start_cell
        self.all_goals = set(goals)
        self.step_delay_ms = tk.IntVar(value=max(50, int(step_delay_ms)))
        self.running = False
        self.after_id: Optional[str] = None
        self.step_in_progress = False

        self.pose = Pose(cell=self.start_cell, heading=HEADING_N)
        self.trail: list[tuple[int, int]] = [self.pose.cell]
        self.active_goals: set[tuple[int, int]] = set(self.all_goals)
        self.known_blocked_cells: set[tuple[int, int]] = set()
        self.confidence = 0.0
        self.localized = False
        self.force_anchor_seek = True
        self.last_action = "-"
        self.last_status = "ready"

        self.vars = {
            "cell": tk.StringVar(value=str(self.pose.cell)),
            "heading": tk.StringVar(value=HEADING_NAMES[self.pose.heading]),
            "status": tk.StringVar(value=self.last_status),
            "last_action": tk.StringVar(value=self.last_action),
            "confidence": tk.StringVar(value=f"{self.confidence:.2f}"),
            "goals": tk.StringVar(value=", ".join(str(g) for g in sorted(self.active_goals))),
        }

        self.canvas = tk.Canvas(self.win, width=CANVAS_W, height=CANVAS_H, bg="#151515", highlightthickness=0)
        self.canvas.grid(row=0, column=0, padx=(10, 12), pady=10)

        side = tk.Frame(self.win, bg="#151515")
        side.grid(row=0, column=1, sticky="n", pady=10)

        for label, key in (
            ("Cell", "cell"),
            ("Heading", "heading"),
            ("Status", "status"),
            ("Last action", "last_action"),
            ("Confidence", "confidence"),
            ("Goals left", "goals"),
        ):
            row = tk.Frame(side, bg="#151515")
            row.pack(anchor="w", pady=2)
            tk.Label(row, text=f"{label}:", width=12, anchor="w", bg="#151515", fg="#bdbdbd").pack(side="left")
            tk.Label(row, textvariable=self.vars[key], bg="#151515", fg="white").pack(side="left")

        btn_row = tk.Frame(side, bg="#151515")
        btn_row.pack(anchor="w", pady=(10, 0))
        tk.Button(btn_row, text="Start/Pause", command=self.toggle_run).pack(side="left", padx=4)
        tk.Button(btn_row, text="Step", command=self.single_step).pack(side="left", padx=4)
        tk.Button(btn_row, text="Reset", command=self.reset).pack(side="left", padx=4)
        tk.Button(btn_row, text="Close", command=self.close).pack(side="left", padx=4)

        delay_row = tk.Frame(side, bg="#151515")
        delay_row.pack(anchor="w", pady=(8, 0))
        tk.Label(delay_row, text="Step delay (ms):", bg="#151515", fg="#bdbdbd").pack(side="left")
        tk.Spinbox(
            delay_row,
            from_=100,
            to=3000,
            increment=50,
            width=6,
            textvariable=self.step_delay_ms,
        ).pack(side="left", padx=6)

        self._draw()
        self._refresh_status()

    def _cell_to_pixel(self, cell: tuple[int, int]) -> tuple[int, int]:
        return MARGIN + cell[0] * CELL_PX, MARGIN + cell[1] * CELL_PX

    def _draw(self) -> None:
        self.canvas.delete("all")
        self.canvas.create_rectangle(
            MARGIN,
            MARGIN,
            MARGIN + GRID_COLS * CELL_PX,
            MARGIN + GRID_ROWS * CELL_PX,
            fill="#14532d",
            outline="",
        )
        for x in range(GRID_COLS + 1):
            px = MARGIN + x * CELL_PX
            self.canvas.create_line(px, MARGIN, px, MARGIN + GRID_ROWS * CELL_PX, fill="#3a3a3a")
        for y in range(GRID_ROWS + 1):
            py = MARGIN + y * CELL_PX
            self.canvas.create_line(MARGIN, py, MARGIN + GRID_COLS * CELL_PX, py, fill="#3a3a3a")
        for cell in WALLS:
            px, py = self._cell_to_pixel(cell)
            self.canvas.create_rectangle(px, py, px + CELL_PX, py + CELL_PX, fill="#e74c3c", outline="")
        for cell, marker_id in ANCHORS.items():
            px, py = self._cell_to_pixel(cell)
            self.canvas.create_rectangle(px + 7, py + 7, px + CELL_PX - 7, py + CELL_PX - 7, fill="#f1c40f", outline="")
            self.canvas.create_text(px + CELL_PX // 2, py + CELL_PX // 2, text=str(marker_id), fill="black")
        for goal in self.active_goals:
            px, py = self._cell_to_pixel(goal)
            self.canvas.create_oval(px + 5, py + 5, px + CELL_PX - 5, py + CELL_PX - 5, fill="#3498db", outline="")
            self.canvas.create_text(px + CELL_PX // 2, py + CELL_PX // 2, text="G", fill="white")
        if len(self.trail) >= 2:
            for a, b in zip(self.trail, self.trail[1:]):
                ax, ay = self._cell_to_pixel(a)
                bx, by = self._cell_to_pixel(b)
                self.canvas.create_line(
                    ax + CELL_PX // 2,
                    ay + CELL_PX // 2,
                    bx + CELL_PX // 2,
                    by + CELL_PX // 2,
                    fill="#2ecc71",
                    width=3,
                )
        cx, cy = self._cell_to_pixel(self.pose.cell)
        cx += CELL_PX // 2
        cy += CELL_PX // 2
        r = CELL_PX // 2 - 8
        h = self.pose.heading
        if h == HEADING_N:
            pts = [(cx, cy - r), (cx - r * 0.7, cy + r * 0.6), (cx + r * 0.7, cy + r * 0.6)]
        elif h == HEADING_E:
            pts = [(cx + r, cy), (cx - r * 0.6, cy - r * 0.7), (cx - r * 0.6, cy + r * 0.7)]
        elif h == HEADING_S:
            pts = [(cx, cy + r), (cx + r * 0.7, cy - r * 0.6), (cx - r * 0.7, cy - r * 0.6)]
        else:
            pts = [(cx - r, cy), (cx + r * 0.6, cy + r * 0.7), (cx + r * 0.6, cy - r * 0.7)]
        self.canvas.create_polygon(*[v for p in pts for v in p], fill="#f39c12", outline="white", width=2)

    def _blocked_dirs_sensor(self) -> tuple[int, int, int, int]:
        cx, cy = self.pose.cell
        out = []
        for h in (HEADING_N, HEADING_E, HEADING_S, HEADING_W):
            dx, dy = HEADING_DELTA[h]
            nb = (cx + dx, cy + dy)
            if not (0 <= nb[0] < GRID_COLS and 0 <= nb[1] < GRID_ROWS):
                out.append(1)
            elif nb in WALLS or nb in self.known_blocked_cells:
                out.append(1)
            else:
                out.append(0)
        return tuple(out)  # type: ignore[return-value]

    def _explore_action(self, blocked_dirs: tuple[int, int, int, int]) -> str:
        if blocked_dirs[self.pose.heading] == 0:
            return ACTION_FORWARD
        left_h = (self.pose.heading - 1) % 4
        right_h = (self.pose.heading + 1) % 4
        left_open = blocked_dirs[left_h] == 0
        right_open = blocked_dirs[right_h] == 0
        if right_open and not left_open:
            return ACTION_RIGHT
        if left_open and not right_open:
            return ACTION_LEFT
        return ACTION_RIGHT

    def _pick_next_action(self) -> Optional[str]:
        if not self.active_goals:
            return None
        blocked_dirs = self._blocked_dirs_sensor()
        if self.confidence < CONFIDENCE_LOW_THRESHOLD:
            self.force_anchor_seek = True
        if self.force_anchor_seek:
            action = plan_to_targets(
                self.pose.cell,
                self.pose.heading,
                set(self.known_blocked_cells),
                set(ANCHORS.keys()),
            )
            return action if action is not None else self._explore_action(blocked_dirs)
        action = plan_to_targets(
            self.pose.cell,
            self.pose.heading,
            set(self.known_blocked_cells),
            set(self.active_goals),
        )
        if action is not None:
            return action
        return self._explore_action(blocked_dirs)

    def _apply_action(self, action: str) -> None:
        self.last_action = action
        if action == ACTION_LEFT:
            self.pose.heading = (self.pose.heading - 1) % 4
            return
        if action == ACTION_RIGHT:
            self.pose.heading = (self.pose.heading + 1) % 4
            return
        if action == ACTION_FORWARD:
            dx, dy = HEADING_DELTA[self.pose.heading]
            nxt = clamp_cell((self.pose.cell[0] + dx, self.pose.cell[1] + dy))
            if nxt not in WALLS:
                self.pose.cell = nxt
                if not self.trail or self.trail[-1] != nxt:
                    self.trail.append(nxt)
            self.confidence = max(0.0, self.confidence - CONFIDENCE_DECAY_PER_STEP)

    def _check_anchors_and_goals(self) -> None:
        if self.pose.cell in ANCHORS:
            self.confidence = CONFIDENCE_RESET_ON_ANCHOR
            self.localized = True
            self.force_anchor_seek = False
        if self.pose.cell in self.active_goals:
            self.active_goals.discard(self.pose.cell)

    def single_step(self) -> None:
        if self.step_in_progress:
            return
        self.step_in_progress = True
        try:
            action = self._pick_next_action()
            if action is None:
                self._on_done()
                return
            self._apply_action(action)
            self._check_anchors_and_goals()
            if not self.active_goals:
                self._on_done()
                return
            self.last_status = "running"
            self._refresh_status()
        finally:
            self.step_in_progress = False

    def _auto_step(self) -> None:
        if not self.running:
            self.after_id = None
            return
        self.single_step()
        if self.running:
            self.after_id = self.win.after(int(self.step_delay_ms.get()), self._auto_step)

    def toggle_run(self) -> None:
        if self.running:
            self.running = False
            self.last_status = "paused"
            if self.after_id is not None:
                try:
                    self.win.after_cancel(self.after_id)
                except Exception:
                    pass
                self.after_id = None
        else:
            self.running = True
            self.last_status = "running"
            if self.after_id is None:
                self.after_id = self.win.after(50, self._auto_step)
        self._refresh_status()

    def reset(self) -> None:
        self.running = False
        if self.after_id is not None:
            try:
                self.win.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None
        self.pose = Pose(cell=self.start_cell, heading=HEADING_N)
        self.trail = [self.pose.cell]
        self.active_goals = set(self.all_goals)
        self.known_blocked_cells.clear()
        self.confidence = 0.0
        self.localized = False
        self.force_anchor_seek = True
        self.last_action = "-"
        self.last_status = "reset"
        self._refresh_status()

    def _on_done(self) -> None:
        self.running = False
        if self.after_id is not None:
            try:
                self.win.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None
        self.last_status = "ALL DELIVERIES DONE!"
        self._refresh_status()
        should_close = messagebox.askyesno(
            "Simulation Complete",
            "ALL DELIVERIES DONE!\n\nClose this simulation window?\n\nChoose No to keep it open.",
            parent=self.win,
        )
        if should_close:
            self.close()

    def _refresh_status(self) -> None:
        self.vars["cell"].set(str(self.pose.cell))
        self.vars["heading"].set(HEADING_NAMES[self.pose.heading])
        self.vars["status"].set(self.last_status)
        self.vars["last_action"].set(self.last_action)
        self.vars["confidence"].set(f"{self.confidence:.2f}")
        if self.active_goals:
            self.vars["goals"].set(", ".join(str(g) for g in sorted(self.active_goals)))
        else:
            self.vars["goals"].set("none")
        self._draw()

    def close(self) -> None:
        self.running = False
        if self.after_id is not None:
            try:
                self.win.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None
        if self.on_close is not None:
            try:
                self.on_close()
            except Exception:
                pass
        self.win.destroy()


class RealRunDashboard:
    def __init__(self, root: tk.Tk, car_base_url: str, camera_url: str, profile: str) -> None:
        self.root = root
        self.root.title("Real Run Dashboard")
        self.root.configure(bg="#1e1e1e")
        self.car_base_url = car_base_url.rstrip("/")
        self.profile = profile
        self.camera = CameraStreamClient(camera_url=camera_url, mode="snapshot")
        self.camera.start()

        self.pose = Pose(cell=(0, 0), heading=HEADING_N)
        self.start_cell = self.pose.cell
        self.all_goals: set[tuple[int, int]] = set(GOALS)
        self.active_goals: set[tuple[int, int]] = set(self.all_goals)
        self.trail: list[tuple[int, int]] = [self.pose.cell]
        self.running = False
        self.paused = True
        self.step_in_progress = False
        self.auto_step_after_id: Optional[str] = None
        self.control_mode = tk.StringVar(value="agent")
        self.step_delay_ms = tk.IntVar(value=550)
        self.last_action = "-"
        self.last_status = "idle"
        self.last_detection: Optional[DetectionResult] = None
        self.last_marker_ids: list[int] = []
        self.last_motion_state = "-"
        self.last_turn_mode = "-"
        self.last_yaw = 0.0
        self.confidence = 0.0
        self.localized = False
        self.force_anchor_seek = True
        self.known_blocked_cells: set[tuple[int, int]] = set()
        self.preview_window_name = "ESP32-CAM Live (Dashboard)"
        cv2.namedWindow(self.preview_window_name, cv2.WINDOW_NORMAL)
        self.preview_refresh_ms = 180
        self.settle_forward_s = 4.0
        self.settle_turn_s = 4.0
        self.camera_sample_offset_s = 3.0
        self.camera_post_action_lag_s = tk.DoubleVar(value=1.5)
        self.has_fresh_frame_for_decision = False
        self.sim_window: Optional[DeliverySimulationWindow] = None

        self._build_ui()
        self._draw_grid()
        self._refresh_status()
        self.root.after(self.preview_refresh_ms, self._preview_tick)

    def _build_ui(self) -> None:
        container = tk.Frame(self.root, bg="#1e1e1e")
        container.pack(padx=12, pady=12)
        self.canvas = tk.Canvas(container, width=CANVAS_W, height=CANVAS_H, bg="#1e1e1e", highlightthickness=0)
        self.canvas.grid(row=0, column=0, padx=(0, 14))

        side = tk.Frame(container, bg="#1e1e1e")
        side.grid(row=0, column=1, sticky="n")

        self.vars = {
            "cell": tk.StringVar(value=str(self.pose.cell)),
            "heading": tk.StringVar(value=HEADING_NAMES[self.pose.heading]),
            "status": tk.StringVar(value="idle"),
            "last_action": tk.StringVar(value="-"),
            "front": tk.StringVar(value="-"),
            "markers": tk.StringVar(value="[]"),
            "yaw": tk.StringVar(value="-"),
            "motion": tk.StringVar(value="-"),
            "turn_mode": tk.StringVar(value="-"),
            "confidence": tk.StringVar(value="0.00"),
            "blocked_dirs": tk.StringVar(value="-"),
            "goal": tk.StringVar(value=", ".join(str(g) for g in sorted(self.active_goals))),
        }

        rows = [
            ("Cell", "cell"),
            ("Heading", "heading"),
            ("Status", "status"),
            ("Last action", "last_action"),
            ("Front status", "front"),
            ("Anchor IDs", "markers"),
            ("Yaw", "yaw"),
            ("Motion state", "motion"),
            ("Turn mode", "turn_mode"),
            ("Confidence", "confidence"),
            ("Blocked N/E/S/W", "blocked_dirs"),
            ("Goals left", "goal"),
        ]
        for lbl, key in rows:
            row = tk.Frame(side, bg="#1e1e1e")
            row.pack(anchor="w", pady=2)
            tk.Label(row, text=f"{lbl}:", width=12, anchor="w", bg="#1e1e1e", fg="#bbbbbb").pack(side="left")
            tk.Label(row, textvariable=self.vars[key], bg="#1e1e1e", fg="white").pack(side="left")

        btn_row = tk.Frame(side, bg="#1e1e1e")
        btn_row.pack(anchor="w", pady=(10, 0))
        tk.Button(btn_row, text="Start/Pause", command=self.toggle_run).pack(side="left", padx=4)
        tk.Button(btn_row, text="Step", command=self.single_step).pack(side="left", padx=4)
        tk.Button(btn_row, text="Reset", command=self.reset).pack(side="left", padx=4)
        tk.Button(btn_row, text="Pause Car/System", command=self.pause_all).pack(side="left", padx=4)
        tk.Button(btn_row, text="Open Simulation", command=self.open_simulation).pack(side="left", padx=4)

        mode_row = tk.Frame(side, bg="#1e1e1e")
        mode_row.pack(anchor="w", pady=(10, 0))
        tk.Label(mode_row, text="Control mode:", bg="#1e1e1e", fg="#bbbbbb").pack(side="left")
        tk.Radiobutton(
            mode_row,
            text="Agent",
            variable=self.control_mode,
            value="agent",
            command=self._on_mode_change,
            bg="#1e1e1e",
            fg="white",
            selectcolor="#1e1e1e",
            activebackground="#1e1e1e",
            activeforeground="white",
        ).pack(side="left", padx=4)
        tk.Radiobutton(
            mode_row,
            text="Manual",
            variable=self.control_mode,
            value="manual",
            command=self._on_mode_change,
            bg="#1e1e1e",
            fg="white",
            selectcolor="#1e1e1e",
            activebackground="#1e1e1e",
            activeforeground="white",
        ).pack(side="left", padx=4)

        manual_row_1 = tk.Frame(side, bg="#1e1e1e")
        manual_row_1.pack(anchor="w", pady=(6, 0))
        tk.Button(manual_row_1, text="Manual Forward (1 cell)", command=lambda: self.manual_step(ACTION_FORWARD)).pack(side="left", padx=4)
        tk.Button(manual_row_1, text="Manual Back (1 cell)", command=lambda: self.manual_step(ACTION_BACK)).pack(side="left", padx=4)

        manual_row_2 = tk.Frame(side, bg="#1e1e1e")
        manual_row_2.pack(anchor="w", pady=(4, 0))
        tk.Button(manual_row_2, text="Manual Left (90°)", command=lambda: self.manual_step(ACTION_LEFT)).pack(side="left", padx=4)
        tk.Button(manual_row_2, text="Manual Right (90°)", command=lambda: self.manual_step(ACTION_RIGHT)).pack(side="left", padx=4)

        delay_row = tk.Frame(side, bg="#1e1e1e")
        delay_row.pack(anchor="w", pady=(8, 0))
        tk.Label(delay_row, text="Step delay (ms):", bg="#1e1e1e", fg="#bbbbbb").pack(side="left")
        tk.Spinbox(delay_row, from_=100, to=3000, increment=50, width=6, textvariable=self.step_delay_ms).pack(
            side="left", padx=6
        )

        cam_lag_row = tk.Frame(side, bg="#1e1e1e")
        cam_lag_row.pack(anchor="w", pady=(6, 0))
        tk.Label(cam_lag_row, text="Camera lag (s):", bg="#1e1e1e", fg="#bbbbbb").pack(side="left")
        tk.Spinbox(
            cam_lag_row,
            from_=0.0,
            to=5.0,
            increment=0.1,
            width=6,
            textvariable=self.camera_post_action_lag_s,
            format="%.2f",
        ).pack(side="left", padx=6)
        tk.Label(
            cam_lag_row,
            text="(wait + drain stale frames after action)",
            bg="#1e1e1e",
            fg="#888",
        ).pack(side="left")

        settings_row = tk.Frame(side, bg="#1e1e1e")
        settings_row.pack(anchor="w", pady=(10, 0))
        tk.Label(settings_row, text="Start x,y:", bg="#1e1e1e", fg="#bbbbbb").pack(side="left")
        self.start_cell_var = tk.StringVar(value=f"{self.start_cell[0]},{self.start_cell[1]}")
        tk.Entry(settings_row, width=8, textvariable=self.start_cell_var).pack(side="left", padx=4)

        goals_row = tk.Frame(side, bg="#1e1e1e")
        goals_row.pack(anchor="w", pady=(6, 0))
        tk.Label(goals_row, text="Goals (x,y ...):", bg="#1e1e1e", fg="#bbbbbb").pack(side="left")
        default_goals_text = " ".join(f"{x},{y}" for x, y in sorted(self.all_goals))
        self.goals_var = tk.StringVar(value=default_goals_text)
        tk.Entry(goals_row, width=24, textvariable=self.goals_var).pack(side="left", padx=4)
        tk.Button(goals_row, text="Apply Start/Goals", command=self.apply_start_and_goals).pack(side="left", padx=4)

    def _cell_to_pixel(self, cell: tuple[int, int]) -> tuple[int, int]:
        return MARGIN + cell[0] * CELL_PX, MARGIN + cell[1] * CELL_PX

    def _draw_grid(self) -> None:
        self.canvas.delete("all")
        self.canvas.create_rectangle(MARGIN, MARGIN, MARGIN + GRID_COLS * CELL_PX, MARGIN + GRID_ROWS * CELL_PX, fill="#14532d", outline="")
        for x in range(GRID_COLS + 1):
            px = MARGIN + x * CELL_PX
            self.canvas.create_line(px, MARGIN, px, MARGIN + GRID_ROWS * CELL_PX, fill="#3a3a3a")
        for y in range(GRID_ROWS + 1):
            py = MARGIN + y * CELL_PX
            self.canvas.create_line(MARGIN, py, MARGIN + GRID_COLS * CELL_PX, py, fill="#3a3a3a")

        for cell in WALLS:
            px, py = self._cell_to_pixel(cell)
            self.canvas.create_rectangle(px, py, px + CELL_PX, py + CELL_PX, fill="#e74c3c", outline="")

        for cell, marker_id in ANCHORS.items():
            px, py = self._cell_to_pixel(cell)
            self.canvas.create_rectangle(px + 7, py + 7, px + CELL_PX - 7, py + CELL_PX - 7, fill="#f1c40f", outline="")
            self.canvas.create_text(px + CELL_PX // 2, py + CELL_PX // 2, text=str(marker_id), fill="black")

        for goal in self.active_goals:
            px, py = self._cell_to_pixel(goal)
            self.canvas.create_oval(px + 5, py + 5, px + CELL_PX - 5, py + CELL_PX - 5, fill="#3498db", outline="")
            self.canvas.create_text(px + CELL_PX // 2, py + CELL_PX // 2, text="G", fill="white")

        if len(self.trail) >= 2:
            for a, b in zip(self.trail, self.trail[1:]):
                ax, ay = self._cell_to_pixel(a)
                bx, by = self._cell_to_pixel(b)
                self.canvas.create_line(
                    ax + CELL_PX // 2, ay + CELL_PX // 2, bx + CELL_PX // 2, by + CELL_PX // 2, fill="#2ecc71", width=3
                )

        self._draw_car()

    def _draw_car(self) -> None:
        cx, cy = self._cell_to_pixel(self.pose.cell)
        cx += CELL_PX // 2
        cy += CELL_PX // 2
        r = CELL_PX // 2 - 8
        h = self.pose.heading
        if h == HEADING_N:
            pts = [(cx, cy - r), (cx - r * 0.7, cy + r * 0.6), (cx + r * 0.7, cy + r * 0.6)]
        elif h == HEADING_E:
            pts = [(cx + r, cy), (cx - r * 0.6, cy - r * 0.7), (cx - r * 0.6, cy + r * 0.7)]
        elif h == HEADING_S:
            pts = [(cx, cy + r), (cx + r * 0.7, cy - r * 0.6), (cx - r * 0.7, cy - r * 0.6)]
        else:
            pts = [(cx - r, cy), (cx + r * 0.6, cy + r * 0.7), (cx + r * 0.6, cy - r * 0.7)]
        self.canvas.create_polygon(*[v for p in pts for v in p], fill="#f39c12", outline="white", width=2)

    def _send_car(self, path: str, params: Optional[dict] = None, timeout: float = 8.0) -> str:
        qs = urllib.parse.urlencode(params or {})
        url = f"{self.car_base_url}{path}"
        if qs:
            url += f"?{qs}"
        return http_get_text(url, timeout=timeout)

    def _poll_car_status(self) -> None:
        try:
            status = http_get_json(f"{self.car_base_url}/status", timeout=2.0)
            self.last_motion_state = status.get("motion_state", "-")
            self.last_turn_mode = status.get("last_turn_mode", "-")
            self.last_yaw = float(status.get("yaw_deg", 0.0))
        except Exception:
            pass

    def _capture_perception(self, update_map: bool = True) -> Optional[DetectionResult]:
        try:
            frame = self.camera.read_frame()
            frame_show, result = analyze_frame(frame, profile=self.profile)
            overlay_detection(frame_show, result, profile=self.profile)
            cv2.imshow(self.preview_window_name, frame_show)
            cv2.waitKey(1)
            self.last_detection = result
            self.last_marker_ids = result.marker_ids
            if update_map and result.blocked:
                dx, dy = HEADING_DELTA[self.pose.heading]
                nb = (self.pose.cell[0] + dx, self.pose.cell[1] + dy)
                if 0 <= nb[0] < GRID_COLS and 0 <= nb[1] < GRID_ROWS:
                    self.known_blocked_cells.add(nb)
            return result
        except Exception:
            return None

    def _preview_tick(self) -> None:
        # Keep camera preview live even while planner/car are paused.
        if self.paused:
            self._capture_perception(update_map=False)
        self.root.after(self.preview_refresh_ms, self._preview_tick)

    def _relocalize_from_anchors(self) -> bool:
        if not self.last_marker_ids:
            return False
        marker_id = self.last_marker_ids[0]
        if marker_id in ANCHOR_BY_ID:
            marker_cell = ANCHOR_BY_ID[marker_id]
            dx, dy = HEADING_DELTA[self.pose.heading]
            inferred_current = (marker_cell[0] - dx, marker_cell[1] - dy)
            if not (0 <= inferred_current[0] < GRID_COLS and 0 <= inferred_current[1] < GRID_ROWS):
                return False
            if inferred_current in WALLS:
                return False
            self.pose.cell = inferred_current
            self.localized = True
            # Keep anchor-seek active until we are physically on an anchor cell.
            # Confidence reset happens only when pose.cell itself is an anchor.
            return True
        return False

    def _blocked_dirs_sensor(self) -> tuple[int, int, int, int]:
        cx, cy = self.pose.cell
        out = []
        for h in (HEADING_N, HEADING_E, HEADING_S, HEADING_W):
            dx, dy = HEADING_DELTA[h]
            nb = (cx + dx, cy + dy)
            if not (0 <= nb[0] < GRID_COLS and 0 <= nb[1] < GRID_ROWS):
                out.append(1)
            elif nb in WALLS or nb in self.known_blocked_cells:
                out.append(1)
            else:
                out.append(0)
        return tuple(out)  # type: ignore[return-value]

    def _explore_action(self, blocked_dirs: tuple[int, int, int, int]) -> str:
        if blocked_dirs[self.pose.heading] == 0:
            return ACTION_FORWARD
        left_h = (self.pose.heading - 1) % 4
        right_h = (self.pose.heading + 1) % 4
        left_open = blocked_dirs[left_h] == 0
        right_open = blocked_dirs[right_h] == 0
        if right_open and not left_open:
            return ACTION_RIGHT
        if left_open and not right_open:
            return ACTION_LEFT
        if right_open and left_open:
            return ACTION_RIGHT
        return ACTION_RIGHT

    def _pick_next_action(self) -> Optional[str]:
        if not self.active_goals:
            return None
        blocked_dirs = self._blocked_dirs_sensor()
        if self.confidence < CONFIDENCE_LOW_THRESHOLD:
            self.force_anchor_seek = True

        if self.force_anchor_seek:
            action = plan_to_targets(self.pose.cell, self.pose.heading, set(self.known_blocked_cells), set(ANCHORS.keys()))
            return action if action is not None else self._explore_action(blocked_dirs)

        action = plan_to_targets(self.pose.cell, self.pose.heading, set(self.known_blocked_cells), set(self.active_goals))
        if action is not None:
            return action
        return self._explore_action(blocked_dirs)

    def _apply_dead_reckoning(self, action: str) -> None:
        if action == ACTION_LEFT:
            self.pose.heading = (self.pose.heading - 1) % 4
            return
        if action == ACTION_RIGHT:
            self.pose.heading = (self.pose.heading + 1) % 4
            return
        dx, dy = HEADING_DELTA[self.pose.heading]
        if action == ACTION_BACK:
            dx, dy = -dx, -dy
        nxt = clamp_cell((self.pose.cell[0] + dx, self.pose.cell[1] + dy))
        if nxt not in WALLS:
            self.pose.cell = nxt
            if not self.trail or self.trail[-1] != nxt:
                self.trail.append(nxt)
            self.confidence = max(0.0, self.confidence - CONFIDENCE_DECAY_PER_STEP)

    def _execute_action(self, action: str) -> None:
        if action == ACTION_FORWARD:
            self._send_car("/test", {"name": "burst"}, timeout=8.0)
        elif action == ACTION_BACK:
            self._send_car("/test", {"name": "backburst"}, timeout=8.0)
        elif action == ACTION_LEFT:
            self._send_car("/test", {"name": "left90"}, timeout=8.0)
        elif action == ACTION_RIGHT:
            self._send_car("/test", {"name": "right90"}, timeout=8.0)
        self.last_action = action

    def _on_mode_change(self) -> None:
        if self.control_mode.get() == "manual":
            self.pause_all()
            self.last_status = "manual mode"
            self._refresh_status()
        else:
            self.last_status = "agent mode"
            self._refresh_status()

    def manual_step(self, action: str) -> None:
        if self.control_mode.get() != "manual":
            self.last_status = "switch to manual mode first"
            self._refresh_status()
            return
        if self.running:
            self.last_status = "stop agent run before manual step"
            self._refresh_status()
            return
        self.single_step(forced_action=action, manual_mode=True)

    def single_step(self, forced_action: Optional[str] = None, manual_mode: bool = False) -> None:
        if self.step_in_progress:
            return
        if self.paused and not manual_mode:
            self.vars["status"].set("paused")
            return
        if not manual_mode and self.control_mode.get() != "agent":
            self.last_status = "manual mode active"
            self._refresh_status()
            return
        self.step_in_progress = True
        was_paused = self.paused
        if manual_mode:
            self.paused = False
        try:
            if forced_action is None and not self.active_goals:
                self._on_all_deliveries_done()
                return
            self._poll_car_status()
            # Ensure we have a CURRENT post-action frame before deciding.
            # On the first step (or after reset/pause) there is no prior
            # post-action drain to chain from, so do one now.
            if not self.has_fresh_frame_for_decision:
                self.last_status = "waiting for camera (initial)..."
                self._refresh_status()
                if not self._drain_and_capture_fresh(float(self.camera_post_action_lag_s.get()), abort_on_pause=not manual_mode):
                    self._refresh_status(extra_status="paused")
                    return
                self._relocalize_from_anchors()
                if self.pose.cell in ANCHORS:
                    self.confidence = CONFIDENCE_RESET_ON_ANCHOR
                    self.localized = True
                    self.force_anchor_seek = False
                self.has_fresh_frame_for_decision = True

            action = forced_action if forced_action is not None else self._pick_next_action()
            if action is None:
                self._on_all_deliveries_done()
                return
            self._execute_action(action)
            self._apply_dead_reckoning(action)

            # Wait for the physical action to settle (motors stop / pose is
            # mechanically reached). No mid-settle decision-making.
            wait_s = self.settle_forward_s if action in (ACTION_FORWARD, ACTION_BACK) else self.settle_turn_s
            if not self._wait_with_pause(wait_s, sample_offset_s=self.camera_sample_offset_s, abort_on_pause=not manual_mode):
                self._refresh_status(extra_status="paused")
                self.has_fresh_frame_for_decision = False
                return

            # Now wait for the ESP32-CAM to actually catch up to the
            # post-action world. Continuously drain frames during this
            # wait so any buffered/stale frames are flushed. ONLY after
            # the lag is over do we read what is truly in front of us.
            self.last_status = "waiting for camera lag..."
            self._refresh_status()
            if not self._drain_and_capture_fresh(float(self.camera_post_action_lag_s.get()), abort_on_pause=not manual_mode):
                self._refresh_status(extra_status="paused")
                self.has_fresh_frame_for_decision = False
                return

            # Now self.last_detection / self.last_marker_ids reflect the
            # actual post-action camera view. Use it to relocalize.
            self._relocalize_from_anchors()
            if self.pose.cell in ANCHORS:
                self.confidence = CONFIDENCE_RESET_ON_ANCHOR
                self.localized = True
                self.force_anchor_seek = False
            if forced_action is None and self.pose.cell in self.active_goals:
                self.active_goals.discard(self.pose.cell)
                if not self.active_goals:
                    self._refresh_status(extra_status="all deliveries done")
                    self._on_all_deliveries_done()
                    return

            # The fresh frame we just captured will drive the NEXT action's
            # decision (no stale capture needed at the top of next step).
            self.has_fresh_frame_for_decision = True
            self.last_status = "manual step done" if manual_mode else "running"
            self._refresh_status()
        finally:
            if manual_mode:
                self.paused = was_paused
            self.step_in_progress = False

    def _drain_and_capture_fresh(self, lag_s: float, abort_on_pause: bool = True) -> bool:
        """Wait `lag_s` seconds AFTER the physical action ends so the
        ESP32-CAM has time to deliver a frame that reflects the
        post-action world. While waiting, continuously fetch & analyze
        frames so stale buffered frames are flushed and the live preview
        keeps refreshing. After the wait, the latest captured frame is
        treated as the authoritative post-action perception (and is also
        used to update `known_blocked_cells`).
        Returns False if user paused during the wait.
        """
        lag_s = max(0.0, float(lag_s))
        start_t = time.time()
        end_t = start_t + lag_s
        last_capture_t = 0.0
        # Drain phase: don't update map yet so the wait period can see
        # stale frames cycle through without polluting state.
        while time.time() < end_t:
            if abort_on_pause and self.paused:
                return False
            now = time.time()
            if now - last_capture_t >= 0.10:
                self._capture_perception(update_map=False)
                last_capture_t = now
            self.root.update_idletasks()
            self.root.update()
            time.sleep(0.03)
        # Final authoritative read AFTER the lag has elapsed. This frame
        # reflects what is actually in front of the car right now, and
        # is allowed to update the blocked-cell map.
        self._capture_perception(update_map=True)
        return True

    def _on_all_deliveries_done(self) -> None:
        self.pause_all()
        self.last_status = "ALL DELIVERIES DONE!"
        self._refresh_status(extra_status=self.last_status)
        should_close = messagebox.askyesno(
            "Mission Complete",
            "ALL DELIVERIES DONE!\n\nDo you want to close the dashboard now?\n\nChoose No to keep it open.",
        )
        if should_close:
            self.close()
            self.root.destroy()

    def apply_start_and_goals(self) -> None:
        try:
            start_cell = parse_cell(self.start_cell_var.get().strip())
            goals = parse_cells_list(self.goals_var.get().strip())
            if not goals:
                raise ValueError("Enter at least one goal cell.")
            if start_cell in WALLS:
                raise ValueError("Start cell cannot be a wall.")
            if any(g in WALLS for g in goals):
                raise ValueError("Goals cannot include wall cells.")
        except Exception as exc:
            messagebox.showerror("Invalid configuration", str(exc))
            return

        self.start_cell = start_cell
        self.all_goals = set(goals)
        self.reset()
        self.last_status = "start/goals updated"
        self._refresh_status()

    def open_simulation(self) -> None:
        try:
            start_cell = parse_cell(self.start_cell_var.get().strip())
            goals = parse_cells_list(self.goals_var.get().strip())
            if not goals:
                raise ValueError("Enter at least one goal cell for simulation.")
            if start_cell in WALLS:
                raise ValueError("Start cell cannot be a wall.")
            if any(g in WALLS for g in goals):
                raise ValueError("Goals cannot include wall cells.")
        except Exception as exc:
            messagebox.showerror("Invalid simulation settings", str(exc))
            return

        if self.sim_window is not None:
            try:
                self.sim_window.close()
            except Exception:
                pass
            self.sim_window = None

        def _clear_ref() -> None:
            self.sim_window = None

        self.sim_window = DeliverySimulationWindow(
            root=self.root,
            start_cell=start_cell,
            goals=set(goals),
            step_delay_ms=int(self.step_delay_ms.get()),
            on_close=_clear_ref,
        )

    def _wait_with_pause(self, seconds: float, sample_offset_s: float = -1.0, abort_on_pause: bool = True) -> bool:
        """Wait while keeping UI responsive; abort early if paused."""
        start_t = time.time()
        end_t = start_t + max(0.0, seconds)
        sampled = False
        while time.time() < end_t:
            if abort_on_pause and self.paused:
                return False
            if (not sampled) and sample_offset_s >= 0.0 and (time.time() - start_t) >= sample_offset_s:
                # Refresh camera during settle period so stream is updated before next decision.
                self._capture_perception(update_map=False)
                sampled = True
            # Keep Tk event loop pumping so Pause button works during waits.
            self.root.update_idletasks()
            self.root.update()
            time.sleep(0.03)
        return True

    def _auto_step(self) -> None:
        if not self.running or self.paused:
            self.auto_step_after_id = None
            return
        if not self.step_in_progress:
            self.single_step()
        self.auto_step_after_id = self.root.after(int(self.step_delay_ms.get()), self._auto_step)

    def toggle_run(self) -> None:
        self.running = not self.running
        self.paused = not self.running
        if self.running:
            self.vars["status"].set("running")
            if self.auto_step_after_id is not None:
                try:
                    self.root.after_cancel(self.auto_step_after_id)
                except Exception:
                    pass
            self.auto_step_after_id = self.root.after(50, self._auto_step)
        else:
            self.pause_all()

    def pause_all(self) -> None:
        self.running = False
        self.paused = True
        self.has_fresh_frame_for_decision = False
        if self.auto_step_after_id is not None:
            try:
                self.root.after_cancel(self.auto_step_after_id)
            except Exception:
                pass
            self.auto_step_after_id = None
        try:
            self._send_car("/pause", timeout=3.0)
        except Exception:
            pass
        self._refresh_status(extra_status="paused")

    def reset(self) -> None:
        self.pause_all()
        self.pose = Pose(cell=self.start_cell, heading=HEADING_N)
        self.active_goals = set(self.all_goals)
        self.trail = [self.pose.cell]
        self.last_action = "-"
        self.last_detection = None
        self.last_marker_ids = []
        self.confidence = 0.0
        self.localized = False
        self.force_anchor_seek = True
        self.known_blocked_cells.clear()
        self.has_fresh_frame_for_decision = False
        self._refresh_status(extra_status="reset")

    def _refresh_status(self, extra_status: Optional[str] = None) -> None:
        self._poll_car_status()
        self.vars["cell"].set(str(self.pose.cell))
        self.vars["heading"].set(HEADING_NAMES[self.pose.heading])
        self.vars["last_action"].set(self.last_action)
        self.vars["markers"].set(str(self.last_marker_ids))
        self.vars["yaw"].set(f"{self.last_yaw:.2f}")
        self.vars["motion"].set(self.last_motion_state)
        self.vars["turn_mode"].set(self.last_turn_mode)
        self.vars["confidence"].set(f"{self.confidence:.2f}")
        self.vars["blocked_dirs"].set(" ".join(str(v) for v in self._blocked_dirs_sensor()))
        if self.active_goals:
            self.vars["goal"].set(", ".join(str(g) for g in sorted(self.active_goals)))
        else:
            self.vars["goal"].set("none")
        if self.last_detection is not None:
            self.vars["front"].set(
                f"{self.last_detection.status} r={self.last_detection.red_ratio:.2f} g={self.last_detection.green_ratio:.2f}"
            )
        else:
            self.vars["front"].set("-")
        if extra_status is not None:
            self.last_status = extra_status
        self.vars["status"].set(self.last_status)
        self._draw_grid()

    def close(self) -> None:
        self.pause_all()
        if self.sim_window is not None:
            try:
                self.sim_window.close()
            except Exception:
                pass
            self.sim_window = None
        self.camera.close()
        cv2.destroyWindow(self.preview_window_name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--car-url", default="http://192.168.4.1")
    ap.add_argument("--camera-url", required=True)
    ap.add_argument("--profile", choices=["Home", "University"], default="Home")
    args = ap.parse_args()

    root = tk.Tk()
    app = RealRunDashboard(root, car_base_url=args.car_url, camera_url=args.camera_url, profile=args.profile)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.close(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
