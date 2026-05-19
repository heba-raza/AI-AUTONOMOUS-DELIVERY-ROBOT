"""
Shared configuration for the RL Maze RC Car project.

Runtime scripts inside draft2 import from this file so MQTT topics and
grid map constants remain consistent.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# MQTT broker
# ---------------------------------------------------------------------------
MQTT_BROKER_HOST = "192.168.1.100"   # laptop IP on local network
MQTT_BROKER_PORT = 1883
MQTT_USERNAME = ""
MQTT_PASSWORD = ""
MQTT_KEEPALIVE = 30

# ---------------------------------------------------------------------------
# MQTT topics
# ---------------------------------------------------------------------------
TOPIC_CAMERA = "car/camera"
TOPIC_COMMAND = "car/command"
TOPIC_STATUS = "car/status"
TOPIC_TELEMETRY = "car/telemetry"
TOPIC_MODE = "car/mode"

# ---------------------------------------------------------------------------
# MQTT action protocol payload keys
# ---------------------------------------------------------------------------
PROTO_VERSION = 1
KEY_VERSION = "v"
KEY_ACTION_ID = "action_id"
KEY_ACTION = "action"
KEY_STATUS = "status"
KEY_OK = "ok"
KEY_TS_MS = "ts_ms"
KEY_DETAIL = "detail"

COMMAND_ACK_TIMEOUT_S = 4.0
COMMAND_RETRY_LIMIT = 1

# ---------------------------------------------------------------------------
# High-level actions
# ---------------------------------------------------------------------------
ACTION_FORWARD = "FORWARD"
ACTION_TURN_LEFT = "TURN_LEFT"
ACTION_TURN_RIGHT = "TURN_RIGHT"
ACTION_STOP = "STOP"

ACTIONS = [ACTION_FORWARD, ACTION_TURN_LEFT, ACTION_TURN_RIGHT]
NUM_ACTIONS = len(ACTIONS)

STATUS_DONE_FORWARD = "DONE_FORWARD"
STATUS_DONE_LEFT = "DONE_LEFT"
STATUS_DONE_RIGHT = "DONE_RIGHT"
STATUS_IDLE = "IDLE"
STATUS_ERROR = "ERROR"

# ---------------------------------------------------------------------------
# Grid map: 5 columns x 10 rows
# ---------------------------------------------------------------------------
GRID_COLS = 5
GRID_ROWS = 10
GOALS_PER_EPISODE = 3

HEADING_N, HEADING_E, HEADING_S, HEADING_W = 0, 1, 2, 3
HEADING_DELTA = {
    HEADING_N: (0, -1),
    HEADING_E: (1, 0),
    HEADING_S: (0, 1),
    HEADING_W: (-1, 0),
}

WALLS: set = {
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

ANCHORS = {
    (0, 0): 0,
    (2, 5): 1,
    (4, 9): 2,
}
WALLS -= set(ANCHORS.keys())

# ---------------------------------------------------------------------------
# Confidence model
# ---------------------------------------------------------------------------
CONFIDENCE_INIT = 1.0
CONFIDENCE_DECAY_PER_STEP = 0.05
CONFIDENCE_RESET_ON_ANCHOR = 1.0
CONFIDENCE_LOW_THRESHOLD = 0.55

# Trained model path used by run_real_car.py
MODEL_PATH = "dqn_model.pth"
