"""Robot expressions — short antenna gestures that convey emotion.

The brain can call `set_expression(emotion)` as a tool; we map the emotion to a
quick antenna movement and run it in a background thread so it never blocks the
voice loop. Antenna-only for now (visible + safe, no motor/wifi hit); head nods
and tilts can be added later.
"""
from __future__ import annotations

import threading
import time

# emotion -> sequence of (left, right, hold_seconds). Antenna angle is ~radians;
# the speaking wiggle uses ~0.18, so 0.4-0.5 reads as a clear, lively gesture.
_SEQ = {
    "happy":     [(0.40, 0.40, 0.15), (0.15, 0.15, 0.15), (0.40, 0.40, 0.15), (0.0, 0.0, 0.1)],
    "excited":   [(0.45, 0.45, 0.10), (0.10, 0.10, 0.10)] * 3 + [(0.0, 0.0, 0.1)],
    "sad":       [(-0.40, -0.40, 0.6), (0.0, 0.0, 0.2)],
    "curious":   [(0.40, -0.40, 0.4), (-0.40, 0.40, 0.4), (0.0, 0.0, 0.1)],
    "surprised": [(0.50, 0.50, 0.35), (0.0, 0.0, 0.1)],
    "yes":       [(0.40, 0.40, 0.12), (0.0, 0.0, 0.12), (0.40, 0.40, 0.12), (0.0, 0.0, 0.1)],
    "no":        [(0.40, -0.40, 0.12), (-0.40, 0.40, 0.12), (0.40, -0.40, 0.12), (0.0, 0.0, 0.1)],
    "neutral":   [(0.0, 0.0, 0.1)],
}
NAMES = sorted(_SEQ)
_DEFAULT = "happy"


def perform(body, emotion: str) -> str:
    """Play an expression on the robot in a background thread. Returns the name used.

    Unknown emotions fall back to a friendly default rather than doing nothing.
    On a body with no robot (laptop/mock) this is a no-op that still returns the name.
    """
    name = (emotion or "").strip().lower()
    seq = _SEQ.get(name)
    if seq is None:
        name, seq = _DEFAULT, _SEQ[_DEFAULT]

    robot = getattr(body, "robot", None)
    set_ant = getattr(robot, "set_target_antenna_joint_positions", None)
    if set_ant is None:
        return name  # nothing to move (mock/laptop)

    def _run():
        for left, right, hold in seq:
            try:
                set_ant([float(left), float(right)])
            except Exception:
                break
            time.sleep(hold)
        try:
            set_ant([0.0, 0.0])
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True).start()
    return name
