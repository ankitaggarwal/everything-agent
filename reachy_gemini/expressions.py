"""Robot expressions — short, characterful gestures that convey emotion.

The brain calls set_expression(emotion); we play a quick sequence of head moves
(tilt / nod / look) + antenna positions + a little body sway, so the robot really
"acts" the feeling instead of just twitching its antennas. Runs in a background
thread (goto_target blocks per step, so steps just chain) and never blocks the loop.

Per step: (head_kwargs_for_create_head_pose, [left_antenna, right_antenna], body_yaw_rad, duration_s)
  head angles are DEGREES: roll = head tilt L/R, pitch = nod (up/down), yaw = look L/R.
  antennas ~ radians (0.5 = perked up, -0.4 = drooped). body_yaw ~ radians (small = subtle sway).
"""
from __future__ import annotations

import threading
import time

_N = ({}, [0.0, 0.0], 0.0, 0.4)  # neutral / home

_SEQ = {
    # bright, bouncy, antennas up, happy little sway
    "happy": [
        ({"pitch": 12}, [0.5, 0.5], 0.0, 0.3),
        ({"pitch": 16, "roll": 9}, [0.55, 0.3], 0.12, 0.28),
        ({"pitch": 14, "roll": -9}, [0.3, 0.55], -0.12, 0.28),
        ({"pitch": 14, "roll": 6}, [0.55, 0.35], 0.08, 0.26),
        _N,
    ],
    # fast, energetic head bobs + flicking antennas + body wiggle
    "excited": [
        ({"pitch": 16}, [0.65, 0.65], 0.1, 0.16),
        ({"pitch": 6}, [0.2, 0.2], -0.1, 0.16),
        ({"pitch": 18}, [0.65, 0.65], 0.1, 0.16),
        ({"pitch": 6}, [0.2, 0.2], -0.1, 0.16),
        ({"pitch": 16}, [0.6, 0.6], 0.0, 0.16),
        _N,
    ],
    # the classic inquisitive head TILT + lean + look to the side, slow and held
    "curious": [
        ({"roll": 24, "pitch": 6, "yaw": 14}, [0.55, -0.15], 0.12, 0.55),
        ({"roll": 24, "pitch": 6, "yaw": 14}, [0.5, -0.1], 0.12, 0.5),   # hold the tilt
        ({"roll": -18, "pitch": 6, "yaw": -10}, [-0.15, 0.55], -0.1, 0.55),
        _N,
    ],
    # head sinks down, antennas droop, slow slump
    "sad": [
        ({"pitch": -22}, [-0.4, -0.4], 0.0, 0.8),
        ({"pitch": -22, "roll": 9}, [-0.4, -0.4], 0.08, 0.6),  # slump to one side
        ({"pitch": -18}, [-0.35, -0.35], 0.0, 0.6),
        _N,
    ],
    # quick snap back/up, antennas shoot up, hold wide, settle
    "surprised": [
        ({"pitch": 24}, [0.7, 0.7], 0.0, 0.15),
        ({"pitch": 20}, [0.65, 0.65], 0.0, 0.3),
        _N,
    ],
    # an emphatic nod
    "yes": [
        ({"pitch": -14}, [0.35, 0.35], 0.0, 0.22),
        ({"pitch": 16}, [0.45, 0.45], 0.0, 0.22),
        ({"pitch": -14}, [0.35, 0.35], 0.0, 0.22),
        ({"pitch": 16}, [0.45, 0.45], 0.0, 0.22),
        _N,
    ],
    # a head/body shake
    "no": [
        ({"yaw": 16}, [0.25, 0.25], 0.18, 0.28),
        ({"yaw": -16}, [0.25, 0.25], -0.18, 0.28),
        ({"yaw": 16}, [0.25, 0.25], 0.18, 0.28),
        _N,
    ],
    "neutral": [_N],
}
NAMES = sorted(_SEQ)
_DEFAULT = "happy"


def perform(body, emotion: str) -> str:
    """Play an expressive gesture on the robot in a background thread. Returns the name used."""
    name = (emotion or "").strip().lower()
    seq = _SEQ.get(name)
    if seq is None:
        name, seq = _DEFAULT, _SEQ[_DEFAULT]

    robot = getattr(body, "robot", None)
    if robot is None:
        return name  # laptop/mock: nothing to move
    try:
        from reachy_mini.utils import create_head_pose
    except Exception as e:
        print(f"[expr] create_head_pose unavailable: {e}", flush=True)
        return name

    def _run():
        for head_kw, antennas, body_yaw, dur in seq:
            try:
                head = create_head_pose(**head_kw)  # degrees by default
                robot.goto_target(head=head,
                                  antennas=[float(antennas[0]), float(antennas[1])],
                                  body_yaw=float(body_yaw), duration=float(dur))  # blocks ~dur
            except Exception as e:
                print(f"[expr] {name} step failed: {e}", flush=True)
                break

    threading.Thread(target=_run, daemon=True).start()
    return name
