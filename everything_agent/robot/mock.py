"""Mock robot: logs instead of moving, so the agent runs with no hardware.

The real one later: a sibling `reachy_mini.py` with `ReachyMiniRobot(Robot)` that
wraps the reachy_mini SDK. Add it to BACKENDS in __init__.py and flip the config
-- nothing else changes.
"""
from __future__ import annotations

import logging

from ..core.ports import Robot

log = logging.getLogger("everything_agent.robot")


class MockRobot(Robot):
    def __init__(self, config, ctx):
        self.config = config

    def connect(self):
        log.info("[mock-robot] connected")

    def disconnect(self):
        log.info("[mock-robot] disconnected")

    def move_head(self, yaw=0.0, pitch=0.0, roll=0.0, duration=0.5):
        log.info("[mock-robot] move_head yaw=%.2f pitch=%.2f roll=%.2f (%.1fs)",
                 yaw, pitch, roll, duration)

    def set_antennas(self, left=0.0, right=0.0):
        log.info("[mock-robot] set_antennas left=%.2f right=%.2f", left, right)

    def reset(self):
        log.info("[mock-robot] reset to neutral pose")
