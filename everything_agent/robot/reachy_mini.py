"""Real robot adapter -- drives a Reachy Mini via the `reachy_mini` SDK.

The on-board daemon owns the hardware; this adapter is a *client* to it. On
connect it acquires the shared media (speaker + mic array) and wakes the robot,
then exposes that MediaManager via `.media` so the Cartesia TTS/STT adapters can
play and capture audio through the same device. The Robot port's head/antenna
calls are mapped onto the SDK's 4x4 head pose + antenna joint positions.

Only one app may control the robot at a time (the daemon enforces a lock), so
stop any running app from the dashboard before launching this.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..core.ports import Robot

log = logging.getLogger("everything_agent.robot")

# Neutral antenna joints [right, left] in radians (matches the SDK's init pose).
_INIT_ANTENNAS = [-0.1745, 0.1745]


class ReachyMiniRobot(Robot):
    def __init__(self, config, ctx):
        self.config = config or {}
        self.mini = None

    # ---- lifecycle ----
    def connect(self) -> None:
        from reachy_mini import ReachyMini

        # Force the LOCAL media backend (GStreamer audio straight to the on-board
        # sound card). The SDK's auto-detect uses is_local_camera_available(),
        # which is flaky and silently falls back to WebRTC (needs a producer that
        # isn't running) -> connection failure. We only need audio, so pin LOCAL;
        # the harmless camera "Internal data stream error" warning can be ignored.
        self.mini = ReachyMini(connection_mode="localhost_only", media_backend="local")
        for step, fn in (("acquire_media", self._acquire), ("wake_up", self._wake),
                         ("start_playing", self._start_playing)):
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                log.warning("[reachy_mini] %s failed: %s", step, e)
        log.info("[reachy_mini] connected")

    def _acquire(self):
        self.mini.acquire_media()

    def _wake(self):
        self.mini.wake_up()

    def _start_playing(self):
        # Put the speaker into streaming mode so the TTS adapter can push frames.
        self.mini.media.start_playing()

    def disconnect(self) -> None:
        if not self.mini:
            return
        for fn in (lambda: self.mini.media.stop_playing(),
                   lambda: self.mini.goto_sleep(),
                   lambda: self.mini.release_media()):
            try:
                fn()
            except Exception:  # noqa: BLE001
                pass
        log.info("[reachy_mini] disconnected")

    # ---- shared media (used by Cartesia TTS + STT adapters) ----
    @property
    def media(self):
        return self.mini.media if self.mini else None

    # ---- movement ----
    def move_head(self, yaw=0.0, pitch=0.0, roll=0.0, duration=0.5) -> None:
        if not self.mini:
            return
        pose = np.eye(4)
        pose[:3, :3] = R.from_euler("ZYX", [yaw, pitch, roll]).as_matrix()
        try:
            self.mini.goto_target(head=pose, duration=max(0.1, duration), body_yaw=None)
        except Exception as e:  # noqa: BLE001
            log.warning("[reachy_mini] move_head failed: %s", e)

    def set_antennas(self, left=0.0, right=0.0) -> None:
        if not self.mini:
            return
        # SDK antenna order is [right, left], radians.
        try:
            self.mini.set_target_antenna_joint_positions([float(right), float(left)])
        except Exception as e:  # noqa: BLE001
            log.warning("[reachy_mini] set_antennas failed: %s", e)

    def reset(self) -> None:
        if not self.mini:
            return
        try:
            self.mini.goto_target(head=np.eye(4), antennas=_INIT_ANTENNAS,
                                  duration=0.5, body_yaw=0.0)
        except Exception as e:  # noqa: BLE001
            log.warning("[reachy_mini] reset failed: %s", e)
