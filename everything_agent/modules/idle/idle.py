"""Idle/aliveness module -- what the robot does when it isn't replying.

Two behaviors, both driven by tick() (the loop runs ticks continuously while it
waits for the wake word AND while it listens -- see agent.py `_be_alive`):

  1. Speaker tracking -- read the mic-array Direction-of-Arrival and turn the head
     toward whoever is talking, so the robot "tracks what you're saying".
  2. Organic idle -- gentle, randomized head drift + antenna breathing so it
     looks alive (not frozen) between conversations.

Plus a simple look_at action the brain can call. All head moves run in a worker
thread (the SDK call blocks) so the audio loop keeps flowing.

DoA mapping (angle -> head yaw) is approximate; tune `doa_scale`/`doa_sign`/
`yaw_max` under `idle:` in config if it tracks the wrong way or over/undershoots.
"""
from __future__ import annotations

import asyncio
import math
import random
import time
from typing import List

from ...core.module import Action, Module

_DIRECTIONS = {"left": (-0.4, 0.0), "right": (0.4, 0.0), "up": (0.0, -0.3),
               "down": (0.0, 0.3), "center": (0.0, 0.0)}


class IdleModule(Module):
    name = "idle"

    def setup(self, robot, providers, memory, config) -> None:
        self.robot = robot
        cfg = (config or {}).get("idle", {}) if isinstance(config, dict) else {}
        self.doa_scale = float(cfg.get("doa_scale", 0.45))
        self.doa_sign = float(cfg.get("doa_sign", -1.0))
        self.yaw_max = float(cfg.get("yaw_max", 0.7))
        now = time.monotonic()
        self._last_track = now
        self._last_move = now
        self._last_ant = now
        self._next_drift = random.uniform(3.0, 6.0)
        self._next_ant = random.uniform(2.5, 5.0)

    # ---- aliveness (called ~4x/second by the loop) ----
    async def tick(self) -> None:
        now = time.monotonic()
        media = getattr(self.robot, "media", None)

        # 1) Track the speaker: turn toward voice DoA (responsive, rate-limited).
        if media is not None and now - self._last_track > 0.4:
            try:
                doa = await asyncio.to_thread(media.get_DoA)
            except Exception:  # noqa: BLE001
                doa = None
            if doa:
                angle, speech = doa
                if speech:
                    self._last_track = self._last_move = now
                    yaw = self._angle_to_yaw(angle)
                    await asyncio.to_thread(self.robot.move_head, yaw, 0.0, 0.0, 0.5)
                    return

        # 2) Organic idle: occasional gentle head drift.
        if now - self._last_move > self._next_drift:
            self._last_move = now
            self._next_drift = random.uniform(3.0, 7.0)
            yaw = random.uniform(-0.25, 0.25)
            pitch = random.uniform(-0.10, 0.12)
            await asyncio.to_thread(self.robot.move_head, yaw, pitch, 0.0,
                                    random.uniform(0.9, 1.5))
        # 3) ...and a small antenna "breath" now and then.
        elif now - self._last_ant > self._next_ant:
            self._last_ant = now
            self._next_ant = random.uniform(2.5, 5.0)
            a = random.uniform(0.05, 0.2)
            self.robot.set_antennas(left=a, right=a)

    def _angle_to_yaw(self, angle: float) -> float:
        a = angle % (2 * math.pi)
        if a > math.pi:
            a -= 2 * math.pi          # -> (-pi, pi]
        return max(-self.yaw_max, min(self.yaw_max, self.doa_sign * a * self.doa_scale))

    # ---- simple movement action the brain can call ----
    def actions(self) -> List[Action]:
        async def look_at(args):
            yaw, pitch = _DIRECTIONS.get(args.get("direction", "center"), (0.0, 0.0))
            self.robot.move_head(yaw=yaw, pitch=pitch)
            return f"Looking {args.get('direction', 'center')}."

        return [
            Action("look_at", "Point the robot's head in a direction "
                   "(left/right/up/down/center)", look_at, {"direction": str}),
        ]
