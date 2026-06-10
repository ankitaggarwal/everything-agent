"""Emotions module -- physical expressions the brain can perform.

Each action conveys a feeling with a short head + antenna choreography (through
the Robot port). Guided by the persona, the brain calls these to react while it
talks -- nodding, perking up, tilting curiously, drooping when sad.

For speed, expressions are FIRE-AND-FORGET: the action kicks off the movement in
the background and returns immediately, so the gesture overlaps with the spoken
reply instead of blocking it. Blocking SDK head moves run in a worker thread to
keep the event loop free.

This is the template for any movement capability: copy the folder, add actions,
register the class in core/registry.py, and list the module in config.
"""
from __future__ import annotations

import asyncio
import logging
from typing import List

from ...core.module import Action, Module

log = logging.getLogger("everything_agent.modules.emotions")


class EmotionsModule(Module):
    name = "emotions"

    def setup(self, robot, providers, memory, config) -> None:
        self.robot = robot
        self._tasks = set()   # asyncio only weakly references tasks; keep them alive

    def _play(self, steps) -> None:
        """Run a choreography in the background. steps: list of either
        ("head", yaw, pitch, roll, dur) or ("ant", left, right, hold)."""
        async def run():
            try:
                for s in steps:
                    if s[0] == "head":
                        await asyncio.to_thread(self.robot.move_head, s[1], s[2], s[3], s[4])
                    else:
                        self.robot.set_antennas(left=s[1], right=s[2])
                        await asyncio.sleep(s[3])
                self.robot.set_antennas(left=0.0, right=0.0)        # back to neutral
                await asyncio.to_thread(self.robot.move_head, 0.0, 0.0, 0.0, 0.3)
            except Exception as e:  # noqa: BLE001
                log.warning("emotion playback failed: %s", e)

        try:
            task = asyncio.get_running_loop().create_task(run())
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        except RuntimeError:  # no loop (e.g. tests) -> run synchronously
            asyncio.run(run())

    def actions(self) -> List[Action]:
        def emotion(name, desc, steps_fn, blurb):
            async def handler(args):
                self._play(steps_fn())
                return blurb
            return Action(name, desc, handler)

        return [
            emotion("express_happy", "Show happiness or joy with body language",
                    lambda: [("ant", 0.7, 0.7, 0.12), ("head", 0.0, -0.2, 0.0, 0.25),
                             ("ant", 0.4, 0.7, 0.12), ("ant", 0.7, 0.4, 0.12)],
                    "*perks up and bobs happily*"),
            emotion("express_excited", "Show excitement or enthusiasm",
                    lambda: [("ant", 0.8, -0.8, 0.1), ("ant", -0.8, 0.8, 0.1),
                             ("ant", 0.8, -0.8, 0.1), ("head", 0.2, 0.0, 0.0, 0.15),
                             ("head", -0.2, 0.0, 0.0, 0.15)],
                    "*vibrates with excitement*"),
            emotion("express_curious", "Show curiosity or interest (head tilt)",
                    lambda: [("head", 0.0, 0.0, 0.35, 0.4), ("ant", 0.6, 0.0, 0.3)],
                    "*tilts head curiously*"),
            emotion("express_confused", "Show confusion or puzzlement",
                    lambda: [("head", 0.0, 0.0, 0.3, 0.35), ("head", 0.0, 0.0, -0.3, 0.35)],
                    "*tilts head back and forth, puzzled*"),
            emotion("express_thinking", "Show that you're thinking it over",
                    lambda: [("head", 0.2, -0.2, 0.0, 0.4), ("ant", 0.3, 0.5, 0.4)],
                    "*looks up, thinking*"),
            emotion("express_sad", "Show sadness or disappointment",
                    lambda: [("ant", -0.6, -0.6, 0.2), ("head", 0.0, 0.3, 0.0, 0.6)],
                    "*droops sadly*"),
            emotion("nod_yes", "Nod to agree or say yes",
                    lambda: [("head", 0.0, 0.25, 0.0, 0.2), ("head", 0.0, -0.1, 0.0, 0.2),
                             ("head", 0.0, 0.25, 0.0, 0.2), ("head", 0.0, -0.1, 0.0, 0.2)],
                    "*nods yes*"),
            emotion("shake_no", "Shake head to disagree or say no",
                    lambda: [("head", 0.3, 0.0, 0.0, 0.18), ("head", -0.3, 0.0, 0.0, 0.18),
                             ("head", 0.3, 0.0, 0.0, 0.18), ("head", -0.3, 0.0, 0.0, 0.18)],
                    "*shakes head no*"),
            emotion("celebrate", "Do a happy little celebration dance",
                    lambda: [("ant", 0.8, 0.8, 0.12), ("head", 0.3, -0.15, 0.0, 0.18),
                             ("head", -0.3, -0.15, 0.0, 0.18), ("head", 0.3, -0.15, 0.0, 0.18),
                             ("ant", 0.8, -0.8, 0.1), ("ant", -0.8, 0.8, 0.1)],
                    "*does a little celebration dance*"),
        ]
