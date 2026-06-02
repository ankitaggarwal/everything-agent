"""Mock TTS: logs the spoken text (no audio) and gives a small antenna gesture so
the robot still feels alive. Reads the robot from the shared context.
"""
from __future__ import annotations

import logging

from ...core.ports import TTS

log = logging.getLogger("everything_agent.expressing.tts")


class MockTTS(TTS):
    def __init__(self, config, ctx):
        self.robot = ctx.robot

    def speak(self, text: str) -> None:
        if not text:
            return
        log.info("🔊 robot says: %s", text)
        self.robot.set_antennas(left=0.3, right=0.3)
        self.robot.set_antennas(left=0.0, right=0.0)
