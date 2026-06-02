"""Mock wake word: no microphone -- the robot is treated as always awake, so the
mock STT (typing) drives the conversation. The real openWakeWord adapter blocks
until it hears the phrase; same interface.
"""
from __future__ import annotations

from ...core.ports import WakeWord


class MockWakeWord(WakeWord):
    def __init__(self, config, ctx):
        self.phrase = config.get("phrase", "hey reachy")

    async def wait(self) -> bool:
        return True   # always "awake"; the typed line is the utterance
