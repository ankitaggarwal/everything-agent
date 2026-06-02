"""Mock STT: you type instead of speak. Returns None on EOF (Ctrl-D / end of
piped input) so the loop can stop cleanly. Real adapters (Parakeet, Whisper,
Cartesia) record from the mic and return the transcript -- same interface.
"""
from __future__ import annotations

import asyncio
from typing import Optional

from ...core.ports import STT


class MockSTT(STT):
    def __init__(self, config, ctx):
        self.config = config

    async def listen(self) -> Optional[str]:
        try:
            text = await asyncio.to_thread(input, "you 🎤> ")
        except EOFError:
            return None
        return text.strip()
