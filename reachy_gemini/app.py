"""The whole agent, in one small loop.

Two things happen at once, forever:
  capture: drain the mic -> ship raw PCM up to Gemini Live
  receive: pull audio + transcripts down from Gemini -> play + animate

That's it. Gemini does the listening, understanding, and talking; we just move
bytes and wiggle antennas.
"""
from __future__ import annotations

import asyncio

from google.genai import types

from . import events
from .body import make_body
from .session import open_session


class Agent:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.body = make_body(cfg)
        self.session = None
        self._speaking = False
        self._tool_this_turn = False

    def _set_speaking(self, speaking: bool) -> None:
        if speaking != self._speaking:
            self._speaking = speaking
            self.body.set_speaking(speaking)

    async def _capture(self) -> None:
        """Mic -> Gemini. Blocking reads run off the event loop."""
        loop = asyncio.get_running_loop()
        while self.session is not None:
            pcm = await loop.run_in_executor(None, self.body.read_mic)
            if pcm:
                await self.session.send_realtime_input(
                    audio=types.Blob(data=pcm, mime_type="audio/pcm;rate=16000")
                )
            else:
                await asyncio.sleep(0.01)

    async def _receive(self) -> None:
        """Gemini -> speaker + motion. This is where every stage gets emitted."""
        async for msg in self.session.receive():
            # --- the black box answered. did it want a tool, or just talk? ---
            if msg.tool_call is not None:
                self._tool_this_turn = True
                names = [fc.name for fc in (msg.tool_call.function_calls or [])]
                events.emit(events.TOOL_CALL, text=", ".join(names) or "(unnamed)")
                continue  # no tools wired yet -> nothing to run

            sc = msg.server_content
            if sc is None:
                continue

            if sc.interrupted:
                self._set_speaking(False)
                self.body.clear_playback()
                events.emit(events.INTERRUPTED)
                continue

            if sc.input_transcription and sc.input_transcription.text:
                events.emit(events.HEARING, text=sc.input_transcription.text)

            if sc.output_transcription and sc.output_transcription.text:
                events.emit(events.SPEAKING, text=sc.output_transcription.text)

            if sc.model_turn and sc.model_turn.parts:
                self._set_speaking(True)
                for part in sc.model_turn.parts:
                    if part.inline_data and part.inline_data.data:
                        self.body.play(part.inline_data.data)

            if sc.turn_complete:
                self._set_speaking(False)
                if not self._tool_this_turn:
                    events.emit(events.NO_TOOL)  # the "just talked, path ends" branch
                events.emit(events.DONE)
                self._tool_this_turn = False

    async def run(self) -> None:
        self.body.start()
        events.emit(events.LISTENING, text=f"({self.body.name}) say something...")
        try:
            async with open_session(self.cfg) as session:
                self.session = session
                await asyncio.gather(self._capture(), self._receive())
        finally:
            self.session = None
            self.body.stop()
