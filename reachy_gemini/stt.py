"""Cartesia Ink-Whisper speech-to-text -- STREAMING.

Instead of buffering the whole utterance and only THEN uploading + transcribing
it (so the latency lands entirely after you stop talking), we open the STT
WebSocket the moment you start speaking and stream the audio live. Cartesia
transcribes as it arrives, so when you stop we just `finalize` and the transcript
is ready almost immediately.

Raw WebSocket (not the SDK) so the wire protocol is pinned regardless of the
installed cartesia version.
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

import numpy as np
import websockets

_SR = 16000  # Cartesia STT input rate

# Common Whisper hallucinations on near-silence -- drop them.
_HALLUCINATIONS = frozenset({
    "you", "thank you", "thanks", "thanks for watching", "bye", "okay", "ok",
    ".", "...", "music", "applause", "subscribe",
})


def _is_noise(text: str) -> bool:
    cleaned = "".join(c for c in text.lower() if c.isalnum() or c == " ").strip()
    return cleaned in _HALLUCINATIONS or len(cleaned) < 2


class SttStream:
    """A live STT connection: push() audio while you speak, finalize() when done."""

    def __init__(self, ws):
        self.ws = ws
        self.segments: list[str] = []
        self._done = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def _consume(self) -> None:
        while True:
            try:
                msg = await self.ws.recv()
            except Exception:
                break
            if isinstance(msg, bytes):
                continue
            try:
                ev = json.loads(msg)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "transcript" and ev.get("is_final"):
                t = (ev.get("text") or "").strip()
                if t:
                    self.segments.append(t)
            if ev.get("type") in ("done", "flush_done", "error"):
                self._done.set()
                break

    async def push(self, pcm_int16: bytes) -> None:
        """Send one mic chunk (16k mono int16) live to Cartesia as float32."""
        if not pcm_int16:
            return
        f32 = (np.frombuffer(pcm_int16, dtype=np.int16).astype(np.float32) / 32768.0)
        try:
            await self.ws.send(f32.tobytes())
        except Exception:
            pass

    async def finalize(self, timeout: float = 4.0) -> str:
        """Flush + return the final transcript ('' on noise/failure)."""
        try:
            await self.ws.send("finalize")
        except Exception:
            pass
        try:
            await asyncio.wait_for(self._done.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
        text = " ".join(self.segments).strip()
        return "" if _is_noise(text) else text


@asynccontextmanager
async def open_stream(*, api_key: str, model: str = "ink-whisper", language: str = "en"):
    """Open a live STT stream; a background task collects transcripts as audio arrives."""
    url = (f"wss://api.cartesia.ai/stt/websocket?model={model}"
           f"&language={language}&encoding=pcm_f32le&sample_rate={_SR}")
    headers = {"X-API-Key": api_key, "Cartesia-Version": "2024-11-13"}
    async with websockets.connect(url, additional_headers=headers, max_size=4_000_000) as ws:
        stream = SttStream(ws)
        stream._task = asyncio.create_task(stream._consume())
        try:
            yield stream
        finally:
            if stream._task:
                stream._task.cancel()
