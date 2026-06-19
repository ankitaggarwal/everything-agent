"""Cartesia STT adapter -- streaming speech-to-text from the robot mic.

Captures audio from the Reachy Mini's shared MediaManager (`ctx.robot.media`),
resamples to 16 kHz, and streams it to Cartesia Ink-Whisper over a WebSocket.
`listen()` returns the first final transcript of an utterance (or None on
timeout, so the loop keeps cycling). Reuses the same Cartesia key as TTS.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Optional

import numpy as np

from ...core.ports import STT

log = logging.getLogger("everything_agent.hearing.stt")

_STT_SR = 16000

# Common Whisper hallucinations on silence -- drop them.
_HALLUCINATIONS = frozenset({
    "you", "thank you", "thanks", "thanks for watching", "bye", "okay", "ok",
    ".", "...", "music", "applause",
})


def _is_noise(text: str) -> bool:
    cleaned = "".join(c for c in text.lower() if c.isalnum() or c == " ").strip()
    return cleaned in _HALLUCINATIONS or len(cleaned) < 2


class CartesiaSTT(STT):
    def __init__(self, config, ctx):
        self.robot = ctx.robot
        cfg = config or {}
        self.model = cfg.get("model", "ink-whisper")
        self.language = cfg.get("language", "en")
        self.timeout = float(cfg.get("timeout", 30.0))   # max seconds to wait per turn
        # End-of-turn: keep collecting final segments until the speaker has been
        # quiet this long. Without it we'd return on the FIRST is_final -- so
        # "Hey Reachy, <pause> what time is it?" would be truncated to "Hey
        # Reachy" (Ink-Whisper endpoints on the mid-sentence pause).
        self.end_gap = float(cfg.get("end_gap", 0.8))
        self.api_key = os.environ.get("CARTESIA_API_KEY")

    async def listen(self) -> Optional[str]:
        media = self.robot.media if self.robot else None
        if not (self.api_key and media is not None):
            log.warning("Cartesia STT not ready (key/media missing)")
            await asyncio.sleep(1.0)
            return ""
        try:
            return await self._listen(media)
        except Exception as e:  # noqa: BLE001
            log.warning("cartesia stt error: %s", e)
            await asyncio.sleep(1.0)
            return ""

    async def _listen(self, media) -> Optional[str]:
        import websockets
        from scipy.signal import resample_poly

        mic_sr = media.get_input_audio_samplerate()
        try:
            media.start_recording()
        except Exception:  # noqa: BLE001
            pass

        url = (f"wss://api.cartesia.ai/stt/websocket?model={self.model}"
               f"&language={self.language}&encoding=pcm_f32le&sample_rate={_STT_SR}")
        headers = {"X-API-Key": self.api_key, "Cartesia-Version": "2024-11-13"}
        segments: list[str] = []          # final pieces of the current utterance
        last_final = [0.0]                # monotonic time of the most recent piece
        done = asyncio.Event()
        deadline = time.monotonic() + self.timeout

        def turn_over() -> bool:
            # Ended once we've heard something and then gone quiet past end_gap.
            return bool(segments) and (time.monotonic() - last_final[0]) > self.end_gap

        async with websockets.connect(url, additional_headers=headers,
                                      max_size=4_000_000) as ws:
            async def pump():
                batch, n = [], 0
                target = _STT_SR // 10  # ~100 ms chunks
                while not done.is_set() and time.monotonic() < deadline:
                    frame = await asyncio.to_thread(media.get_audio_sample)
                    if frame is None or frame.size == 0:
                        await asyncio.sleep(0.005)
                        continue
                    if frame.ndim == 2:
                        frame = frame.mean(axis=1).astype(np.float32)
                    if mic_sr != _STT_SR:
                        frame = resample_poly(frame, _STT_SR, mic_sr).astype(np.float32)
                    batch.append(frame)
                    n += frame.shape[0]
                    if n >= target:
                        try:
                            await ws.send(np.concatenate(batch).astype(np.float32).tobytes())
                        except Exception:  # noqa: BLE001
                            return
                        batch, n = [], 0

            async def consume():
                while not done.is_set() and time.monotonic() < deadline:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=0.3)
                    except asyncio.TimeoutError:
                        if turn_over():           # quiet long enough -> end turn
                            done.set()
                        continue
                    except Exception:  # noqa: BLE001
                        done.set()
                        return
                    if isinstance(msg, bytes):
                        continue
                    try:
                        ev = json.loads(msg)
                    except json.JSONDecodeError:
                        continue
                    if ev.get("type") == "transcript" and ev.get("is_final"):
                        text = (ev.get("text") or "").strip()
                        if text and not _is_noise(text):
                            log.info("heard segment: %r", text)
                            segments.append(text)
                            last_final[0] = time.monotonic()
                    if turn_over():
                        done.set()

            await asyncio.gather(pump(), consume())

        utterance = " ".join(segments).strip()
        if utterance:
            log.info("heard: %r", utterance)
        return utterance
