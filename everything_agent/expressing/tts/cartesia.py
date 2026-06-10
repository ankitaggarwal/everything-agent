"""Cartesia TTS adapter -- real voice, streamed through the robot's speaker.

Synthesizes text with Cartesia Sonic over a WebSocket (raw float32 PCM) and
pushes the audio frames to the Reachy Mini's shared MediaManager (obtained from
`ctx.robot.media`). On a laptop with the mock robot there's no speaker, so it
logs instead. A little antenna movement accompanies speech.

Needs CARTESIA_API_KEY and a CARTESIA_VOICE_ID (env or config).
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import uuid

import numpy as np

from ...core.ports import TTS

log = logging.getLogger("everything_agent.expressing.tts")

_TTS_SR = 24000  # Cartesia raw PCM sample rate we request


def _clean(text: str) -> str:
    # Strip non-ASCII (emoji etc.) -- Cartesia renders them as odd sounds.
    return re.sub(r"[^\x00-\x7F]+", " ", text or "").strip()


class CartesiaTTS(TTS):
    def __init__(self, config, ctx):
        self.robot = ctx.robot
        cfg = config or {}
        self.model = cfg.get("model", "sonic-3")
        self.language = cfg.get("language", "en")
        self.voice_id = cfg.get("voice_id") or os.environ.get("CARTESIA_VOICE_ID", "")
        self.api_key = os.environ.get("CARTESIA_API_KEY")
        if not self.api_key:
            log.warning("Cartesia TTS selected but CARTESIA_API_KEY is not set")
        if not self.voice_id:
            log.warning("Cartesia TTS selected but no voice_id (CARTESIA_VOICE_ID)")

    def speak(self, text: str) -> None:
        text = _clean(text)
        if not text:
            return
        media = self.robot.media if self.robot else None
        if not (self.api_key and self.voice_id and media is not None):
            log.info("🔊 (cartesia not ready) robot would say: %s", text)
            return
        log.info("🔊 speaking: %s", text)
        self.robot.set_antennas(left=0.3, right=0.3)   # perk up while speaking
        # speak() is called synchronously from inside the agent's running event
        # loop, so we can't asyncio.run() here. Run the streaming coroutine in a
        # worker thread with its own loop and block until the audio has played.
        import threading
        err: dict = {}

        def _runner():
            try:
                asyncio.run(self._speak(text, media))
            except Exception as e:  # noqa: BLE001
                err["e"] = e

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        t.join()
        if "e" in err:
            log.warning("cartesia speak failed: %s", err["e"])
        self.robot.set_antennas(left=0.0, right=0.0)
        self._drain_mic(media)

    @staticmethod
    def _drain_mic(media) -> None:
        # The mic kept recording while we talked, so our own voice is now queued
        # up -- discard it so the next listen() doesn't transcribe the robot.
        try:
            for _ in range(2000):   # bounded: ~stops even if frames keep arriving
                frame = media.get_audio_sample()
                if frame is None or getattr(frame, "size", 0) == 0:
                    break
        except Exception:  # noqa: BLE001
            pass

    async def _speak(self, text: str, media) -> None:
        import time
        from cartesia import AsyncCartesia
        from scipy.signal import resample_poly

        out_sr = media.get_output_audio_samplerate()
        total_samples = 0
        t_start = time.monotonic()
        first_audio = None
        client = AsyncCartesia(api_key=self.api_key)
        try:
            async with client.tts.websocket_connect() as conn:
                ctx = conn.context(f"ctx-{uuid.uuid4().hex[:8]}")
                await ctx.send(
                    model_id=self.model,
                    transcript=text,
                    voice={"mode": "id", "id": self.voice_id},
                    output_format={"container": "raw", "encoding": "pcm_f32le",
                                   "sample_rate": _TTS_SR},
                    language=self.language,
                    continue_=False,
                )
                await ctx.no_more_inputs()
                async for event in ctx.receive():
                    audio = getattr(event, "audio", None) or getattr(event, "data", None)
                    if isinstance(audio, (bytes, bytearray)):
                        frame = np.frombuffer(audio, dtype=np.float32)
                        if out_sr != _TTS_SR:
                            frame = resample_poly(frame, out_sr, _TTS_SR).astype(np.float32)
                        if first_audio is None:
                            first_audio = time.monotonic() - t_start
                            log.info("⏱  tts first audio in %.2fs", first_audio)
                        total_samples += frame.shape[0]
                        await asyncio.to_thread(media.push_audio_sample, frame)
                    if getattr(event, "type", None) in ("done", "flush_done"):
                        break
        finally:
            try:
                await client.close()
            except Exception:  # noqa: BLE001
                pass
        # Let the buffered audio finish playing before we return (so the mic
        # loop doesn't immediately transcribe the robot's own voice).
        if total_samples:
            await asyncio.sleep(total_samples / float(out_sr) + 0.2)
