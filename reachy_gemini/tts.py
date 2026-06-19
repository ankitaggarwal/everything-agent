"""Cartesia Sonic text-to-speech.

Streams a reply's audio from Cartesia and pushes it through the robot speaker
(body.play, which resamples + hands frames to the daemon). Blocks until the
audio has finished playing, so the turn-based loop in app.py doesn't start
listening again while the robot is still talking (and hear itself).
"""
from __future__ import annotations

import asyncio
import re

import numpy as np

_TTS_SR = 24000  # raw PCM rate we request from Cartesia


def _clean(text: str) -> str:
    # Strip non-ASCII (emoji etc.) -- Cartesia renders them as odd sounds.
    return re.sub(r"[^\x00-\x7F]+", " ", text or "").strip()


async def speak(text: str, body, *, api_key: str, model: str = "sonic-2",
                voice_id: str, language: str = "en", on_first_audio=None) -> None:
    """Synthesize `text` and play it through `body`. Returns once playback ends.

    `on_first_audio` (if given) is called the moment the first audio chunk arrives
    -- used to stamp time-to-first-audio for the latency profiler.
    """
    text = _clean(text)
    if not text or not api_key or not voice_id:
        return
    from cartesia import AsyncCartesia

    client = AsyncCartesia(api_key=api_key)
    n_samples = 0
    body.set_speaking(True)
    try:
        stream = await client.tts.bytes(  # async def -> await, then iterate the chunks
            model_id=model,
            transcript=text,
            voice={"mode": "id", "id": voice_id},
            output_format={"container": "raw", "encoding": "pcm_f32le",
                           "sample_rate": _TTS_SR},
            language=language,
        )
        async for chunk in stream:
            if not chunk:
                continue
            if n_samples == 0 and on_first_audio is not None:
                on_first_audio()
            f32 = np.frombuffer(chunk, dtype=np.float32)
            n_samples += f32.size
            pcm16 = (np.clip(f32, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            body.play(pcm16)  # expects 24 kHz int16; resamples to the speaker rate
        # wait for the queued audio to actually play out before we return
        if n_samples:
            await asyncio.sleep(n_samples / _TTS_SR + 0.3)
    finally:
        body.set_speaking(False)
        try:
            await client.close()
        except Exception:
            pass
