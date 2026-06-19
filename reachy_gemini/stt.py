"""Cartesia Ink-Whisper speech-to-text.

Takes one buffered utterance (16 kHz mono int16 PCM, as gathered by the VAD gate
in app.py) and returns the transcribed text. We stream the audio to Cartesia's
STT WebSocket, send `finalize`, and collect the final transcript segments.

Raw WebSocket (not the SDK) so the wire protocol is pinned regardless of the
installed cartesia version.
"""
from __future__ import annotations

import asyncio
import json

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


async def transcribe(pcm_int16: bytes, *, api_key: str, model: str = "ink-whisper",
                     language: str = "en", timeout: float = 8.0) -> str:
    """utterance PCM (16k mono int16) -> text. Returns '' on noise/failure."""
    if not pcm_int16 or not api_key:
        return ""
    # int16 -> float32 in [-1, 1], which is what we tell Cartesia we're sending.
    f32 = (np.frombuffer(pcm_int16, dtype=np.int16).astype(np.float32) / 32768.0)
    audio = f32.tobytes()

    url = (f"wss://api.cartesia.ai/stt/websocket?model={model}"
           f"&language={language}&encoding=pcm_f32le&sample_rate={_SR}")
    headers = {"X-API-Key": api_key, "Cartesia-Version": "2024-11-13"}
    segments: list[str] = []

    async with websockets.connect(url, additional_headers=headers, max_size=4_000_000) as ws:
        # ship the whole utterance in ~100 ms chunks, then ask the server to flush.
        step = (_SR // 10) * 4  # 4 bytes per float32 sample
        for i in range(0, len(audio), step):
            await ws.send(audio[i:i + step])
        await ws.send("finalize")

        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=deadline - asyncio.get_running_loop().time())
            except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
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
                    segments.append(t)
            if ev.get("type") in ("done", "flush_done", "error"):
                break

    text = " ".join(segments).strip()
    return "" if _is_noise(text) else text
