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


def _pcm_to_wav(pcm_int16: bytes, rate: int = _SR) -> bytes:
    import io
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm_int16)
    return buf.getvalue()


async def transcribe_gemini(audio_int16: bytes, *, client, model: str) -> str:
    """Transcribe a buffered utterance with Gemini (multimodal) -- much better at
    accented English than ink-whisper. Returns '' on noise/failure."""
    if not audio_int16 or client is None:
        return ""
    from google.genai import types as gt
    wav = _pcm_to_wav(audio_int16)
    try:
        resp = await client.aio.models.generate_content(
            model=model,
            contents=[
                gt.Part(inline_data=gt.Blob(mime_type="audio/wav", data=wav)),
                gt.Part(text="Transcribe the speech in this audio verbatim. Return ONLY the "
                             "exact words spoken, with no commentary. If there is no clear "
                             "speech, return nothing at all."),
            ],
            config=gt.GenerateContentConfig(
                thinking_config=gt.ThinkingConfig(thinking_budget=0), max_output_tokens=200),
        )
        text = (resp.text or "").strip()
    except Exception as e:
        print(f"[stt] gemini transcribe error: {e}", flush=True)
        return ""
    return "" if _is_noise(text) else text


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


async def transcribe(audio_int16: bytes, *, api_key: str, model: str = "ink-whisper",
                     language: str = "en") -> str:
    """Regular (buffered) Cartesia STT: transcribe a complete utterance at once.

    Faster than Gemini STT but less accurate -- the streaming variant fragmented
    transcripts, so we send the whole clean utterance and finalize once.
    """
    if not audio_int16 or not api_key:
        return ""
    async with open_stream(api_key=api_key, model=model, language=language) as stream:
        step = (_SR // 10) * 2  # ~100 ms of int16 audio (2 bytes/sample)
        for i in range(0, len(audio_int16), step):
            await stream.push(audio_int16[i:i + step])
        return await stream.finalize()


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
