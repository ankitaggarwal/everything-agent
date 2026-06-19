"""Cartesia voice round-trip smoke test (no robot, no agent).

Proves the two halves of "Claude can talk to the robot and read what it hears":

    text  --Cartesia Sonic TTS-->  speech (WAV)  --Cartesia Ink-Whisper STT-->  text

Run:  .venv/bin/python scripts/voice_roundtrip.py "Hey Reachy, what time is it?"

Writes the synthesized audio to /tmp/roundtrip.wav so a human can play it back,
and prints the STT transcript + per-leg latency so we can judge fidelity/speed.
This is the same Cartesia models the robot uses (sonic-3 + ink-whisper), so a
green result here means the robot's hearing+voice path is sound.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid

import numpy as np
import soundfile as sf
from dotenv import load_dotenv

load_dotenv()

SR = 16000  # one sample rate end-to-end keeps the test simple (STT wants 16 kHz)
API_KEY = os.environ.get("CARTESIA_API_KEY")
VOICE_ID = os.environ.get("CARTESIA_VOICE_ID")
OUT_WAV = "/tmp/roundtrip.wav"


async def synthesize(text: str) -> tuple[np.ndarray, float]:
    """text -> float32 mono PCM at SR, plus seconds-to-first-audio."""
    from cartesia import AsyncCartesia

    client = AsyncCartesia(api_key=API_KEY)
    chunks: list[np.ndarray] = []
    t0 = time.monotonic()
    first = None
    try:
        async with client.tts.websocket_connect() as conn:
            ctx = conn.context(f"ctx-{uuid.uuid4().hex[:8]}")
            await ctx.send(
                model_id="sonic-3",
                transcript=text,
                voice={"mode": "id", "id": VOICE_ID},
                output_format={"container": "raw", "encoding": "pcm_f32le",
                               "sample_rate": SR},
                language="en",
                continue_=False,
            )
            await ctx.no_more_inputs()
            async for event in ctx.receive():
                audio = getattr(event, "audio", None) or getattr(event, "data", None)
                if isinstance(audio, (bytes, bytearray)):
                    if first is None:
                        first = time.monotonic() - t0
                    chunks.append(np.frombuffer(audio, dtype=np.float32))
                if getattr(event, "type", None) in ("done", "flush_done"):
                    break
    finally:
        await client.close()
    pcm = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    return pcm, (first or 0.0)


async def transcribe(pcm: np.ndarray) -> tuple[str, float]:
    """float32 PCM at SR -> (final transcript, seconds to final)."""
    import websockets

    url = (f"wss://api.cartesia.ai/stt/websocket?model=ink-whisper"
           f"&language=en&encoding=pcm_f32le&sample_rate={SR}")
    headers = {"X-API-Key": API_KEY, "Cartesia-Version": "2024-11-13"}
    t0 = time.monotonic()
    final = {"text": None, "t": 0.0}
    async with websockets.connect(url, additional_headers=headers,
                                  max_size=4_000_000) as ws:
        async def send():
            step = SR // 10  # 100 ms chunks, like the live adapter
            for i in range(0, len(pcm), step):
                await ws.send(pcm[i:i + step].astype(np.float32).tobytes())
                await asyncio.sleep(0.02)  # pace it like a real mic stream
            await ws.send("finalize")

        async def recv():
            while final["text"] is None:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=8.0)
                except asyncio.TimeoutError:
                    return
                if isinstance(msg, bytes):
                    continue
                ev = json.loads(msg)
                if ev.get("type") == "transcript" and ev.get("is_final"):
                    txt = (ev.get("text") or "").strip()
                    if txt:
                        final["text"] = txt
                        final["t"] = time.monotonic() - t0

        await asyncio.gather(send(), recv())
    return final["text"] or "", final["t"]


async def main() -> int:
    text = " ".join(sys.argv[1:]) or "Hey Reachy, what time is it?"
    if not (API_KEY and VOICE_ID):
        print("ERROR: CARTESIA_API_KEY / CARTESIA_VOICE_ID not set in .env")
        return 2

    print(f"  I say     : {text!r}")
    pcm, tts_first = await synthesize(text)
    dur = len(pcm) / SR
    sf.write(OUT_WAV, pcm, SR)
    print(f"  TTS       : {dur:.1f}s of audio, first byte {tts_first:.2f}s -> {OUT_WAV}")

    heard, stt_t = await transcribe(pcm)
    print(f"  It hears  : {heard!r}  (final in {stt_t:.2f}s)")

    a = "".join(c for c in text.lower() if c.isalnum())
    b = "".join(c for c in heard.lower() if c.isalnum())
    print(f"  Match     : {'OK round-trip faithful' if a and a in b or b in a else 'DIVERGED -- inspect'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
