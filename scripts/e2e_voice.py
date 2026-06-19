"""Autonomous end-to-end voice test (no human needed).

Synthesize a spoken question with macOS `say`, stream it into Gemini Live exactly
like a mic would, capture the spoken answer, play it on the Mac speaker, and report
what was heard, what was said, and the latency to first audio.

    python scripts/e2e_voice.py "What is the capital of France? One short sentence."
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
import time
import wave

import numpy as np
import sounddevice as sd
from google.genai import types

sys.path.insert(0, ".")
from reachy_gemini.config import load_config
from reachy_gemini.session import open_session

IN_RATE, OUT_RATE = 16000, 24000


def synth(text: str, path: str = "/tmp/e2e_q.wav") -> bytes:
    subprocess.run(
        ["say", "-o", path, "--data-format=LEI16@16000", text], check=True
    )
    with wave.open(path, "rb") as w:
        assert w.getframerate() == IN_RATE, w.getframerate()
        return w.readframes(w.getnframes())


async def ask(cfg: dict, pcm: bytes) -> dict:
    t0 = time.time()
    heard, said, audio = "", "", bytearray()
    first_audio = None
    async with open_session(cfg) as s:
        chunk = int(IN_RATE * 0.05) * 2  # 50 ms of int16

        async def feed() -> None:
            for i in range(0, len(pcm), chunk):
                await s.send_realtime_input(
                    audio=types.Blob(data=pcm[i:i + chunk], mime_type="audio/pcm;rate=16000")
                )
                await asyncio.sleep(0.05)  # real-time pace
            await s.send_realtime_input(audio_stream_end=True)

        feeder = asyncio.create_task(feed())
        async for m in s.receive():
            sc = m.server_content
            if not sc:
                continue
            if sc.input_transcription and sc.input_transcription.text:
                heard += sc.input_transcription.text
            if sc.output_transcription and sc.output_transcription.text:
                said += sc.output_transcription.text
            if sc.model_turn and sc.model_turn.parts:
                for p in sc.model_turn.parts:
                    if p.inline_data and p.inline_data.data:
                        if first_audio is None:
                            first_audio = time.time() - t0
                        audio += p.inline_data.data
            if sc.turn_complete:
                break
        await feeder
    return {"heard": heard, "said": said, "audio": bytes(audio),
            "first_audio_s": first_audio, "total_s": time.time() - t0}


def main() -> int:
    q = " ".join(sys.argv[1:]) or "What is the capital of France? Answer in one short sentence."
    cfg = load_config()
    print(f"Q (spoken): {q!r}")
    pcm = synth(q)
    print(f"  synthesized {len(pcm)} bytes ({len(pcm)/2/IN_RATE:.1f}s of speech)")
    r = asyncio.run(ask(cfg, pcm))
    print(f"  heard back (STT of my speech): {r['heard']!r}")
    print(f"  model said:                    {r['said']!r}")
    print(f"  latency to first audio: {r['first_audio_s']:.2f}s | turn total: {r['total_s']:.2f}s")
    print(f"  reply audio: {len(r['audio'])} bytes")
    if r["audio"]:
        print("  playing reply on Mac speaker...")
        sd.play(np.frombuffer(r["audio"], dtype=np.int16), OUT_RATE)
        sd.wait()
    return 0 if r["audio"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
