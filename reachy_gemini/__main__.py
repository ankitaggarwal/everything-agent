"""Entry point.

  python -m reachy_gemini              # full-duplex voice loop (mic <-> Gemini)
  python -m reachy_gemini --check      # no mic: send one text turn, prove the
                                       # Gemini Live connection + key + model work
  python -m reachy_gemini --check --speak
                                       # same, but play the reply OUT LOUD through
                                       # the configured body (proves audio on robot)
"""
from __future__ import annotations

import asyncio
import sys

from google.genai import types

from .app import Agent
from .body import make_body
from .config import load_config
from .session import open_session


async def check(cfg: dict, speak: bool = False) -> int:
    """Deterministic smoke test: text in, audio out. No mic needed."""
    body = None
    if speak:
        body = make_body(cfg)
        body.start()
        print(f"Playing reply through the '{body.name}' body.", flush=True)

    print(f"Connecting to {cfg['gemini']['model']} ...", flush=True)
    async with open_session(cfg) as session:
        print("Connected. Sending a text turn.", flush=True)
        await session.send_client_content(
            turns=types.Content(role="user", parts=[types.Part(text="Say hi in five words.")]),
            turn_complete=True,
        )
        audio_bytes, said = 0, ""
        async for msg in session.receive():
            sc = msg.server_content
            if sc is None:
                continue
            if sc.output_transcription and sc.output_transcription.text:
                said += sc.output_transcription.text
            if sc.model_turn and sc.model_turn.parts:
                for part in sc.model_turn.parts:
                    if part.inline_data and part.inline_data.data:
                        audio_bytes += len(part.inline_data.data)
                        if body is not None:
                            body.play(part.inline_data.data)
            if sc.turn_complete:
                break
        if body is not None:
            await asyncio.sleep(4)  # let the speaker drain
            body.stop()
    print(f"\nOK -- got {audio_bytes} bytes of audio. Model said: {said!r}")
    return 0 if audio_bytes else 1


def main() -> int:
    cfg = load_config()
    if "--check" in sys.argv:
        return asyncio.run(check(cfg, speak="--speak" in sys.argv))
    try:
        asyncio.run(Agent(cfg).run())
    except KeyboardInterrupt:
        print("\nbye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
