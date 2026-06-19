"""Drive a full conversation with the agent -- no hardware, real Cartesia voice.

For each line in the script below this:
  1. synthesizes it to speech with Cartesia (Claude's "voice"),
  2. feeds that audio into the loopback robot's mic,
  3. runs ONE agent cycle (STT -> address gate -> router -> brain -> TTS),
  4. prints what STT heard + the reply, and writes the spoken reply to a WAV.

So Claude can speak to the robot and read/hear exactly what comes back, and a
human can play the /tmp/reply_*.wav files. This is the autonomous build/test
loop for the voice path until the physical robot is online.

Run:  .venv/bin/python scripts/converse.py
      .venv/bin/python scripts/converse.py "your own line"  ["another line" ...]
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time

import numpy as np
import soundfile as sf
from dotenv import load_dotenv

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                      # for voice_roundtrip
sys.path.insert(0, os.path.dirname(_HERE))     # for the everything_agent package

from voice_roundtrip import synthesize         # noqa: E402  (text -> 16 kHz PCM)
from everything_agent.__main__ import load_config  # noqa: E402
from everything_agent.agent import EverythingAgent  # noqa: E402

load_dotenv()

DEFAULT_SCRIPT = [
    "Hey Reachy, what time is it?",
    "Reachy, in one sentence, what can you do?",
    "Thanks Reachy, goodbye.",
]


async def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    os.environ["EVERYTHING_AGENT_CONFIG"] = os.path.join(
        os.path.dirname(_HERE), "config.loopback.yaml")
    script = sys.argv[1:] or DEFAULT_SCRIPT

    agent = EverythingAgent(load_config())
    await agent.start()
    media = agent.ctx.robot.media
    out_sr = media.get_output_audio_samplerate()

    print("\n" + "=" * 64)
    for i, utt in enumerate(script, 1):
        pcm, _ = await synthesize(utt)         # Claude's voice, 16 kHz mono
        media.feed(pcm)                        # ...into the robot's "ear"
        n_before = len(agent.ctx.memory.turns)
        t0 = time.monotonic()
        await agent._cycle()                   # the real pipeline, one turn
        dt = time.monotonic() - t0

        out = media.drain_output()
        wav = None
        if out.size:
            wav = f"/tmp/reply_{i}.wav"
            sf.write(wav, out, out_sr)

        spoke = len(agent.ctx.memory.turns) > n_before
        reply = agent.ctx.memory.turns[-1][1] if spoke else "(ignored / no reply)"
        print(f"\n[{i}] You say : {utt}")
        print(f"    Robot   : {reply}")
        print(f"    reply audio: {wav or '(none)'}   |   round-trip {dt:.2f}s")
        if not agent._running:                 # a "goodbye" ends the loop
            break

    print("\n" + "=" * 64)
    agent._running = False
    if agent.ctx.robot is not None:
        agent.ctx.robot.reset()
        agent.ctx.robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
