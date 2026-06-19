"""Barge-in test on the loopback agent (no echo): can the user cut the robot off?

Says a long sentence, then feeds an 'interruption' into the mic partway through,
and checks the speech actually stopped early. Also runs a no-interruption control.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
from stress_test import synth, VOICES  # noqa: E402  (also sets EVERYTHING_AGENT_CONFIG)
from everything_agent.__main__ import load_config  # noqa: E402
from everything_agent.agent import EverythingAgent  # noqa: E402

LONG = ("Let me tell you a long and detailed story about the history of space "
        "exploration, starting many decades ago with the very first rockets, and "
        "continuing all the way through the moon landings and beyond into today.")


async def say_for(agent, text, feed_at=None, interrupt_pcm=None):
    media = agent.ctx.robot.media
    media.drain_output()
    t0 = time.monotonic()
    task = asyncio.create_task(agent._say(text))
    if feed_at is not None:
        await asyncio.sleep(feed_at)
        media.feed(interrupt_pcm)
    await task
    dt = time.monotonic() - t0
    spoken = media.drain_output()
    secs = len(spoken) / media.get_output_audio_samplerate()
    return dt, secs, agent._interrupted


async def main():
    agent = EverythingAgent(load_config())
    await agent.start()
    interrupt = await synth("stop, stop, never mind, hold on", "C")

    print("\n=== CONTROL: no interruption (should speak the whole thing) ===")
    dt, secs, intr = await say_for(agent, LONG)
    print(f"  took {dt:.1f}s, produced {secs:.1f}s of speech, interrupted={intr}")
    full = secs

    print("\n=== BARGE-IN: interrupt ~1.5s in (should stop early) ===")
    dt, secs, intr = await say_for(agent, LONG, feed_at=1.5, interrupt_pcm=interrupt)
    print(f"  took {dt:.1f}s, produced {secs:.1f}s of speech, interrupted={intr}")

    ok = intr and secs < full * 0.7
    print(f"\nRESULT: barge-in {'WORKS' if ok else 'did NOT cut speech short'} "
          f"(stopped at {secs:.1f}s vs full {full:.1f}s)")
    agent._running = False
    agent.ctx.robot.disconnect()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
