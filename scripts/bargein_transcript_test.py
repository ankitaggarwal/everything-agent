"""Test echo-robust ("just talk to interrupt") barge-in on the loopback agent.

Two cases while the robot is mid-reply:
  1. ECHO  -- feed the robot's OWN words back into the mic -> must NOT interrupt.
  2. USER  -- feed brand-new words into the mic            -> MUST interrupt,
              and the new words become the next turn's input.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
from stress_test import synth  # noqa: E402  (sets EVERYTHING_AGENT_CONFIG=loopback)
from everything_agent.__main__ import load_config  # noqa: E402
from everything_agent.agent import EverythingAgent  # noqa: E402

STORY = ("Let me tell you a long story about the history of space exploration, "
         "starting many decades ago with the very first rockets and satellites, "
         "moving through the moon landings, and continuing into the present day.")
ECHO_FRAGMENT = "starting many decades ago with the very first rockets and satellites"
USER_WORDS = "wait stop, what's the weather in Tokyo"


async def trial(agent, feed_text, feed_voice, at=1.2):
    media = agent.ctx.robot.media
    media.drain_output()
    agent._interrupt_text = None
    agent._interrupted = False
    interrupt = await synth(feed_text, feed_voice)
    t0 = time.monotonic()
    task = asyncio.create_task(agent._say(STORY))
    await asyncio.sleep(at)
    media.feed(interrupt)
    await task
    dt = time.monotonic() - t0
    secs = len(media.drain_output()) / media.get_output_audio_samplerate()
    return dt, secs, agent._interrupted, agent._interrupt_text


async def main():
    agent = EverythingAgent(load_config())
    await agent.start()

    print("\n=== ECHO: feed the robot's own words (must NOT interrupt) ===")
    dt, secs, intr, itext = await trial(agent, ECHO_FRAGMENT, "A")
    print(f"  took {dt:.1f}s, {secs:.1f}s spoken, interrupted={intr}  -> "
          f"{'PASS (ignored its own echo)' if not intr else 'FAIL (self-interrupted)'}")

    print("\n=== USER: feed new words (must interrupt + capture them) ===")
    dt, secs, intr, itext = await trial(agent, USER_WORDS, "C")
    print(f"  took {dt:.1f}s, {secs:.1f}s spoken, interrupted={intr}")
    print(f"  captured next-turn input: {itext!r}")
    ok = intr and itext
    print(f"  -> {'PASS (interrupted, will answer the new question)' if ok else 'FAIL'}")

    agent._running = False
    agent.ctx.robot.disconnect()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
