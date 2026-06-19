"""On-robot echo probe: while the robot speaks (nobody else talking), how loud
does its OWN voice come back through the mic? Decides whether energy-based
barge-in is viable on this hardware or needs echo handling.

Run ON the robot (service stopped first):
    cd ~/everything-agent
    EVERYTHING_AGENT_CONFIG=config.reachy.yaml /venvs/apps_venv/bin/python scripts/bargein_echo_probe.py
"""
import asyncio
import os
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except Exception:
    pass

from everything_agent.core.context import AgentContext
from everything_agent.robot.reachy_mini import ReachyMiniRobot
from everything_agent.expressing.tts.cartesia import CartesiaTTS

ctx = AgentContext(config={})
robot = ReachyMiniRobot({}, ctx)
ctx.robot = robot
print(">>> connecting", flush=True)
robot.connect()
tts = CartesiaTTS({"voice_id": os.environ.get("CARTESIA_VOICE_ID"),
                   "model": "sonic-3", "language": "en"}, ctx)

PHRASE = ("This is a test of whether I can hear my own voice while I am speaking. "
          "One, two, three, four, five, six, seven, eight, nine, ten.")


async def main():
    stop = threading.Event()
    rms_log = []

    async def watch():
        media = robot.media
        try:
            media.start_recording()
        except Exception:
            pass
        while not stop.is_set():
            frame = await asyncio.to_thread(media.get_audio_sample)
            if frame is None or getattr(frame, "size", 0) == 0:
                await asyncio.sleep(0.005)
                continue
            if getattr(frame, "ndim", 1) == 2:
                frame = frame.mean(axis=1)
            frame = frame.astype(np.float32)
            if frame.size:
                rms_log.append(float(np.sqrt(np.mean(frame * frame))))
            await asyncio.sleep(0)

    w = asyncio.create_task(watch())
    print(">>> speaking + listening to self", flush=True)
    await asyncio.to_thread(tts.speak, PHRASE)
    stop.set()
    await w

    arr = np.array(rms_log) if rms_log else np.zeros(1)
    for thr in (0.02, 0.05, 0.1, 0.2):
        print(f"   self-voice frames above {thr:.2f}: {(arr > thr).mean()*100:5.1f}%", flush=True)
    print(f">>> RESULT frames={len(arr)} max_rms={arr.max():.3f} "
          f"mean_rms={arr.mean():.3f} p90={np.percentile(arr,90):.3f}", flush=True)
    print(">>> verdict: " + (
        "ECHO PROBLEM -- mic hears the robot loudly; energy barge-in would self-trigger"
        if (arr > 0.02).mean() > 0.2 else
        "LOW ECHO -- energy barge-in likely viable (pick threshold above max_rms)"), flush=True)


asyncio.run(main())
robot.reset()
robot.disconnect()
print("PROBE DONE", flush=True)
