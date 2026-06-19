"""Stress-test the whole agent through the loopback pipeline (real Cartesia voice).

Exercises the hard cases: multiple people speaking in one earful, fuzzy/garbled
address, ambient chatter that must be ignored, off-topic/unknowable questions
(must answer, never go silent), web/current-info, weather, rapid back-to-back
turns, barge-in during a reply, and pure noise. Prints a PASS/FAIL-ish verdict
per scenario plus a summary.

Run:  .venv/bin/python scripts/stress_test.py 2>/tmp/stress.err
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import uuid

import numpy as np
from dotenv import load_dotenv

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
load_dotenv()
os.environ["EVERYTHING_AGENT_CONFIG"] = os.path.join(os.path.dirname(_HERE), "config.loopback.yaml")
from everything_agent.__main__ import load_config       # noqa: E402
from everything_agent.agent import EverythingAgent       # noqa: E402

SR = 16000
API_KEY = os.environ.get("CARTESIA_API_KEY")
VOICES = {  # distinct Cartesia voices to simulate different people
    "A": os.environ.get("CARTESIA_VOICE_ID") or "65209f8e-6140-4a20-b819-3cc2e21da19b",
    "B": "a0e99841-438c-4a64-b679-ae501e7d6091",
    "C": "79a125e8-cd45-4c13-8a67-188112f4dd22",
}


async def synth(text: str, voice: str) -> np.ndarray:
    from cartesia import AsyncCartesia
    client = AsyncCartesia(api_key=API_KEY)
    chunks = []
    try:
        async with client.tts.websocket_connect() as conn:
            ctx = conn.context(f"c-{uuid.uuid4().hex[:8]}")
            await ctx.send(model_id="sonic-3", transcript=text,
                           voice={"mode": "id", "id": VOICES[voice]},
                           output_format={"container": "raw", "encoding": "pcm_f32le",
                                          "sample_rate": SR},
                           language="en", continue_=False)
            await ctx.no_more_inputs()
            async for ev in ctx.receive():
                audio = getattr(ev, "audio", None) or getattr(ev, "data", None)
                if isinstance(audio, (bytes, bytearray)):
                    chunks.append(np.frombuffer(audio, dtype=np.float32))
                if getattr(ev, "type", None) in ("done", "flush_done"):
                    break
    finally:
        await client.close()
    return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)


class Capture(logging.Handler):
    """Grab 'heard'/'not addressed' lines so we can see what STT+gate did."""
    def __init__(self):
        super().__init__(); self.lines = []
    def emit(self, rec):
        self.lines.append(rec.getMessage())
    def take(self):
        out = self.lines; self.lines = []; return out


# (label, [(voice, text), ...], expectation)  expectation in:
#   "reply"  -> must produce a non-empty reply (never silent)
#   "ignore" -> must NOT respond (ambient / not addressed)
SCENARIOS = [
    ("normal: time",        [("A", "Hey Reachy, what time is it?")], "reply"),
    ("fuzzy address",       [("A", "Erichy, tell me a quick joke.")], "reply"),
    ("unknown/private",     [("A", "Reachy, what's my bank account balance?")], "reply"),
    ("absurd/unknowable",   [("A", "Reachy, what number am I thinking of right now?")], "reply"),
    ("out-of-scope action", [("A", "Reachy, order me a pizza and book a flight to Mars.")], "reply"),
    ("weather",             [("A", "Reachy, what's the weather in Berlin?")], "reply"),
    ("web/current info",    [("A", "Reachy, give me one news headline from this week.")], "reply"),
    ("ambient (ignore)",    [("B", "I think we should get tacos for lunch tomorrow.")], "ignore"),
    ("two people, one ask", [("B", "Hey Reachy what's the capital of France"),
                             ("C", "no wait, the capital of Japan")], "reply"),
    ("overheard goodbye",   [("B", "alright everyone, goodbye and good night")], "ignore"),
    ("noise/gibberish",     [("A", "mmmhmm uh brr tk tk")], "ignore"),
]


async def run_scenario(agent, cap, label, parts, expect):
    media = agent.ctx.robot.media
    # Build one earful: concatenate each speaker's audio (a short gap between).
    pcm = []
    for voice, text in parts:
        pcm.append(await synth(text, voice))
        pcm.append(np.zeros(int(SR * 0.25), np.float32))
    media.feed(np.concatenate(pcm))
    cap.take()
    n0 = len(agent.transcript)
    t0 = time.monotonic()
    try:
        await asyncio.wait_for(agent._cycle(), timeout=45)
    except asyncio.TimeoutError:
        return {"label": label, "verdict": "TIMEOUT", "heard": "?", "reply": "(timed out)",
                "dt": 45.0, "expect": expect}
    dt = time.monotonic() - t0
    logs = cap.take()
    heard = next((l.split("heard: ", 1)[1] for l in logs if "heard: " in l), "")
    ignored = any("not addressed" in l for l in logs)
    replied = len(agent.transcript) > n0
    reply = agent.transcript[-1]["reply"] if replied else ("(ignored)" if ignored else "(no reply)")
    if expect == "reply":
        verdict = "PASS" if (replied and reply.strip()) else "FAIL"
    else:  # ignore
        verdict = "PASS" if not replied else "FAIL"
    return {"label": label, "verdict": verdict, "heard": heard or "(none)",
            "reply": reply, "dt": dt, "expect": expect}


async def barge_in(agent, cap):
    """Feed a 2nd utterance WHILE the 1st turn is mid-reply -> shows whether the
    robot can be interrupted (current design: turn-based, so expect it can't)."""
    media = agent.ctx.robot.media
    media.feed(await synth("Reachy, tell me a long story about deep space.", "A"))
    cap.take(); n0 = len(agent.transcript)
    task = asyncio.create_task(agent._cycle())
    await asyncio.sleep(6)                      # let it get into the reply
    media.feed(await synth("Reachy, stop, never mind!", "C"))   # the interruption
    try:
        await asyncio.wait_for(task, timeout=45)
    except asyncio.TimeoutError:
        task.cancel()
    logs = cap.take()
    heard2 = "Reachy, stop" .lower()
    interrupted = any("stop" in l.lower() and "heard" in l.lower() for l in logs)
    return {"label": "barge-in during reply", "interrupted": interrupted,
            "turns_added": len(agent.transcript) - n0}


async def main():
    logging.basicConfig(level=logging.WARNING)
    cap = Capture()
    logging.getLogger("everything_agent").addHandler(cap)
    logging.getLogger("everything_agent").setLevel(logging.INFO)
    logging.getLogger("everything_agent.hearing.stt").addHandler(cap)
    logging.getLogger("everything_agent.hearing.stt").setLevel(logging.INFO)

    agent = EverythingAgent(load_config())
    await agent.start()

    print("\n" + "=" * 78)
    print("STRESS TEST — full pipeline, real Cartesia voice")
    print("=" * 78)
    results = []
    for label, parts, expect in SCENARIOS:
        r = await run_scenario(agent, cap, label, parts, expect)
        results.append(r)
        mark = "✓" if r["verdict"] == "PASS" else "✗"
        print(f"\n{mark} [{r['verdict']}] {label}  ({r['dt']:.1f}s, want={expect})")
        print(f"    heard : {r['heard'][:90]}")
        print(f"    reply : {r['reply'][:160]}")

    print("\n" + "-" * 78)
    bi = await barge_in(agent, cap)
    print(f"• barge-in during reply: heard the interruption mid-speech = {bi['interrupted']}, "
          f"turns added = {bi['turns_added']}")
    print("  (turn-based loop: the robot finishes speaking before it listens again)")

    npass = sum(1 for r in results if r["verdict"] == "PASS")
    print("\n" + "=" * 78)
    print(f"SUMMARY: {npass}/{len(results)} scenarios passed")
    for r in results:
        if r["verdict"] != "PASS":
            print(f"  {r['verdict']}: {r['label']} -> {r['reply'][:80]}")
    print("=" * 78)

    agent._running = False
    agent.ctx.robot.disconnect()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
