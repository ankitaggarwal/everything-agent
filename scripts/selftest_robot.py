"""Hardware self-test: exercise the real Reachy Mini adapters (movement + voice).

Run ON the robot:
    cd ~/everything-agent
    EVERYTHING_AGENT_CONFIG=config.reachy.yaml /venvs/apps_venv/bin/python scripts/selftest_robot.py

Proves the reachy_mini robot adapter and the Cartesia TTS adapter work end to end
before launching the full agent loop.
"""
import os
import sys
import time

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

print(">>> connecting"); robot.connect()
print(">>> looking around")
robot.move_head(yaw=0.4, duration=0.6)
robot.move_head(yaw=-0.4, duration=0.6)
robot.move_head(pitch=-0.25, duration=0.6)
robot.move_head(duration=0.6)
print(">>> antennas"); robot.set_antennas(left=0.6, right=-0.6); time.sleep(0.4); robot.set_antennas(0, 0)

tts = CartesiaTTS({"voice_id": os.environ.get("CARTESIA_VOICE_ID"),
                   "model": "sonic-3", "language": "en"}, ctx)
print(">>> speaking")
tts.speak("Hi! I am the everything agent, now alive on Reachy Mini. Let's build something great.")

print(">>> reset + disconnect"); robot.reset(); robot.disconnect()
print("SELFTEST DONE")
