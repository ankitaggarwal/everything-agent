"""Boot the agent (loopback, web enabled) and hit the settings API -- no hardware."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from dotenv import load_dotenv  # noqa: E402
load_dotenv()
# Use a throwaway copy so the apply test (which persists config) never clobbers
# the real config.loopback.yaml.
import shutil  # noqa: E402
_tmp_cfg = "/tmp/loopback_smoketest.yaml"
shutil.copy(os.path.join(os.path.dirname(_HERE), "config.loopback.yaml"), _tmp_cfg)
os.environ["EVERYTHING_AGENT_CONFIG"] = _tmp_cfg
from everything_agent.__main__ import load_config  # noqa: E402
from everything_agent.agent import EverythingAgent  # noqa: E402


def get(path):
    return urllib.request.urlopen("http://127.0.0.1:8081" + path, timeout=5).read()


def post(path, obj):
    req = urllib.request.Request("http://127.0.0.1:8081" + path,
                                 data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(req, timeout=5).read()


async def main():
    agent = EverythingAgent(load_config())
    await agent.start()
    # seed a fake turn so the transcript view has content
    agent.transcript.append({"t": 0, "heard": "hey reachy what time is it",
                             "reply": "It's 9:18 AM.", "stt": 1.2, "brain": 0.9,
                             "tts": 2.1, "total": 4.2})
    page = get("/")
    print("GET /            ->", len(page), "bytes,", "has <title>:", b"<title>" in page)
    st = json.loads(get("/api/status"))
    print("GET /api/status  -> online:", st["online"], "| trigger:", st["trigger"],
          "| modules:", st["modules"], "| voices:", len(st["catalogue"]["voices"]),
          "| turns:", len(st["transcript"]))
    r = json.loads(post("/api/apply", {"trigger": "robbie"}))
    print("POST /api/apply  -> live trigger change:", r, "| now:", agent.trigger)
    r2 = json.loads(post("/api/apply", {"persona": "You are a witty robot."}))
    print("POST persona     ->", r2)
    print("RESULT: web settings API OK")
    agent._running = False
    agent.ctx.robot.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
