"""Reachy Mini app entry point.

The daemon's AppManager discovers this via the `reachy_mini_apps` entry point,
launches it, and hands us a ready ReachyMini (LOCAL media backend, media handed
over by the daemon). We just run the same Gemini Live loop against it.

Start it from the dashboard, or:
    curl -X POST http://localhost:8000/api/apps/gemini/start
"""
from __future__ import annotations

import asyncio
import threading

from reachy_mini import ReachyMini
from reachy_mini.apps.app import ReachyMiniApp

from .app import Agent
from .body import RobotMediaBody
from .config import load_config


class GeminiApp(ReachyMiniApp):
    """Full-duplex Gemini Live voice on the Reachy Mini."""

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        cfg = load_config()
        body = RobotMediaBody(reachy_mini, cfg)
        agent = Agent(cfg, body=body)
        asyncio.run(agent.run(stop_event=stop_event))


def main() -> None:
    # The daemon launches us as `python -m reachy_gemini.reachy_app`.
    # wrapped_run() builds the ReachyMini (LOCAL media, handed over by the daemon)
    # and calls GeminiApp.run().
    app = GeminiApp()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()


if __name__ == "__main__":
    main()
