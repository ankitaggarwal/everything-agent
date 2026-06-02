"""ROBOT port -- the robot body. Adapters: mock now, reachy_mini later."""
from __future__ import annotations

from ..core.plugins import load

BACKENDS = {
    "mock": "everything_agent.robot.mock:MockRobot",
    "reachy_mini": "everything_agent.robot.reachy_mini:ReachyMiniRobot",
}


def build(config, ctx):
    return load(BACKENDS, config, ctx, default="mock")
