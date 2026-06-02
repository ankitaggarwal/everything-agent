"""WAKEWORD port -- listens locally for "Hey Reachy".

Adapters: mock (always awake / always-listening) and openwakeword (on-device
ONNX detection of the wake phrase).
"""
from __future__ import annotations

from ...core.plugins import load

BACKENDS = {
    "mock": "everything_agent.hearing.wakeword.mock:MockWakeWord",
    "openwakeword": "everything_agent.hearing.wakeword.openwakeword:OpenWakeWord",
}


def build(config, ctx):
    return load(BACKENDS, config, ctx, default="mock")
