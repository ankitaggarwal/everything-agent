"""STT port -- speech-to-text.

Adapters: mock now (you type). Add local `parakeet.py` / `whisper.py`, or a
cloud `cartesia.py`, later -- each one line in the map below.
"""
from __future__ import annotations

from ...core.plugins import load

BACKENDS = {
    "mock": "everything_agent.hearing.stt.mock:MockSTT",
    "cartesia": "everything_agent.hearing.stt.cartesia:CartesiaSTT",
    # "whisper":  "everything_agent.hearing.stt.whisper:WhisperSTT",
    # "parakeet": "everything_agent.hearing.stt.parakeet:ParakeetSTT",
}


def build(config, ctx):
    return load(BACKENDS, config, ctx, default="mock")
