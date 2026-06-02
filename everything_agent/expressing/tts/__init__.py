"""TTS port -- text-to-speech (the robot's voice).

Adapters: mock (logs) and cartesia (real). Add piper.py / parakeet.py (local) or
elevenlabs.py the same way -- each one line below.
"""
from __future__ import annotations

from ...core.plugins import load

BACKENDS = {
    "mock": "everything_agent.expressing.tts.mock:MockTTS",
    "cartesia": "everything_agent.expressing.tts.cartesia:CartesiaTTS",
    # "parakeet": "everything_agent.expressing.tts.parakeet:ParakeetTTS",
    # "elevenlabs": "everything_agent.expressing.tts.elevenlabs:ElevenLabsTTS",
}


def build(config, ctx):
    return load(BACKENDS, config, ctx, default="mock")
