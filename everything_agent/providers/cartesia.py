"""Example provider integration: Cartesia (text-to-speech).

Providers are thin clients for external services. They are created once at
startup, shared with every module via `setup(..., providers, ...)`, and a module
(e.g. a future VoiceModule) calls them. This file is the template for any
provider -- swap Cartesia for ElevenLabs, OpenAI, Gemini, etc.

Mock mode (no API key) just logs, so the structure runs without credentials.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("everything_agent.providers.cartesia")


class CartesiaProvider:
    def __init__(self, config: dict):
        self.enabled = (config or {}).get("enabled", False)
        self.api_key = os.environ.get("CARTESIA_API_KEY")
        self._client = None
        if self.enabled and self.api_key:
            self._client = self._connect()

    def _connect(self):
        try:
            from cartesia import Cartesia
            return Cartesia(api_key=self.api_key)
        except Exception:
            log.warning("cartesia package not installed -- running in mock mode")
            return None

    def speak(self, text: str) -> None:
        if self._client is None:
            log.info("[cartesia:mock] would speak: %r", text)
            return
        # Real TTS call goes here, streaming audio to the robot speaker.
        log.info("[cartesia] speaking: %r", text)


def build_providers(config: dict) -> dict:
    """Construct all enabled providers from config -> dict shared with modules."""
    providers = {}
    cfg = (config or {}).get("providers", {})
    providers["cartesia"] = CartesiaProvider(cfg.get("cartesia", {}))
    return providers
