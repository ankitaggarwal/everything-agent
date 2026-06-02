"""ROUTER port -- the fast brain that picks instant vs. escalate.

Adapters: mock (keyword rules, no key) and anthropic (Haiku). Add gemini.py
(Gemini Flash) the same way.
"""
from __future__ import annotations

from ...core.plugins import load

BACKENDS = {
    "mock": "everything_agent.brain.router.mock:MockRouter",
    "anthropic": "everything_agent.brain.router.anthropic:AnthropicRouter",
    # "gemini": "everything_agent.brain.router.gemini:GeminiRouter",
}


def build(config, ctx):
    return load(BACKENDS, config, ctx, default="mock")
