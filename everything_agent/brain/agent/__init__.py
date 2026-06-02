"""AGENT port -- the deliberate, tool-using brain.

Adapters:
  mock       -- no LLM; crude keyword tool-call so the pipeline works today.
  haiku      -- conversational, no tools (stopgap).
  fast       -- low-latency tool calling via the Anthropic API directly (no
                subprocess). Best for snappy spoken turns.
  claude_sdk -- the full Claude Agent SDK + MCP servers + approval gate (spawns
                the Claude Code CLI, so higher latency; use when you need MCP).
"""
from __future__ import annotations

from ...core.plugins import load

BACKENDS = {
    "mock": "everything_agent.brain.agent.mock:MockAgent",
    "haiku": "everything_agent.brain.agent.haiku:HaikuAgent",
    "fast": "everything_agent.brain.agent.fast:FastAgent",
    "claude_sdk": "everything_agent.brain.agent.claude_sdk:ClaudeSdkAgent",
}


def build(config, ctx):
    return load(BACKENDS, config, ctx, default="mock")
