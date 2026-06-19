"""Anthropic router: one tiny Haiku call decides the path. Needs the `anthropic`
package and ANTHROPIC_API_KEY. This is the real Tier-1 brain (ROADMAP Phase 1).
"""
from __future__ import annotations

import json
import logging

from ...core.ports import Decision, Router

log = logging.getLogger("everything_agent.brain.router")

_SYSTEM = (
    "You are the fast router for Reachy, a cheerful, playful little desk robot that "
    "kids love talking to. Read the utterance and decide how to handle it. Reply "
    'with ONLY JSON: {"action": "instant"|"agent"|"ignore", "reply": "<text if '
    'instant>"}. '
    "Lean toward RESPONDING -- a chatty robot delights kids. Use 'instant' for "
    "greetings, small talk, simple facts, or any reply you can give in ONE warm, "
    "playful, kid-friendly sentence (put it in 'reply'). If the words are garbled or "
    "you can't quite tell what they asked, STILL respond 'instant' with a cheerful "
    "'Ooh, say that again?' rather than ignoring them. Use 'agent' when it needs "
    "tools, current info, weather, timers, or real actions. Use 'ignore' ONLY when "
    "the input is empty or pure background noise with nothing aimed at you. "
    "CRITICAL: instant 'reply' is spoken to a small child -- keep it to ONE short, "
    "snappy sentence UNDER 12 WORDS. Ask at most one quick question. Never list "
    "options or stack multiple questions. Short and fun beats long and chatty."
)


class AnthropicRouter(Router):
    def __init__(self, config, ctx):
        self.model = config.get("model", "claude-haiku-4-5")
        # A hung routing call stalls the whole voice loop, so keep it short.
        self.timeout = float(config.get("timeout", 10.0))
        self._client = None   # reused across turns (keeps the HTTP pool warm)

    def _get_client(self):
        if self._client is None:
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic(timeout=self.timeout)
        return self._client

    async def decide(self, text: str, memory_context: str = "") -> Decision:
        if not text or not text.strip():
            return Decision("ignore")
        client = self._get_client()
        user = (memory_context + "\n\nUser: " + text) if memory_context else text
        msg = await client.messages.create(
            model=self.model, max_tokens=200, system=_SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        raw = msg.content[0].text if msg.content else ""
        try:
            return self._parse(raw)
        except Exception:
            log.warning("Router returned non-JSON (%r) -- escalating to agent", raw[:60])
            return Decision("agent")

    @staticmethod
    def _parse(raw: str) -> Decision:
        # Haiku sometimes wraps the JSON in prose or ```json fences; pull out the
        # first {...} block before parsing.
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            raw = raw[start:end + 1]
        data = json.loads(raw)
        return Decision(data.get("action", "agent"), data.get("reply", ""))
