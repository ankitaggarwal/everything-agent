"""The brain: one fast Gemini Flash text call. Your words in, a short reply out.

No Gemini Live, no streaming session -- just generate_content against the fastest
flash model, with thinking turned OFF for the lowest latency. Keeps a little
conversation history so follow-ups have context.
"""
from __future__ import annotations

from google import genai
from google.genai import types

_DEFAULT_PERSONA = (
    "You are Reachy Mini, a small friendly desk robot. Keep replies short and "
    "spoken-natural -- a sentence or two -- since you are talking out loud."
)


class Brain:
    def __init__(self, cfg: dict):
        g = cfg.get("gemini", {})
        self.model = g.get("model", "gemini-2.5-flash")
        self.persona = g.get("persona", _DEFAULT_PERSONA)
        self.max_tokens = int(g.get("max_tokens", 200))
        self.max_turns = int(g.get("history_turns", 6))
        self.client = genai.Client(api_key=g["api_key"])
        self.history: list[types.Content] = []

    async def ask(self, text: str) -> str:
        """User text -> reply text. Returns '' on failure (kept out of history)."""
        self.history.append(types.Content(role="user", parts=[types.Part(text=text)]))
        try:
            resp = await self.client.aio.models.generate_content(
                model=self.model,
                contents=self.history,
                config=types.GenerateContentConfig(
                    system_instruction=self.persona,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    max_output_tokens=self.max_tokens,
                ),
            )
            reply = (resp.text or "").strip()
        except Exception as e:  # don't let a brain hiccup kill the loop
            print(f"[brain] error: {e}", flush=True)
            self.history.pop()
            return ""
        if reply:
            self.history.append(types.Content(role="model", parts=[types.Part(text=reply)]))
            if len(self.history) > self.max_turns * 2:
                self.history = self.history[-self.max_turns * 2:]
        else:
            self.history.pop()
        return reply
