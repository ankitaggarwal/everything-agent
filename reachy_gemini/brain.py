"""The brain: one fast Gemini Flash call with tools (function calling).

Your words in, a short reply out -- plus the model can call tools: get_current_time,
set_expression, look_around, dance, and ignore (stay silent). Thinking is OFF for
the lowest latency. Keeps a little conversation history for context.
"""
from __future__ import annotations

from google import genai
from google.genai import types

from . import tools as tools_mod

_DEFAULT_PERSONA = (
    "You are Reachy Mini, a small friendly desk robot. Keep replies short and "
    "spoken-natural -- a sentence or two -- since you are talking out loud."
)

_TOOL_HINT = (
    "\n\nYou have a body and tools. Use set_expression(emotion) to react with your "
    "antennas; look_around() when curious or asked to look; dance(move) when asked to "
    "dance or to celebrate; get_current_time() for the time or date. If what you heard "
    "is clearly not addressed to you, is background chatter, or needs no reply at all, "
    "call ignore() and say nothing."
)


class Brain:
    def __init__(self, cfg: dict, body=None):
        g = cfg.get("gemini", {})
        self.model = g.get("model", "gemini-2.5-flash")
        self.persona = g.get("persona", _DEFAULT_PERSONA)
        self.max_tokens = int(g.get("max_tokens", 1000))
        self.max_turns = int(g.get("history_turns", 6))
        self.client = genai.Client(api_key=g["api_key"])
        self.history: list[types.Content] = []
        self.ctx = {"ignored": False}
        self.tools = tools_mod.build_tools(body, self.ctx) if body is not None else []
        self.ignored = False  # did the model choose to stay silent this turn?

    async def ask(self, text: str) -> str:
        """User text -> reply text (''=no reply). Sets self.ignored if it chose silence."""
        self.ctx["ignored"] = False
        self.ignored = False
        self.history.append(types.Content(role="user", parts=[types.Part(text=text)]))
        try:
            resp = await self.client.aio.models.generate_content(
                model=self.model,
                contents=self.history,
                config=types.GenerateContentConfig(
                    system_instruction=self.persona + _TOOL_HINT,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),  # thinking OFF
                    max_output_tokens=self.max_tokens,
                    tools=self.tools,
                ),
            )
            reply = (resp.text or "").strip()
        except Exception as e:  # don't let a brain hiccup kill the loop
            print(f"[brain] error: {e}", flush=True)
            self.history.pop()
            return ""

        self.ignored = bool(self.ctx.get("ignored"))
        if self.ignored:
            self.history.pop()  # drop ambient / not-for-me from context
            return ""
        if reply:
            self.history.append(types.Content(role="model", parts=[types.Part(text=reply)]))
            if len(self.history) > self.max_turns * 2:
                self.history = self.history[-self.max_turns * 2:]
        else:
            self.history.pop()
        return reply
