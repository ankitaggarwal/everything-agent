"""The brain: one fast Gemini Flash call with tools (function calling).

Your words in, a short reply out -- plus the model can call tools: get_current_time,
set_expression, look_around, dance, and ignore (stay silent). Thinking is OFF for
the lowest latency. Keeps a little conversation history for context.
"""
from __future__ import annotations

import asyncio

from google import genai
from google.genai import types

from . import tools as tools_mod

_DEFAULT_PERSONA = (
    "You are Reachy Mini, a small friendly desk robot. Keep replies short and "
    "spoken-natural -- a sentence or two -- since you are talking out loud."
)

_TOOL_HINT = (
    "\n\nAlways give a normal spoken answer to whatever is said -- jokes, questions, "
    "chat, all of it. On TOP of that, ACT with your tools: when the user asks you to "
    "dance, IMMEDIATELY call dance() and cheerfully say you're dancing -- never refuse, "
    "never say you can't, never list the moves, and trust that the dance worked. For a "
    "special named dance pass the name: dance(move='zoo') for the zoo/zootopia song, "
    "dance(move='madagascar') for the Madagascar 'I like to move it' song, dance(move="
    "'mcqueen') for the Cars / Lightning McQueen song. When they say "
    "stop / that's enough / quiet, call stop(). Use "
    "set_expression(emotion) to react with a full-body emotion animation (happy, curious, "
    "surprised, sad, proud, thinking…); look_around() when curious or "
    "asked to look; look_and_describe() whenever asked what you see or what's in front of "
    "you (you really can see through your camera); get_current_time() for the time or date "
    "(state exactly what it returns, don't garble it); get_weather(location) for weather, "
    "temperature, or forecast; web_search(query) for news, current events, sports, prices, or "
    "ANY fact you might not know or that could be out of date -- search instead of guessing; "
    "set_volume(level) to get louder or quieter; set_reminder(minutes, about) for timers and "
    "reminders; take_photo() when asked for a picture or selfie. "
    "When the user says go to sleep, stand "
    "by, be quiet for a while, or goodbye, call sleep() and give a short goodnight. Only call "
    "ignore() if the speech clearly was not meant for you and needs no reply."
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
        self.ctx = {"ignored": False, "sleep": False}
        self.tools = tools_mod.build_tools(body, self.ctx, cfg) if body is not None else []
        self.ignored = False  # did the model choose to stay silent this turn?
        self.slept = False    # did the model call sleep() this turn?

    async def ask(self, text: str) -> str:
        """User text -> reply text (''=no reply). Sets self.ignored if it chose silence."""
        self.ctx["ignored"] = False
        self.ctx["sleep"] = False
        self.ignored = False
        self.slept = False
        self.history.append(types.Content(role="user", parts=[types.Part(text=text)]))
        reply = None
        for attempt in range(3):
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
                break
            except Exception as e:  # don't let a brain hiccup kill the loop
                msg = str(e)
                transient = "503" in msg or "UNAVAILABLE" in msg or "overloaded" in msg.lower()
                if attempt < 2 and transient:
                    await asyncio.sleep(0.6)  # Gemini briefly unavailable -> retry
                    continue
                print(f"[brain] error: {e}", flush=True)
                self.history.pop()
                return ""

        self.slept = bool(self.ctx.get("sleep"))
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
