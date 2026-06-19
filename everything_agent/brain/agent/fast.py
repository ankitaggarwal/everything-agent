"""Fast agent brain -- low-latency tool calling straight through the Anthropic API.

Unlike `claude_sdk` (which spawns the Claude Code CLI subprocess each turn and
adds seconds of latency), this makes a direct Messages call with the local module
actions exposed as tools, looping on `tool_use` until a final answer. Ideal for
snappy spoken turns. Trade-off: no MCP servers (use claude_sdk when you need
those). Uses the persona + the approval gate; defaults to Haiku for speed.
"""
from __future__ import annotations

import logging

from ...core.ports import AgentBrain

log = logging.getLogger("everything_agent.brain.agent")

_MAX_TOOL_ROUNDS = 4   # safety cap on tool round-trips per turn


def _anthropic_type(py_type) -> str:
    if py_type in (int, float):
        return "number"
    if py_type is bool:
        return "boolean"
    return "string"


class FastAgent(AgentBrain):
    def __init__(self, config, ctx):
        cfg = config or {}
        self.model = cfg.get("model", "claude-haiku-4-5")
        self.max_tokens = int(cfg.get("max_tokens", 256))
        # Per API call -- a spoken turn shouldn't hang for the SDK's default 10 min.
        self.timeout = float(cfg.get("timeout", 30.0))
        from ...persona import DEFAULT_PERSONA
        self.personality = cfg.get("personality") or DEFAULT_PERSONA
        self.actions = ctx.actions
        self.approval = ctx.approval
        self._by_name = {a.name: a for a in self.actions}
        self._tools = self._build_tools()
        # Anthropic's server-side web search lets the robot answer "what's
        # happening in the world" / current facts. Haiku doesn't support the
        # dynamic-filtering _20260209 build, so use the stable _20250305 one.
        self.web_search = bool(cfg.get("web_search", False))
        self._web_tool = {"type": "web_search_20250305", "name": "web_search",
                          "max_uses": int(cfg.get("web_search_max_uses", 3))}
        self._client = None   # reused across turns (keeps the HTTP pool warm)

    def _get_client(self):
        if self._client is None:
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic(timeout=self.timeout)
        return self._client

    def _build_tools(self):
        tools = []
        for a in self.actions:
            props, required = {}, []
            for pname, ptype in (a.params or {}).items():
                props[pname] = {"type": _anthropic_type(ptype)}
                required.append(pname)
            tools.append({
                "name": a.name,
                "description": a.description,
                "input_schema": {"type": "object", "properties": props,
                                 "required": required},
            })
        return tools

    async def _run_tool(self, name: str, args: dict) -> str:
        action = self._by_name.get(name)
        if action is None:
            return "(unknown tool)"
        if action.sensitive and not await self.approval.confirm(f"{name}({args})"):
            return "User declined."
        try:
            # str() in case a handler strays from the "return text" contract --
            # a non-string tool_result would fail the whole API call.
            return str(await action.handler(args or {}))
        except Exception as e:  # noqa: BLE001
            log.warning("tool %s failed: %s", name, e)
            return f"(tool error: {e})"

    async def run(self, text: str, memory_context: str = "") -> str:
        client = self._get_client()
        system = self.personality
        if memory_context:
            system += "\n\nWhat you know:\n" + memory_context
        messages = [{"role": "user", "content": text}]
        # Local module tools + (optionally) Anthropic's server-side web search.
        tools = self._tools + ([self._web_tool] if self.web_search else [])
        final = ""
        web_disabled = False

        for _ in range(_MAX_TOOL_ROUNDS):
            try:
                msg = await client.messages.create(
                    model=self.model, max_tokens=self.max_tokens, system=system,
                    tools=tools, messages=messages,
                )
            except Exception as e:  # noqa: BLE001
                # If the workspace rejects web_search, drop it and retry once so a
                # bad entitlement degrades to "no live info" instead of silence.
                if self.web_search and not web_disabled and "web_search" in str(e):
                    log.warning("web_search rejected (%s) -- retrying without it", e)
                    tools = self._tools
                    web_disabled = True
                    continue
                raise
            text_parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
            if text_parts:
                final = " ".join(t for t in text_parts if t).strip()
            # Server-side tools (web_search) hit the per-turn cap -> resume by
            # re-sending; the server picks up where it left off.
            if msg.stop_reason == "pause_turn":
                messages.append({"role": "assistant", "content": msg.content})
                continue
            # Only CLIENT-side tool_use needs us to run something; server_tool_use
            # (web_search) is executed by the API and returned inline.
            tool_uses = [b for b in msg.content if getattr(b, "type", None) == "tool_use"]
            if msg.stop_reason != "tool_use" or not tool_uses:
                break
            messages.append({"role": "assistant", "content": msg.content})
            results = []
            for tu in tool_uses:
                out = await self._run_tool(tu.name, tu.input)
                results.append({"type": "tool_result", "tool_use_id": tu.id,
                                "content": out})
            messages.append({"role": "user", "content": results})

        # Never return silence: if the model produced no text (e.g. it only made
        # tool calls, or hit the round cap), give a warm, honest fallback so the
        # robot always says *something* back.
        return final or ("Hmm, I'm not totally sure about that one -- "
                         "but ask me another way and I'll give it a real go.")
