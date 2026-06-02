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
        from ...persona import DEFAULT_PERSONA
        self.personality = cfg.get("personality") or DEFAULT_PERSONA
        self.actions = ctx.actions
        self.approval = ctx.approval
        self._by_name = {a.name: a for a in self.actions}
        self._tools = self._build_tools()
        self._client = None   # reused across turns (keeps the HTTP pool warm)

    def _get_client(self):
        if self._client is None:
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic()
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
            return await action.handler(args or {})
        except Exception as e:  # noqa: BLE001
            log.warning("tool %s failed: %s", name, e)
            return f"(tool error: {e})"

    async def run(self, text: str, memory_context: str = "") -> str:
        client = self._get_client()
        system = self.personality
        if memory_context:
            system += "\n\nWhat you know:\n" + memory_context
        messages = [{"role": "user", "content": text}]
        final = ""

        for _ in range(_MAX_TOOL_ROUNDS):
            msg = await client.messages.create(
                model=self.model, max_tokens=self.max_tokens, system=system,
                tools=self._tools, messages=messages,
            )
            text_parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
            if text_parts:
                final = " ".join(t for t in text_parts if t).strip()
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

        return final or "(done)"
