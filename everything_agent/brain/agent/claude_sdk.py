"""Claude Agent SDK adapter -- the real deliberate brain.

Wraps each local module Action as an SDK tool (adding the approval gate for
`sensitive` ones) AND connects external MCP servers from config -- the path for
Telegram, Swiggy, your other agent "Clio", etc. Needs claude-agent-sdk, Python
3.10+, Node + Claude Code CLI, and ANTHROPIC_API_KEY.

Reads the module Actions and Approval gate from the shared context.
"""
from __future__ import annotations

import logging

from ...core.ports import AgentBrain

log = logging.getLogger("everything_agent.brain.agent")


class ClaudeSdkAgent(AgentBrain):
    def __init__(self, config, ctx):
        import claude_agent_sdk           # raises if not installed -> see config
        self._sdk = claude_agent_sdk
        self.model = config.get("model", "claude-opus-4-8")
        self.mcp_servers_cfg = config.get("mcp_servers", {}) or {}
        # auth: "api_key" (default) uses ANTHROPIC_API_KEY; "login" uses the
        # CLAUDE_CODE_OAUTH_TOKEN from `claude setup-token` / `claude login`,
        # which unlocks the models your subscription allows (e.g. Opus) regardless
        # of the key's limits. The CLI prefers ANTHROPIC_API_KEY when present, so
        # in login mode we drop it from the env just for the SDK call below (the
        # router still uses it in-process; calls are sequential so this is safe).
        self.auth = config.get("auth", "api_key")
        # Personality: config override, else the shared default persona.
        from ...persona import DEFAULT_PERSONA
        self.personality = config.get("personality") or DEFAULT_PERSONA
        self.actions = ctx.actions
        self.approval = ctx.approval

    def _wrap(self, action):
        approval = self.approval

        @self._sdk.tool(action.name, action.description, action.params)
        async def _tool(args, _action=action):
            if _action.sensitive and not await approval.confirm(f"{_action.name}({args})"):
                return {"content": [{"type": "text", "text": "User declined."}]}
            result = await _action.handler(args)
            return {"content": [{"type": "text", "text": result}]}

        return _tool

    async def run(self, text: str, memory_context: str = "") -> str:
        sdk = self._sdk
        mcp_servers, allowed = {}, []

        # 1. local actions -> an in-process MCP server
        local_tools = [self._wrap(a) for a in self.actions]
        if local_tools:
            mcp_servers["local"] = sdk.create_sdk_mcp_server(
                name="local", version="0.1.0", tools=local_tools)
            allowed += [f"mcp__local__{a.name}" for a in self.actions]

        # 2. external MCP servers from config (Telegram, Clio, Swiggy, ...)
        for name, spec in self.mcp_servers_cfg.items():
            mcp_servers[name] = spec
            allowed.append(f"mcp__{name}__*")

        system_prompt = self.personality
        if memory_context:
            system_prompt += "\n\nWhat you know:\n" + memory_context
        options = sdk.ClaudeAgentOptions(
            model=self.model,
            system_prompt=system_prompt,
            mcp_servers=mcp_servers,
            allowed_tools=allowed,
        )

        # In login mode, drop ANTHROPIC_API_KEY for this call so the CLI uses the
        # OAuth token (CLAUDE_CODE_OAUTH_TOKEN); restore it afterwards.
        import os
        saved_key = os.environ.pop("ANTHROPIC_API_KEY", None) if self.auth == "login" else None
        try:
            final = ""
            async for message in sdk.query(prompt=text, options=options):
                chunk = getattr(message, "result", None) or getattr(message, "text", None)
                if chunk:
                    final = chunk
            return final or "(done)"
        finally:
            if saved_key is not None:
                os.environ["ANTHROPIC_API_KEY"] = saved_key
