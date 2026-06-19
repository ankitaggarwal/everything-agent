"""Web search via Tavily -- current info, news, live facts. Free tier (no card).

Exposes a `web_search` tool the brain calls for anything beyond its training
data ("what's happening in the world", today's news, current prices). Tavily's
`include_answer` returns a one-paragraph synthesis that's ideal to speak aloud,
with source titles appended. Needs TAVILY_API_KEY; without it the tool returns a
friendly "I can't look that up right now" so the robot is never silent.
"""
from __future__ import annotations

import logging
import os
from typing import List

from ...core.module import Action, Module

log = logging.getLogger("everything_agent.modules.web_search")

_ENDPOINT = "https://api.tavily.com/search"


class WebSearchModule(Module):
    name = "web_search"

    def setup(self, robot, providers, memory, config) -> None:
        cfg = (config or {}).get("web_search", {}) or {}
        self.api_key = os.environ.get("TAVILY_API_KEY", "")
        self.max_results = int(cfg.get("max_results", 5))
        if not self.api_key:
            log.info("web_search module loaded but TAVILY_API_KEY not set (degraded)")

    def actions(self) -> List[Action]:
        async def web_search(args):
            query = (args.get("query") or "").strip()
            if not query:
                return "What would you like me to look up?"
            if not self.api_key:
                return ("I can't search the web right now -- there's no search key "
                        "set up yet. I can still help from what I already know.")
            return await self._search(query)

        return [Action(
            "web_search",
            "Search the web for current information, news, or live facts beyond "
            "your training data. 'query' is what to look up.",
            web_search, params={"query": str},
        )]

    async def _search(self, query: str) -> str:
        import httpx

        try:
            async with httpx.AsyncClient(timeout=10.0) as http:
                r = await http.post(_ENDPOINT, json={
                    "api_key": self.api_key, "query": query,
                    "max_results": self.max_results, "include_answer": True,
                    "search_depth": "basic",
                })
                data = r.json() or {}
        except Exception as e:  # noqa: BLE001
            log.warning("tavily search failed: %s", e)
            return f"I tried to look that up but couldn't reach the search service just now."

        answer = (data.get("answer") or "").strip()
        results = data.get("results") or []
        if answer:
            sources = ", ".join(x.get("title", "") for x in results[:2] if x.get("title"))
            return answer + (f" (Sources: {sources}.)" if sources else "")
        if results:
            top = results[0]
            return f"{top.get('title', '')}: {(top.get('content') or '')[:280]}"
        return f"I searched for '{query}' but didn't find a clear answer."
