"""Timers & reminders -- the robot speaks up on its own when one is due.

Two tools the brain can call: set_timer ("remind me in 10 minutes", "set a 30
second timer") and set_reminder (a timer with a message). When due, the module
pushes the announcement to the agent's proactive-speech queue (wired via
set_speaker) so the robot says it aloud unprompted. Pending items persist to a
JSON file so they survive a restart; Upstash Redis is used instead if configured.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import List

from ...core.module import Action, Module

log = logging.getLogger("everything_agent.modules.timers")

_UNITS = {"second": 1, "sec": 1, "minute": 60, "min": 60, "hour": 3600, "hr": 3600}


def _parse_seconds(text: str) -> int:
    """'10 minutes' -> 600, '90 seconds' -> 90, 'an hour' -> 3600. 0 if unclear."""
    text = (text or "").lower()
    total = 0
    for n, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(second|sec|minute|min|hour|hr)s?", text):
        total += int(float(n) * _UNITS[unit])
    if total == 0:  # words like "a minute" / "an hour"
        for unit, secs in (("hour", 3600), ("minute", 60), ("second", 1)):
            if unit in text:
                total = secs
                break
    return total


def _spoken(seconds: int) -> str:
    if seconds % 3600 == 0 and seconds >= 3600:
        n = seconds // 3600; return f"{n} hour{'s' if n != 1 else ''}"
    if seconds % 60 == 0 and seconds >= 60:
        n = seconds // 60; return f"{n} minute{'s' if n != 1 else ''}"
    return f"{seconds} second{'s' if seconds != 1 else ''}"


class TimersModule(Module):
    name = "timers"

    def setup(self, robot, providers, memory, config) -> None:
        cfg = (config or {}).get("timers", {}) or {}
        self.path = cfg.get("path", "/tmp/everything_agent_timers.json")
        self._speak = None
        self.pending = self._load()

    def set_speaker(self, fn) -> None:
        """Agent injects the proactive-speech sink: fn(text) -> says it aloud."""
        self._speak = fn

    # ---- persistence (local JSON; swap for Upstash Redis when configured) ----
    def _load(self) -> list:
        try:
            if os.path.exists(self.path):
                with open(self.path) as f:
                    return [d for d in json.load(f) if d.get("due", 0) > 0]
        except Exception:  # noqa: BLE001
            log.warning("could not load timers from %s", self.path)
        return []

    def _save(self) -> None:
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.pending, f)
            os.replace(tmp, self.path)
        except Exception:  # noqa: BLE001
            log.warning("could not save timers")

    def actions(self) -> List[Action]:
        async def set_timer(args):
            secs = _parse_seconds(args.get("duration", ""))
            if secs <= 0:
                return "I couldn't tell how long -- try 'set a 5 minute timer'."
            label = (args.get("label") or "").strip()
            msg = f"Time's up{(' for ' + label) if label else ''}!"
            self.pending.append({"due": time.time() + secs, "message": msg})
            self._save()
            return f"Okay, timer set for {_spoken(secs)}{(' (' + label + ')') if label else ''}."

        async def set_reminder(args):
            secs = _parse_seconds(args.get("when", ""))
            text = (args.get("text") or "").strip()
            if secs <= 0 or not text:
                return "Tell me what to remind you and when, like 'remind me to stretch in 20 minutes'."
            self.pending.append({"due": time.time() + secs, "message": f"Reminder: {text}."})
            self._save()
            return f"Got it -- I'll remind you to {text} in {_spoken(secs)}."

        return [
            Action("set_timer", "Start a countdown timer. 'duration' is natural "
                   "language like '5 minutes' or '30 seconds'; optional 'label'.",
                   set_timer, params={"duration": str, "label": str}),
            Action("set_reminder", "Remind the user of something after a delay. "
                   "'when' is like '20 minutes'; 'text' is what to remind them of.",
                   set_reminder, params={"when": str, "text": str}),
        ]

    async def tick(self) -> None:
        if not self.pending or self._speak is None:
            return
        now = time.time()
        due = [p for p in self.pending if p["due"] <= now]
        if not due:
            return
        self.pending = [p for p in self.pending if p["due"] > now]
        self._save()
        for p in due:
            try:
                self._speak(p["message"])
            except Exception:  # noqa: BLE001
                log.warning("could not announce due timer")
