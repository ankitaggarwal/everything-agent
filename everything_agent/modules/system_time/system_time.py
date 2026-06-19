"""Example module #2: system time -- your first "real" tool (ROADMAP Phase 2).

This is the template to copy for any new capability. It exposes two actions the
agent brain can call; neither is sensitive, so neither needs approval.

To add YOUR next tool (Telegram, Spotify, ...): copy this folder, write the
actions, register the class in core/registry.py, and list it in config.yaml.
Nothing else changes.
"""
from __future__ import annotations

from datetime import datetime
from typing import List

from ...core.module import Action, Module


class SystemTimeModule(Module):
    name = "system_time"

    def setup(self, robot, providers, memory, config) -> None:
        pass  # no setup needed

    def actions(self) -> List[Action]:
        async def get_time(args):
            return "It's " + datetime.now().strftime("%-I:%M %p") + "."

        async def get_date(args):
            return "Today is " + datetime.now().strftime("%A, %B %-d, %Y") + "."

        return [
            Action("get_time", "Get the current local time of day", get_time),
            Action("get_date", "Get today's date", get_date),
        ]

    async def perceive(self) -> str:
        # Surface the time as an always-on perception so even the instant router
        # answers "what time is it?" correctly, instead of the Haiku reflex
        # bluffing "I don't have the time" (it can't call the get_time tool).
        now = datetime.now()
        return ("The current time is " + now.strftime("%-I:%M %p")
                + " on " + now.strftime("%A, %B %-d, %Y") + ".")
