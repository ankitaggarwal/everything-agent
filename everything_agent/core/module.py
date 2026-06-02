"""The Module contract -- the ONE thing you extend to grow the robot.

A *module* is one self-contained capability (idle behavior, telling the time,
sending a Telegram message, ...). A module can contribute three independent
things; implement only the ones you need:

  actions()  -> the real capabilities the agent brain can invoke. Each Action is
                plain async Python (NO Claude SDK import!). The brain turns them
                into Agent-SDK tools automatically. This keeps modules trivial.
  perceive() -> a short text snapshot of what this module senses right now. The
                router reads all perceptions to help decide what to do.
  tick()     -> autonomous idle behavior when nobody is talking to the robot
                (e.g. a gentle look-around).

To add a capability you write actions(); you almost never touch the core.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional


@dataclass
class Action:
    """One callable capability exposed to the brain.

    name        : tool name the LLM will call, e.g. "get_time".
    description : what it does -- the LLM reads this to decide when to use it.
    params      : {arg_name: type} schema, e.g. {"direction": str}. {} = no args.
    handler     : async function (args: dict) -> str. Does the work, returns text.
    sensitive   : if True, the Approval gate asks the user before running it
                  (use for anything that spends money or can't be undone).
    """
    name: str
    description: str
    handler: Callable[[Dict[str, Any]], Awaitable[str]]
    params: Dict[str, Any] = field(default_factory=dict)
    sensitive: bool = False


class Module:
    name: str = "module"

    def setup(self, robot, providers: dict, memory, config: dict) -> None:
        """Wire up the module. Receives the Robot interface, shared providers,
        the Memory store, and the full config dict."""

    def actions(self) -> List[Action]:
        """Capabilities the brain can call. Default: none."""
        return []

    async def perceive(self) -> Optional[str]:
        """Short situation string, or None if nothing to report."""
        return None

    async def tick(self) -> None:
        """Optional idle/background behavior, run when the robot is not engaged."""
