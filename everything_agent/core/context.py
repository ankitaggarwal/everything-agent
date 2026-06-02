"""AgentContext -- the shared bag of things every adapter might need.

Passed to every adapter's constructor as `ctx`. An adapter pulls out only what
it uses (e.g. the Cartesia TTS adapter reads ctx.providers["cartesia"] and
ctx.robot; the Whisper STT adapter needs nothing). This keeps every adapter's
constructor identical -- __init__(self, config, ctx) -- so the generic loader
can build any of them the same way.

It is filled in stages during EverythingAgent.start() (providers → robot →
memory → approval → actions), because later parts depend on earlier ones.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentContext:
    config: dict
    providers: Dict[str, Any] = field(default_factory=dict)
    robot: Optional[Any] = None        # a ports.Robot
    memory: Optional[Any] = None       # a ports.Memory
    approval: Optional[Any] = None     # core.approval.Approval
    actions: List[Any] = field(default_factory=list)  # module.Action list
