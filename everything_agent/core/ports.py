"""The PORTS -- the interfaces the agent core depends on.

This is the heart of the architecture. The core (agent.py) only ever talks to
these abstract interfaces, never to a concrete library. Each one has one or more
*adapters* (concrete implementations) in its own package, chosen by config.

To support a new backend (Whisper, Parakeet, Cartesia, Mem0, ElevenLabs...),
you write an adapter that subclasses one of these -- you never change this file.

Think "ports & adapters" (hexagonal architecture): ports here, adapters around
the edges. Every adapter has the same constructor shape: __init__(self, config,
ctx), where `ctx` is the shared AgentContext (see core/context.py).
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional


# ---- HEAR ----
class WakeWord(abc.ABC):
    @abc.abstractmethod
    async def wait(self) -> bool:
        """Block until the wake word is heard; return True when woken."""


class STT(abc.ABC):
    @abc.abstractmethod
    async def listen(self) -> Optional[str]:
        """Return the transcribed utterance, or None if input has ended."""


# ---- DECIDE ----
@dataclass
class Decision:
    action: str          # "instant" | "agent" | "ignore"
    reply: str = ""


class Router(abc.ABC):
    @abc.abstractmethod
    async def decide(self, text: str, memory_context: str = "") -> Decision:
        """Fast routing: answer instantly, escalate to the agent, or ignore."""


class AgentBrain(abc.ABC):
    @abc.abstractmethod
    async def run(self, text: str, memory_context: str = "") -> str:
        """Deliberate, tool-using turn. Returns the reply text."""


# ---- EXPRESS ----
class TTS(abc.ABC):
    @abc.abstractmethod
    def speak(self, text: str) -> None:
        """Say the text aloud (and optionally move expressively)."""


# ---- BODY ----
class Robot(abc.ABC):
    @abc.abstractmethod
    def connect(self) -> None: ...
    @abc.abstractmethod
    def disconnect(self) -> None: ...
    @abc.abstractmethod
    def move_head(self, yaw=0.0, pitch=0.0, roll=0.0, duration=0.5) -> None: ...
    @abc.abstractmethod
    def set_antennas(self, left=0.0, right=0.0) -> None: ...
    @abc.abstractmethod
    def reset(self) -> None: ...


# ---- MIND ----
class Memory(abc.ABC):
    @abc.abstractmethod
    def remember(self, key: str, value: str) -> None:
        """Store a long-term fact."""
    @abc.abstractmethod
    def add_turn(self, user: str, reply: str) -> None:
        """Record one exchange in short-term memory."""
    @abc.abstractmethod
    def context(self) -> str:
        """Return a short text block of what's known, fed to the brains."""
