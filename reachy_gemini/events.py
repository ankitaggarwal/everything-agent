"""A dead-simple event bus so you can SEE what the agent is doing.

Every stage of the loop (listening, sending audio, the model thinking, speaking,
tool-call-or-not, barge-in) calls emit(). By default it prints a tidy line to the
console; the diagram and any future web UI can subscribe via add_listener().
"""
from __future__ import annotations

from typing import Callable

# Stages, in the order a turn flows through them. Mirrors docs/diagram.html.
LISTENING = "listening"     # mic open, waiting for you
HEARING = "hearing"         # your speech is being transcribed
THINKING = "thinking"       # your turn ended; model is deciding
TOOL_CALL = "tool_call"     # model asked to run a tool (action path)
NO_TOOL = "no_tool"         # model chose to just talk -> path ends ("dies")
SPEAKING = "speaking"       # model audio is playing back
INTERRUPTED = "interrupted"  # you spoke over it (barge-in)
DONE = "done"               # turn complete

_ICONS = {
    LISTENING: "🎙️ ", HEARING: "👂", THINKING: "🧠", TOOL_CALL: "🔧",
    NO_TOOL: "💬", SPEAKING: "🔊", INTERRUPTED: "✋", DONE: "✓",
}

_listeners: list[Callable[[str, dict], None]] = []


def add_listener(fn: Callable[[str, dict], None]) -> None:
    _listeners.append(fn)


def emit(stage: str, **data) -> None:
    text = data.get("text", "")
    icon = _ICONS.get(stage, "·")
    line = f"{icon} {stage:<11}" + (f" {text}" if text else "")
    print(line, flush=True)
    for fn in _listeners:
        try:
            fn(stage, data)
        except Exception:
            pass
