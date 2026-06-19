"""A dead-simple event bus so you can SEE what the agent is doing.

Every stage of the loop (listening, sending audio, the model thinking, speaking,
tool-call-or-not, barge-in) calls emit(). By default it prints a tidy line to the
console; the diagram and any future web UI can subscribe via add_listener().
"""
from __future__ import annotations

from typing import Callable

# Stages, in the order a turn flows through them. Mirrors docs/diagram.html.
# Pipeline: you speak -> (VAD) -> Cartesia STT -> Gemini text -> Cartesia TTS -> speaker.
LISTENING = "listening"       # mic armed, waiting for you (idle)
HEARING = "hearing"           # local VAD opened: the mic is capturing your voice
TRANSCRIBING = "transcribing"  # utterance sent to Cartesia STT
TRANSCRIBED = "transcribed"   # STT returned your words as text
THINKING = "thinking"         # your text sent to Gemini; waiting for the reply
TOOL_CALL = "tool"            # the model called a tool (time / expression / dance / ...)
REPLY = "reply"               # Gemini's reply text is in
SPEAKING = "speaking"         # Cartesia TTS audio is playing back
DONE = "done"                 # turn complete
ERROR = "error"               # something in the pipeline failed (debug aid)

_ICONS = {
    LISTENING: "🎙️ ", HEARING: "👂", TRANSCRIBING: "✍️ ", TRANSCRIBED: "📝",
    THINKING: "🧠", TOOL_CALL: "🔧", REPLY: "💬", SPEAKING: "🔊", DONE: "✓", ERROR: "⚠️ ",
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
