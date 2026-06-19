"""The brain's tools (function calls).

Each tool is a plain Python callable that the Gemini SDK auto-executes. Every
tool emits a `tool` event so the diagram shows the call happening (with its
timing), and any robot motion runs in a background thread so it never blocks the
voice loop. The `ignore` tool flips a flag on the shared ctx so the agent stays
completely silent for that turn.
"""
from __future__ import annotations

import threading
import time

from . import events, expressions

# The 20 dance moves from the Reachy Mini dances library.
try:
    from reachy_mini_dances_library.dance_move import AVAILABLE_MOVES as _DANCES
    DANCE_NAMES = sorted(_DANCES)
except Exception:  # library missing (laptop/mock)
    _DANCES, DANCE_NAMES = {}, []


def _emit(name: str, detail: str, t0: float) -> None:
    txt = f"{name}({detail})" if detail else f"{name}()"
    events.emit(events.TOOL_CALL, text=txt, ms=int((time.monotonic() - t0) * 1000))


def _bg(fn) -> None:
    threading.Thread(target=fn, daemon=True).start()


def build_tools(body, ctx: dict) -> list:
    """Return the list of tool callables, closing over the robot body + shared ctx."""
    robot = getattr(body, "robot", None)

    def get_current_time() -> str:
        """Get the current local date and time. Call whenever the user asks the time or date."""
        t0 = time.monotonic()
        now = time.strftime("%A, %B %d %Y, %I:%M %p").replace(" 0", " ")
        _emit("get_current_time", now, t0)
        return now

    def set_expression(emotion: str) -> str:
        """Show an emotion with the robot's antennas. emotion is one of: happy, excited,
        sad, curious, surprised, yes, no, neutral. Use it naturally to react to the user."""
        t0 = time.monotonic()
        name = expressions.perform(body, emotion)
        _emit("set_expression", name, t0)
        return f"expressed {name}"

    def look_around() -> str:
        """Rotate the head and body to look around the room. Use when curious or asked to look around."""
        t0 = time.monotonic()
        if robot is not None:
            def _run():
                try:
                    for yaw in (0.6, -0.6, 0.0):
                        robot.goto_target(body_yaw=float(yaw), duration=0.8)
                        time.sleep(0.85)
                except Exception:
                    pass
            _bg(_run)
        _emit("look_around", "", t0)
        return "looking around"

    def dance(move: str) -> str:
        """Make the robot perform a named dance move. Use when asked to dance or to celebrate."""
        t0 = time.monotonic()
        name = (move or "").strip().lower()
        if name not in _DANCES and DANCE_NAMES:
            name = "simple_nod" if "simple_nod" in _DANCES else DANCE_NAMES[0]
        if robot is not None and name in _DANCES:
            from reachy_mini_dances_library.dance_move import DanceMove
            def _run():
                try:
                    robot.play_move(DanceMove(name))
                except Exception:
                    pass
            _bg(_run)
        _emit("dance", name, t0)
        return f"dancing {name}"

    if DANCE_NAMES:
        dance.__doc__ = ("Make the robot perform a dance move. `move` must be one of: "
                         + ", ".join(DANCE_NAMES) + ". Use when asked to dance or to celebrate.")

    def ignore() -> str:
        """Stay completely silent and do NOT respond. Call this when the speech was clearly
        not addressed to you, was background chatter, or simply needs no reply at all."""
        t0 = time.monotonic()
        ctx["ignored"] = True
        _emit("ignore", "staying silent", t0)
        return "ignored — staying silent"

    return [get_current_time, set_expression, look_around, dance, ignore]
