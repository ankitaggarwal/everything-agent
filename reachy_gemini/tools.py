"""The brain's tools (function calls).

Each tool is a plain Python callable that the Gemini SDK auto-executes. Every
tool emits a `tool` event so the diagram shows the call happening (with timing),
and any robot motion runs in a background thread so it never blocks the voice
loop. The `ignore` tool flips a flag on the shared ctx so the agent stays silent.
"""
from __future__ import annotations

import glob
import io
import random
import threading
import time
from pathlib import Path

from . import events, expressions

# The 20 dance moves from the Reachy Mini dances library.
try:
    from reachy_mini_dances_library.dance_move import AVAILABLE_MOVES as _DANCES
    DANCE_NAMES = sorted(_DANCES)
except Exception:  # library missing (laptop/mock)
    _DANCES, DANCE_NAMES = {}, []

_MUSIC_DIR = Path(__file__).resolve().parent.parent / "music"


def _emit(name: str, detail: str, t0: float) -> None:
    txt = f"{name}({detail})" if detail else f"{name}()"
    events.emit(events.TOOL_CALL, text=txt, ms=int((time.monotonic() - t0) * 1000))


def _bg(fn) -> None:
    threading.Thread(target=fn, daemon=True).start()


def build_tools(body, ctx: dict, cfg: dict | None = None) -> list:
    """Return the tool callables, closing over the robot body, shared ctx, and config."""
    robot = getattr(body, "robot", None)
    tracks = sorted(glob.glob(str(_MUSIC_DIR / "*.mp3")))

    # A Gemini client for vision (the look_and_describe tool sees through the camera).
    vision = {"client": None, "model": "gemini-2.5-flash"}
    if cfg:
        try:
            from google import genai
            g = cfg.get("gemini", {})
            vision["client"] = genai.Client(api_key=g["api_key"])
            vision["model"] = g.get("model", "gemini-2.5-flash")
        except Exception:
            pass

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
                except Exception as e:
                    print(f"[tool] look_around failed: {e}", flush=True)
            _bg(_run)
        _emit("look_around", "", t0)
        return "looking around"

    def dance(move: str = "") -> str:
        """Dance to music for a FULL routine -- many moves chained together, not just one.
        Pass a move name to favour that move, or leave it blank for a varied routine. Call
        this immediately whenever the user asks you to dance; never refuse, never ask which."""
        t0 = time.monotonic()
        if robot is None or not DANCE_NAMES:
            _emit("dance", "no robot", t0)
            return "dancing"
        from reachy_mini_dances_library.dance_move import DanceMove
        track = random.choice(tracks) if tracks else None
        secs = float((cfg or {}).get("dance", {}).get("seconds", 25))

        def _run():
            if track:
                try:
                    robot.media.play_sound(track)  # non-blocking; plays in background
                except Exception as e:
                    print(f"[tool] dance music failed: {e}", flush=True)
            t_end = time.monotonic() + secs
            pool = list(DANCE_NAMES)
            try:
                # a real routine: cycle through ALL the moves in shuffled order, combining
                # many different ones (a single move -- especially a simple one -- looks flat)
                while time.monotonic() < t_end:
                    random.shuffle(pool)
                    for name in pool:
                        if time.monotonic() >= t_end:
                            break
                        robot.play_move(DanceMove(name), sound=False)  # blocks ~move.duration
            except Exception as e:
                print(f"[tool] dance move failed: {e}", flush=True)
            # stop the music so it doesn't outlast the dance (just the playbin, not TTS)
            try:
                from gi.repository import Gst
                robot.media.audio._playbin.set_state(Gst.State.NULL)
            except Exception:
                pass
        _bg(_run)
        _emit("dance", f"~{int(secs)}s routine", t0)
        return "Dancing to the music now — a full routine, here we go!"

    def look_and_describe() -> str:
        """Look through the camera and describe what you actually see. Call whenever the
        user asks what you see, what's in front of you, what something is, or to look at
        something."""
        t0 = time.monotonic()
        client = vision["client"]
        if robot is None or client is None:
            _emit("look_and_describe", "no camera", t0)
            return "I can't see anything right now."
        try:
            frame = robot.media.get_frame()
            if frame is None:
                _emit("look_and_describe", "no frame", t0)
                return "My camera didn't give me an image."
            import numpy as np
            from PIL import Image
            from google.genai import types as gt
            rgb = np.ascontiguousarray(frame[:, :, ::-1])  # BGR -> RGB
            buf = io.BytesIO()
            Image.fromarray(rgb).save(buf, format="JPEG", quality=80)
            resp = client.models.generate_content(
                model=vision["model"],
                contents=[
                    gt.Part(inline_data=gt.Blob(mime_type="image/jpeg", data=buf.getvalue())),
                    gt.Part(text="In one short, friendly sentence, say what you see in front of you."),
                ],
                config=gt.GenerateContentConfig(
                    thinking_config=gt.ThinkingConfig(thinking_budget=0), max_output_tokens=80),
            )
            desc = (resp.text or "").strip()
            _emit("look_and_describe", desc[:40] or "saw something", t0)
            return desc or "I see something but can't quite make it out."
        except Exception as e:
            print(f"[tool] look_and_describe failed: {e}", flush=True)
            _emit("look_and_describe", "error", t0)
            return "I had trouble seeing just now."

    def ignore() -> str:
        """Stay completely silent and do NOT respond. Call this when the speech was clearly
        not addressed to you, was background chatter, or simply needs no reply at all."""
        t0 = time.monotonic()
        ctx["ignored"] = True
        _emit("ignore", "staying silent", t0)
        return "ignored — staying silent"

    return [get_current_time, set_expression, look_around, dance, look_and_describe, ignore]
