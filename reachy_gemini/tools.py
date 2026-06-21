"""The brain's tools (function calls).

Each tool is a plain Python callable that the Gemini SDK auto-executes. Every
tool emits a `tool` event so the diagram shows the call happening (with timing),
and any robot motion runs in a background thread so it never blocks the voice
loop. The `ignore` tool flips a flag on the shared ctx so the agent stays silent.
"""
from __future__ import annotations

import glob
import io
import json
import os
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

# The punchier moves -- used for higher-BPM tracks (style matching).
_ENERGETIC = {"headbanger_combo", "polyrhythm_combo", "dizzy_spin", "interwoven_spirals",
              "jackson_square", "groovy_sway_and_roll", "stumble_and_recover", "grid_snap",
              "chicken_peck", "side_to_side_sway", "sharp_side_tilt"}

_MUSIC_DIR = Path(__file__).resolve().parent.parent / "music"

# WMO weather codes -> short spoken descriptions (Open-Meteo's `weather_code`).
_WMO = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "foggy", 51: "drizzling", 53: "drizzling", 55: "drizzling",
    56: "freezing drizzle", 57: "freezing drizzle", 61: "raining lightly",
    63: "raining", 65: "raining heavily", 66: "freezing rain", 67: "freezing rain",
    71: "snowing lightly", 73: "snowing", 75: "snowing heavily", 77: "snow grains",
    80: "with rain showers", 81: "with rain showers", 82: "with heavy showers",
    85: "with snow showers", 86: "with snow showers", 95: "thunderstorms",
    96: "thunderstorms with hail", 99: "thunderstorms with hail",
}


def _emit(name: str, detail: str, t0: float) -> None:
    txt = f"{name}({detail})" if detail else f"{name}()"
    events.emit(events.TOOL_CALL, text=txt, ms=int((time.monotonic() - t0) * 1000))


def _bg(fn) -> None:
    threading.Thread(target=fn, daemon=True).start()


def _goto_sleep(robot) -> None:
    """A gentle 'resting' posture: head dips, antennas relax. Motors stay ENABLED so the
    head holds the pose (we do NOT disable motors -- that's what makes the head flop)."""
    try:
        from reachy_mini.utils import create_head_pose
        robot.goto_target(head=create_head_pose(pitch=-12, degrees=True),
                          antennas=[-0.3, -0.3], body_yaw=0.0, duration=1.2)
    except Exception as e:
        print(f"[tool] goto_sleep failed: {e}", flush=True)


def build_tools(body, ctx: dict, cfg: dict | None = None, memory=None) -> list:
    """Return the tool callables, closing over the robot body, shared ctx, config, memory."""
    robot = getattr(body, "robot", None)
    tracks = sorted(glob.glob(str(_MUSIC_DIR / "*.mp3")))
    try:
        meta = json.loads((_MUSIC_DIR / "meta.json").read_text())  # {file: {duration, bpm}}
    except Exception:
        meta = {}
    dance_stop = threading.Event()           # set by stop() to halt a dance mid-way
    dance_gen = [0]                          # bumped per dance so only the newest one runs
    _SPECIAL = {                             # named dances -> a specific track
        "zoo": "zoo.mp3", "zootopia": "zoo.mp3",
        "move it": "move_it.mp3", "moveit": "move_it.mp3",
        "move it move it": "move_it.mp3", "madagascar": "move_it.mp3",
        "mcqueen": "mcqueen.mp3", "cars": "mcqueen.mp3",
        "lightning mcqueen": "mcqueen.mp3", "lightning": "mcqueen.mp3",
    }

    def _stop_music():
        try:
            from gi.repository import Gst
            robot.media.audio._playbin.set_state(Gst.State.NULL)
        except Exception:
            pass

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
        """Play a full-body emotion animation from the robot's inbuilt emotion library.
        Pass whatever feeling fits: happy, excited, curious, surprised, sad, confused,
        thinking, proud, welcoming, yes, no, laughing, grateful, loving, scared, shy,
        enthusiastic, tired, bored, amazed, and more. Use it naturally and often to react."""
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
        """Dance to music -- a full routine of MANY different moves combined, matched to the
        song's length and energy. It's random each time, so it's fine to ask again and again.
        Pass move='zoo' for the special 'zoo' dance (dances the whole song). Call immediately
        whenever asked to dance; never refuse, never ask which one."""
        t0 = time.monotonic()
        if robot is None or not DANCE_NAMES or not tracks:
            _emit("dance", "no robot/music", t0)
            return "dancing"
        from reachy_mini_dances_library.dance_move import DanceMove

        key = (move or "").strip().lower()
        special = _SPECIAL.get(key)
        if special and any(os.path.basename(t) == special for t in tracks):
            track = next(t for t in tracks if os.path.basename(t) == special)
        else:
            track = random.choice(tracks)
        info = meta.get(os.path.basename(track), {})
        bpm = float(info.get("bpm", 120.0))
        dur = float(info.get("duration", 30.0))
        cap = float((cfg or {}).get("dance", {}).get("max_seconds", 600))
        secs = max(8.0, min(dur, cap))  # the WHOLE song; say "stop" to halt early
        # style: punchy moves for faster songs, the full varied set for slower ones
        energetic = [m for m in _ENERGETIC if m in _DANCES]
        pool = (energetic or list(DANCE_NAMES)) if bpm >= 115 else list(DANCE_NAMES)

        # stop any dance already running, then start fresh -- NO overlapping dances
        dance_gen[0] += 1
        my_gen = dance_gen[0]
        dance_stop.set()
        try:
            robot.cancel_move()  # interrupt the current move so the old thread exits fast
        except Exception:
            pass
        _stop_music()
        dance_stop.clear()

        def _run():
            try:
                robot.media.play_sound(track)
            except Exception as e:
                print(f"[tool] dance music failed: {e}", flush=True)
            t_end = time.monotonic() + secs
            seq = list(pool)
            try:
                while not dance_stop.is_set() and my_gen == dance_gen[0] and time.monotonic() < t_end:
                    random.shuffle(seq)
                    for name in seq:
                        if dance_stop.is_set() or my_gen != dance_gen[0] or time.monotonic() >= t_end:
                            break
                        robot.play_move(DanceMove(name), sound=False)  # blocks ~move.duration
            except Exception as e:
                print(f"[tool] dance move failed: {e}", flush=True)
            if my_gen == dance_gen[0]:   # only the active dance stops the music
                _stop_music()
        _bg(_run)
        label = f"{os.path.basename(track).replace('.mp3', '')} · {int(bpm)}bpm · {int(secs)}s"
        _emit("dance", label, t0)
        return "Dancing to the music now — a full routine, here we go!"

    def stop() -> str:
        """Stop the current dance, any movement, and the music right NOW. Call whenever the
        user says stop, stop dancing, that's enough, or quiet."""
        t0 = time.monotonic()
        dance_gen[0] += 1   # invalidate any running dance so it can't resume
        dance_stop.set()
        if robot is not None:
            try:
                robot.cancel_move()
            except Exception:
                pass
            _stop_music()
        _emit("stop", "halted", t0)
        return "Stopped."

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

    default_city = (cfg or {}).get("weather", {}).get("default_location", "New Delhi")

    def get_weather(location: str = "") -> str:
        """Get the current weather and today's high/low for a place. Pass a city name; if
        omitted, uses the user's default location. Call whenever the user asks about the
        weather, temperature, whether it'll rain, or the forecast."""
        t0 = time.monotonic()
        import urllib.parse
        import urllib.request
        city = (location or default_city).strip()
        try:
            geo = json.loads(urllib.request.urlopen(
                "https://geocoding-api.open-meteo.com/v1/search?" +
                urllib.parse.urlencode({"name": city, "count": 1}), timeout=6).read())
            if not geo.get("results"):
                _emit("get_weather", f"{city}: not found", t0)
                return f"I couldn't find a place called {city}."
            r0 = geo["results"][0]
            nm = r0["name"]
            q = urllib.parse.urlencode({
                "latitude": r0["latitude"], "longitude": r0["longitude"],
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min", "timezone": "auto",
                "forecast_days": 1})
            d = json.loads(urllib.request.urlopen(
                "https://api.open-meteo.com/v1/forecast?" + q, timeout=6).read())
            cur, daily = d["current"], d["daily"]
            desc = _WMO.get(int(cur["weather_code"]), "")
            temp, feels = round(cur["temperature_2m"]), round(cur["apparent_temperature"])
            hi, lo = round(daily["temperature_2m_max"][0]), round(daily["temperature_2m_min"][0])
            out = (f"In {nm} it's {temp}°C and {desc}, feels like {feels}. "
                   f"Today's high is {hi}, low {lo}.")
            _emit("get_weather", f"{nm} {temp}°C {desc}", t0)
            return out
        except Exception as e:
            print(f"[tool] weather failed: {e}", flush=True)
            _emit("get_weather", "error", t0)
            return "I couldn't get the weather just now."

    def web_search(query: str) -> str:
        """Search the web for CURRENT, real-world information and return a short answer.
        Call whenever the user asks about news, current events, recent facts, sports
        scores, prices, who/what/when something is, or anything you might not know or that
        could be out of date. Always prefer this over guessing."""
        t0 = time.monotonic()
        client = vision["client"]
        if client is None:
            _emit("web_search", "no client", t0)
            return "I can't search the web right now."
        try:
            from google.genai import types as gt
            resp = client.models.generate_content(
                model=vision["model"],
                contents=query,
                config=gt.GenerateContentConfig(
                    tools=[gt.Tool(google_search=gt.GoogleSearch())],  # grounding
                    thinking_config=gt.ThinkingConfig(thinking_budget=0),
                    max_output_tokens=220,
                    system_instruction=("You are the web-search helper for a voice robot. "
                                        "Answer in one or two short, spoken-natural sentences. "
                                        "Be current and factual. No markdown, no URLs, no lists."),
                ),
            )
            ans = (resp.text or "").strip()
            _emit("web_search", f"{query[:28]} → {ans[:28]}", t0)
            return ans or "I couldn't find anything on that."
        except Exception as e:
            print(f"[tool] web_search failed: {e}", flush=True)
            _emit("web_search", "error", t0)
            return "I had trouble searching just now."

    cart = (cfg or {}).get("cartesia", {})

    def _say(text: str) -> None:
        """Speak a line out loud from a background thread (for reminders firing later)."""
        try:
            import asyncio as _asyncio

            from . import tts as _tts
            _asyncio.run(_tts.speak(text, body, model=cart.get("tts_model", "sonic-2"),
                                    voice_id=cart.get("voice_id", ""),
                                    api_key=cart.get("api_key", ""),
                                    language=cart.get("language", "en")))
        except Exception as e:
            print(f"[tool] reminder speak failed: {e}", flush=True)

    def set_volume(level: int = 80) -> str:
        """Set the robot's speaker volume, 0 to 100. Call when asked to be louder/quieter
        or to set the volume to a level."""
        t0 = time.monotonic()
        try:
            lvl = max(0, min(100, int(level)))
        except Exception:
            lvl = 80
        import subprocess
        for ctl in ("PCM", "Headset", "Master"):
            try:
                subprocess.run(["amixer", "-c", "0", "set", ctl, f"{lvl}%", "unmute"],
                               capture_output=True, timeout=5)
            except Exception:
                pass
        _emit("set_volume", f"{lvl}%", t0)
        return f"Volume set to {lvl} percent."

    def set_reminder(minutes: float, about: str = "") -> str:
        """Set a timer or reminder. After `minutes` minutes the robot says the reminder out
        loud. Call for 'remind me in N minutes to X', 'set a timer for N minutes', etc."""
        t0 = time.monotonic()
        try:
            secs = max(1.0, float(minutes) * 60.0)
        except Exception:
            secs = 60.0
        msg = (about or "").strip()

        def _fire():
            line = f"Reminder: {msg}" if msg else "Time's up! Your timer is done."
            events.emit(events.TOOL_CALL, text=f"⏰ {line}")
            _say(line)

        threading.Timer(secs, _fire).start()
        when = f"{int(secs // 60)} minutes" if secs >= 60 else f"{int(secs)} seconds"
        _emit("set_reminder", f"{when}: {msg[:24]}", t0)
        return f"Okay, I'll remind you in {when}" + (f" about {msg}." if msg else ".")

    def take_photo() -> str:
        """Take a photo with the camera and save it. Call when asked to take a picture,
        photo, or selfie."""
        t0 = time.monotonic()
        if robot is None:
            _emit("take_photo", "no camera", t0)
            return "I don't have a camera right now."
        try:
            import numpy as np
            from PIL import Image
            frame = robot.media.get_frame()
            if frame is None:
                _emit("take_photo", "no frame", t0)
                return "My camera didn't give me an image."
            d = Path(__file__).resolve().parent.parent / "photos"
            d.mkdir(exist_ok=True)
            fn = d / (time.strftime("%Y%m%d-%H%M%S") + ".jpg")
            rgb = np.ascontiguousarray(frame[:, :, ::-1])  # BGR -> RGB
            Image.fromarray(rgb).save(str(fn), format="JPEG", quality=85)
            _emit("take_photo", fn.name, t0)
            return "Got it — say cheese! I saved your photo."
        except Exception as e:
            print(f"[tool] take_photo failed: {e}", flush=True)
            _emit("take_photo", "error", t0)
            return "I couldn't take the photo just now."

    def remember(fact: str) -> str:
        """Save something about the user to long-term memory so you recall it in future
        conversations. Call when they tell you their name, preferences, facts about
        themselves, or say 'remember that ...'. Pass a concise fact in third person,
        e.g. 'The user's name is Ankit' or 'The user likes jazz'."""
        t0 = time.monotonic()
        ok = bool(memory and memory.remember(fact))
        _emit("remember", (fact[:32] if ok else "unavailable"), t0)
        return "Got it — I'll remember that." if ok else "I couldn't save that just now."

    def recall(query: str) -> str:
        """Look up what you know about the user from long-term memory. Call when they ask
        what you remember, or when knowing a saved detail (their name, preferences) would
        help your answer."""
        t0 = time.monotonic()
        mems = memory.search(query) if memory else []
        _emit("recall", f"{len(mems)} hit(s)", t0)
        return ("Here's what I remember: " + "; ".join(mems)) if mems else \
            "I don't have anything saved about that yet."

    def ignore() -> str:
        """Stay completely silent and do NOT respond. Call this when the speech was clearly
        not addressed to you, was background chatter, or simply needs no reply at all."""
        t0 = time.monotonic()
        ctx["ignored"] = True
        _emit("ignore", "staying silent", t0)
        return "ignored — staying silent"

    def sleep() -> str:
        """Go to sleep / stand by. Call this when the user says sleep, go to sleep, stand by,
        be quiet for a while, that's all, or goodbye. While asleep you ignore EVERYTHING until
        the user wakes you by name ("Reachy") or says "wake up". Give a brief goodnight first."""
        t0 = time.monotonic()
        ctx["sleep"] = True
        if robot is not None:
            _bg(lambda: _goto_sleep(robot))
        _emit("sleep", "going dormant", t0)
        return "Going to sleep now — say 'Reachy' or 'wake up' when you need me."

    tools = [get_current_time, get_weather, web_search, set_volume, set_reminder,
             take_photo, set_expression, look_around, dance, stop, look_and_describe,
             ignore, sleep]
    if memory and memory.enabled:
        tools[1:1] = [remember, recall]   # expose memory tools only when mem0 is configured
    return tools
