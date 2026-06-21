"""Robot expressions via the INBUILT emotion library.

Reachy Mini ships a library of professionally-recorded, full-body emotion
animations -- `pollen-robotics/reachy-mini-emotions-library`, which the daemon
preloads/caches on the robot at startup. We play those instead of hand-built head
poses: they're expressive and, crucially, guaranteed reachable (our hand-tuned
poses overflowed the head's IK envelope -> "Collision / not achievable").

set_expression(emotion) maps the brain's emotion word (happy, curious, sad, ...)
to the closest recorded move and plays it with robot.play_move() -- the same call
we already use for dances. The real move names are discovered at load time via
list_moves(), so we adapt to whatever the dataset actually contains. Falls back to
a gentle antenna wiggle only if the library or robot isn't available.
"""
from __future__ import annotations

import random
import threading

_DATASET = "pollen-robotics/reachy-mini-emotions-library"

# Map the brain's emotion words -> name-stems to look for in the dataset, in priority
# order. We play any recorded move whose name starts with / contains a stem, e.g.
# "curious" -> curious1 / curious2 / inquiring1. Discovered names win; this is just
# how we route an abstract emotion onto whatever the library happens to call it.
_ALIASES = {
    "happy":     ["happy", "cheerful", "joy", "glad", "content", "amused"],
    "excited":   ["excited", "amazed", "enthusiast", "success", "celebr", "energ"],
    "curious":   ["curious", "inquiring", "intrigued", "interested", "attentive"],
    "surprised": ["surprised", "amazed", "shock", "wow", "astonish", "oups"],
    "sad":       ["sad", "disappointed", "down", "sorry", "unhappy", "dejected"],
    "confused":  ["confused", "puzzled", "uncertain", "thoughtful", "thinking"],
    "thinking":  ["thinking", "thoughtful", "ponder", "wondering", "reflect"],
    "proud":     ["proud", "confident"],
    "welcoming": ["welcoming", "greeting", "waving", "hello"],
    "yes":       ["yes", "affirmative", "agree", "nod", "approval"],
    "no":        ["no", "disagree", "refuse", "deny", "disapprov"],
    "neutral":   ["attentive", "idle", "calm", "listening", "neutral"],
}
NAMES = sorted(_ALIASES)           # the emotion words we advertise to the brain
_DEFAULT_WORD = "happy"

_lib = {"moves": None, "names": [], "loaded": False}
_lock = threading.Lock()           # one expression at a time -- no overlapping play_move


def _load() -> None:
    if _lib["loaded"]:
        return
    _lib["loaded"] = True
    try:
        from reachy_mini.motion.recorded_move import RecordedMoves
        rm = RecordedMoves(_DATASET)                  # loads from the daemon's cache
        _lib["moves"] = rm
        _lib["names"] = rm.list_moves()
        print(f"[expr] {len(_lib['names'])} emotion moves loaded: {_lib['names']}", flush=True)
    except Exception as e:
        print(f"[expr] emotion library unavailable ({e}) -- antenna fallback", flush=True)


def _pick(emotion: str):
    """Choose a recorded-move name for the emotion word, or None if the library's empty."""
    names = _lib["names"]
    if not names:
        return None
    stems = _ALIASES.get(emotion, [emotion]) + _ALIASES[_DEFAULT_WORD]
    low = [(n, n.lower()) for n in names]
    for stem in stems:
        hits = [n for n, nl in low if nl.startswith(stem) or stem in nl]
        if hits:
            return random.choice(hits)      # variety among curious1 / curious2 / ...
    return random.choice(names)             # last resort: still do something expressive


def _antenna_fallback(robot) -> None:
    """A quick, always-reachable antenna flick when the recorded library isn't there."""
    import time
    try:
        for a in (0.5, -0.3, 0.4, 0.0):
            robot.set_target_antenna_joint_positions([float(a), float(-a)])
            time.sleep(0.12)
    except Exception:
        pass


def perform(body, emotion: str) -> str:
    """Play an expressive emotion on the robot (background thread). Returns a label."""
    word = (emotion or "").strip().lower() or _DEFAULT_WORD
    robot = getattr(body, "robot", None)
    if robot is None:
        return word                          # laptop/mock: nothing to move

    def _run():
        if not _lock.acquire(blocking=False):
            return                           # an expression is already playing; skip
        try:
            _load()
            name = _pick(word)
            if name is None:
                _antenna_fallback(robot)
                return
            try:
                move = _lib["moves"].get(name)
                robot.play_move(move, initial_goto_duration=0.6)   # blocks ~move duration
            except Exception as e:
                print(f"[expr] play {name!r} failed: {e}", flush=True)
                _antenna_fallback(robot)
        finally:
            _lock.release()

    threading.Thread(target=_run, daemon=True).start()
    return word
