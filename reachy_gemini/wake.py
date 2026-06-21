"""Wake-word gating + sleep -- so Reachy only acts when it's actually addressed.

The mic picks up background talk, and STT even invents words on noise, so the robot
replies to things you never said. The fix: it only engages when you call its name
("Reachy"). But its name is exactly the word STT mangles most -- "richie", "reach",
"ritchie", "reachy" -- so we match a whole FAMILY of variants fuzzily, never a literal
"reachy" (that would lock you out the moment STT writes "richie").

After it answers it stays awake for a short follow-up window, so you don't have to say
the name every single sentence. The sleep() tool drops it fully dormant until you
explicitly wake it ("Reachy" / "wake up").

State: idle -> (hears its name) -> active -> (window elapses) -> idle
                                      ^                                |
                                      +--------- follow-up ------------+
       asleep -> (hears name / "wake up") -> active
"""
from __future__ import annotations

import re
import time

# What STT actually writes when you say "Reachy". Curated from real mis-hears: the
# name ends on a -y / -ie / -i vowel, so we match stems (reach/rich/ritch/reech/leech)
# only WITH such an ending. Bare "reach" / "rich" are deliberately NOT matched -- they
# are common words ("reach for it", "she is rich") and would false-trigger constantly.
# Note the ridd/riddh/redd/ridh stems: Gemini STT transcribes "Reachy" as "Riddhi"
# (Cartesia hears "Richie") -- confirmed live by the dual-STT compare. The -hi ending
# catches "riddhi" specifically.
_WAKE_RE = re.compile(
    r"\b(?:reach|reech|rich|ritch|leech|ridd|riddh|redd|ridh)(?:y|ie|i|ey|ee|ies|hi|ing|in)\b",
    re.IGNORECASE,
)
# Full name-variants STT writes whole (these don't fit the stem+ending shape above).
# "reaching"/"reachin" are real Gemini mis-hears of "Reachy" (seen live).
_WAKE_EXACT = {"reachy", "reachie", "reachi", "richie", "ritchie", "reechy",
               "reacher", "richy", "leechy", "reachey", "riche", "reaching", "reachin",
               "riddhi", "riddy", "reddy", "ridhi", "reddi", "reedy", "riddhe"}

# Phrases that wake it specifically out of sleep.
_SLEEP_WAKE_RE = re.compile(r"\b(wake up|wake|wakey|good morning|you awake|hello reachy)\b",
                            re.IGNORECASE)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9' ]+", " ", (text or "").lower()).strip()


def is_wake(text: str) -> bool:
    """True if the utterance addresses Reachy by its (often garbled) name."""
    t = _norm(text)
    if not t:
        return False
    if any(w in _WAKE_EXACT for w in t.split()):
        return True
    return bool(_WAKE_RE.search(t))


_STRIP_RE = re.compile(
    r"\b(?:hey|hi|ok|okay)\b|" + _WAKE_RE.pattern + r"|\b(?:" + "|".join(_WAKE_EXACT) + r")\b",
    re.IGNORECASE,
)


def strip_wake(text: str) -> str:
    """Drop the name token (and any leading hey/hi/ok) so the brain gets just the command.

    "reachy what time is it" -> "what time is it". If nothing's left (the user only
    said the name), returns "" -- the caller treats that as a bare attention call.
    """
    cleaned = _STRIP_RE.sub(" ", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.!?-")
    return cleaned


class WakeGate:
    """Decides, per utterance, whether Reachy should act -- and tracks sleep."""

    def __init__(self, cfg: dict):
        w = (cfg or {}).get("wake", {})
        self.require = bool(w.get("require_word", True))   # must say the name when idle
        self.followup_s = float(w.get("followup_s", 25))   # stay awake this long after a reply
        self.mode = "idle"                                  # idle | active | asleep
        self._active_until = 0.0

    def sleep(self) -> None:
        self.mode = "asleep"
        self._active_until = 0.0

    def note_replied(self) -> None:
        """Call after a real reply -- refreshes the follow-up window."""
        if self.mode != "asleep":
            self.mode = "active"
            self._active_until = time.monotonic() + self.followup_s

    def decide(self, text: str):
        """-> (addressed: bool, reason: str, cleaned_text: str).

        reason is a short label for the diagram so no turn is ever swallowed silently.
        """
        now = time.monotonic()
        woke = is_wake(text)

        if self.mode == "asleep":
            if woke or _SLEEP_WAKE_RE.search(_norm(text)):
                self._wake(now)
                return True, "woke", strip_wake(text)
            return False, "asleep", text

        # Follow-up window still open -> no name needed.
        if self.mode == "active" and now < self._active_until:
            return True, "follow-up", text

        # Idle (or the window just lapsed).
        self.mode = "idle"
        if not self.require:
            self._wake(now)
            return True, "always-on", text
        if woke:
            self._wake(now)
            return True, "addressed", strip_wake(text)
        return False, "no wake word", text

    def _wake(self, now: float) -> None:
        self.mode = "active"
        self._active_until = now + self.followup_s
