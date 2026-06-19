"""The orchestrator -- everything-agent's main loop.

It reads like the architecture diagram (docs/architecture.html):

    HEAR     wake word  ->  speech-to-text
    DECIDE   router      ->  instant reply  OR  escalate to the agent brain
    ACT      agent brain ->  tools / MCP servers  (+ approval gate)
    EXPRESS  speak the reply + a little movement
    MEMORY   remember the exchange

Every subsystem is a *port* with a chosen *adapter* (see core/ports.py). This
file just builds them from config via the registry and runs the loop -- it never
imports a concrete backend. To change behavior, change a config line or add an
adapter; you don't touch this file.
"""
from __future__ import annotations

import asyncio
import difflib
import logging
import random
import re
import time
from collections import deque

from .brain.agent import build as build_agent
from .brain.router import build as build_router
from .core.approval import Approval
from .core.context import AgentContext
from .core.ports import Decision
from .core.registry import Registry
from .expressing.tts import build as build_tts
from .hearing.stt import build as build_stt
from .hearing.wakeword import build as build_wakeword
from .memory import build as build_memory
from .providers.cartesia import build_providers
from .robot import build as build_robot

log = logging.getLogger("everything_agent")

_QUIT = {"quit", "exit", "bye", "goodbye"}

# Common ways STT mis-transcribes an uncommon name, so addressing still works.
_TRIGGER_MISHEARS = {
    # Cartesia Ink-Whisper mangles "reachy" into a wide, inconsistent cluster in
    # practice (observed live: riki, eriki, ricci, richie, erichy...). We match the
    # whole cluster so "hey reachy" actually wakes it. "reachy" is a poor wake word
    # for this STT -- a cleaner long-term fix is a more STT-friendly trigger.
    "reachy": [
        "reachy", "reachie", "reachee", "reachi",
        "richie", "ritchie", "richy", "richee", "richi",
        "ricci", "ricky", "rickey", "rikki", "riki", "ricki", "rickie",
        "erichy", "erichie", "eriki", "ericky", "erichee", "ericki",
    ],
}


class EverythingAgent:
    def __init__(self, config: dict):
        self.config = config
        self.ctx = AgentContext(config=config)
        self._running = False

    async def start(self) -> None:
        cfg, ctx = self.config, self.ctx

        # --- shared infrastructure, built in dependency order ---
        ctx.providers = build_providers(cfg)
        ctx.robot = build_robot(cfg.get("robot", {}), ctx)
        ctx.memory = build_memory(cfg.get("memory", {}), ctx)
        ctx.approval = Approval(cfg.get("approval", {}))
        ctx.robot.connect()

        # --- capabilities (modules) -> their actions go into the context ---
        self.registry = Registry(ctx.robot, ctx.providers, ctx.memory, cfg)
        self.registry.load()
        ctx.actions = self.registry.all_actions()

        # --- HEAR / DECIDE / EXPRESS adapters (may read ctx.actions etc.) ---
        hearing = cfg.get("hearing", {})
        brain = cfg.get("brain", {})
        self.wakeword = build_wakeword(hearing.get("wakeword", {}), ctx)
        self.stt = build_stt(hearing.get("stt", {}), ctx)
        self.router = build_router(brain.get("router", {}), ctx)
        self.agent_brain = build_agent(brain.get("agent", {}), ctx)
        self.voice = build_tts(cfg.get("express", {}).get("tts", {}), ctx)

        # Barge-in: let the user interrupt the robot mid-reply. A mic watcher runs
        # while TTS plays and cuts it off on sustained speech. NOTE: the mic has no
        # echo cancellation, so on real hardware energy mode can hear the robot's
        # OWN voice -- tune `energy` high or keep it off until verified on-device.
        bi = (cfg.get("express", {}) or {}).get("barge_in", {}) or {}
        self.barge_in = {
            "enabled": bool(bi.get("enabled", False)),
            # "transcript" = echo-robust (STT during playback, ignore our own
            # words); "energy" = simple loudness (only safe with no echo / AEC).
            "mode": bi.get("mode", "energy"),
            "energy": float(bi.get("energy", 0.02)),     # RMS threshold (energy mode)
            "min_ms": float(bi.get("min_ms", 250)),      # sustained speech to trip
            "grace_ms": float(bi.get("grace_ms", 350)),  # ignore the first moments
        }
        self._interrupted = False
        self._interrupt_text = None   # words the user cut in with -> next turn
        # A deliberate beat before replying, so it doesn't snap back the instant
        # you pause -- feels calmer and less jumpy.
        self.reply_pause = float(cfg.get("brain", {}).get("agent", {}).get("reply_pause", 0.0))
        # Auto-emotion: even on instant (no-tool) replies, the robot reacts with a
        # body gesture picked from the reply's tone -- so it's always expressive.
        _emo = ("express_happy", "express_excited", "express_curious", "express_confused",
                "express_thinking", "express_sad", "nod_yes", "shake_no", "celebrate")
        self._emotions = {a.name: a for a in (self.ctx.actions or []) if a.name in _emo}

        # Address gate: when there's no acoustic wake word (always-listening),
        # only act on utterances that name the robot. `trigger` is the name to
        # listen for; we also match common mis-hearings of it. Empty = act on all.
        trigger = (cfg.get("hearing", {}).get("trigger") or "").strip().lower()
        self.trigger = trigger
        self.trigger_variants = sorted(
            {trigger, *_TRIGGER_MISHEARS.get(trigger, [])}, key=len, reverse=True
        ) if trigger else []
        # Follow-up window: after a reply, keep accepting input WITHOUT the name
        # for this many seconds, so a conversation flows naturally (name it once
        # to start, then just keep talking). 0 disables it.
        self.followup_window = float(cfg.get("hearing", {}).get("followup_window", 25))
        self.engaged_until = 0.0

        # Proactive speech: modules (timers/reminders) can push utterances the
        # robot says unprompted. A background task drains the queue; a lock keeps
        # proactive speech from overlapping a reply.
        self.proactive: asyncio.Queue = asyncio.Queue()
        self._speak_lock = asyncio.Lock()
        for m in getattr(self.registry, "modules", []):
            if hasattr(m, "set_speaker"):
                m.set_speaker(lambda text: self.proactive.put_nowait(text))
        asyncio.create_task(self._proactive_loop())

        # Recent turns for the settings webpage's live view (heard/reply/timings).
        self.transcript = deque(maxlen=40)
        self._loop = asyncio.get_running_loop()
        self._start_web()

        self._running = True
        log.info("everything-agent online. Say something (or 'quit').")

    # ---- settings webpage hooks (see everything_agent/web/server.py) ----
    def _start_web(self) -> None:
        web = self.config.get("web", {}) or {}
        if not web.get("enabled", False):
            return
        try:
            from .web.server import start_web
            self._web = start_web(self, host=web.get("host", "0.0.0.0"),
                                  port=int(web.get("port", 8080)))
            log.info("settings webpage on http://%s:%s",
                     web.get("host", "0.0.0.0"), web.get("port", 8080))
        except Exception:
            log.exception("settings webpage failed to start (continuing without it)")

    async def run(self) -> None:
        try:
            await self.start()
            while self._running:
                try:
                    await self._cycle()
                except (KeyboardInterrupt, asyncio.CancelledError):
                    raise
                except Exception:
                    # One bad turn (network blip, API hiccup, mic glitch) must
                    # never kill the always-on robot: log it and keep listening.
                    log.exception("cycle failed -- recovering")
                    await asyncio.sleep(1.0)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            log.info("shutting down...")
            if self.ctx.robot is not None:   # start() may have failed before connect
                self.ctx.robot.reset()
                self.ctx.robot.disconnect()

    async def _cycle(self) -> None:
        # HEAR --------------------------------------------------------
        # Stay alive (idle micro-moves) and track the speaker (mic-array DoA)
        # while waiting for the wake word AND while listening -- then cancel that
        # ambient behavior so emotion gestures + TTS own the body during the reply.
        # If the user just barged in over the last reply, their words ARE this
        # turn's input -- skip waiting for the wake word and listening afresh.
        from_interrupt = False
        engaged = time.monotonic() < self.engaged_until
        if self._interrupt_text:
            text = self._interrupt_text
            self._interrupt_text = None
            t_stt = 0.0
            from_interrupt = True
        else:
            alive = asyncio.create_task(self._be_alive())
            try:
                # In the follow-up window we skip the wake word so the conversation
                # continues hands-free; otherwise wait to be woken (acoustic wake
                # word, or instantly in always-listening/mock mode).
                if not engaged:
                    if not await self.wakeword.wait():
                        return
                t0 = time.monotonic()
                text = await self.stt.listen()
                t_stt = time.monotonic() - t0
            finally:
                alive.cancel()
                try:
                    await alive
                except asyncio.CancelledError:
                    pass

        if text is None:                          # input ended (Ctrl-D)
            self._running = False
            return
        if not text.strip():
            return

        # ADDRESS GATE -- act when named, OR within the follow-up window after a
        # recent reply (so a back-and-forth doesn't need the name every turn).
        # A barge-in is already part of an active exchange, so it skips the gate.
        if not from_interrupt:
            addressed, cleaned = self._addressed(text)
            if addressed:
                text = cleaned
            elif engaged:
                log.info("follow-up (engaged) -- accepting without name: %r", text[:50])
            else:
                log.info("not addressed (no %r) -- ignoring: %r", self.trigger, text[:50])
                return

        # Quit only when addressed -- an overheard "bye" (always-listening mode)
        # must not shut the robot down. Tolerate STT punctuation ("Goodbye.").
        if text.strip().lower().strip(" .,!?") in _QUIT:
            await self._say("Bye!")
            self._running = False
            return

        # DECIDE + ACT -----------------------------------------------
        reply, t_brain = await self._handle(text)

        # EXPRESS + MEMORY -- skipped when there's nothing to say (router "ignore").
        t1 = time.monotonic()
        if reply:
            await self._express(reply)                  # react with a gesture first
            if self.reply_pause:
                await asyncio.sleep(self.reply_pause)   # a calm beat before speaking
            await self._say(reply)
            self.ctx.memory.add_turn(text, reply)
            # Stay engaged for a follow-up so the user can keep talking name-free.
            self.engaged_until = time.monotonic() + self.followup_window
        t_tts = time.monotonic() - t1
        log.info("⏱  stt=%.2fs  brain=%.2fs  tts=%.2fs  total=%.2fs",
                 t_stt, t_brain, t_tts, t_stt + t_brain + t_tts)
        self.transcript.append({
            "t": time.time(), "heard": text, "reply": reply,
            "stt": round(t_stt, 2), "brain": round(t_brain, 2),
            "tts": round(t_tts, 2), "total": round(t_stt + t_brain + t_tts, 2),
        })

    async def _say(self, text: str) -> None:
        """speak() blocks until the audio finishes playing, so run it in a worker
        thread -- the event loop stays free and emotion gestures (background
        tasks) keep animating DURING speech instead of freezing until it ends.
        The lock serializes against proactive speech (timers) so they never talk
        over each other. With barge-in on, a mic watcher runs alongside playback
        and cuts it short when the user starts talking."""
        async with self._speak_lock:
            if not self.barge_in["enabled"] or self.ctx.robot.media is None:
                await asyncio.to_thread(self.voice.speak, text)
                return
            import threading
            stop = threading.Event()
            self._interrupted = False
            speak_task = asyncio.create_task(asyncio.to_thread(self.voice.speak, text, stop))
            watch_task = asyncio.create_task(self._watch_barge_in(stop, text))
            done, _ = await asyncio.wait({speak_task, watch_task},
                                        return_when=asyncio.FIRST_COMPLETED)
            stop.set()                       # end speech if the watcher tripped
            if watch_task not in done:
                watch_task.cancel()          # speech finished first -> stop watching
            await asyncio.gather(speak_task, watch_task, return_exceptions=True)

    async def _watch_barge_in(self, stop, spoken_text: str) -> bool:
        if self.barge_in["mode"] == "transcript":
            return await self._watch_barge_in_transcript(stop, spoken_text)
        return await self._watch_barge_in_energy(stop)

    async def _watch_barge_in_energy(self, stop) -> bool:
        """Read the mic while the robot speaks; trip on sustained speech energy.
        Only safe where the mic can't hear the robot (loopback, AEC, headset)."""
        import time as _t

        import numpy as np
        media = self.ctx.robot.media
        sr = media.get_input_audio_samplerate() or 16000
        try:
            media.start_recording()
        except Exception:  # noqa: BLE001
            pass
        need = int(self.barge_in["min_ms"] / 1000.0 * sr)
        grace_until = _t.monotonic() + self.barge_in["grace_ms"] / 1000.0
        above = 0
        while not stop.is_set():
            frame = await asyncio.to_thread(media.get_audio_sample)
            if frame is None or getattr(frame, "size", 0) == 0:
                await asyncio.sleep(0.005)
                continue
            if _t.monotonic() < grace_until:    # ignore onset / residual
                continue
            if getattr(frame, "ndim", 1) == 2:
                frame = frame.mean(axis=1)
            frame = frame.astype(np.float32)
            rms = float(np.sqrt(np.mean(frame * frame))) if frame.size else 0.0
            if rms >= self.barge_in["energy"]:
                above += frame.shape[0]
                if above >= need:
                    self._interrupted = True
                    stop.set()
                    log.info("🤚 barge-in (rms=%.3f) -- stopping speech", rms)
                    return True
            else:
                above = 0
        return False

    async def _watch_barge_in_transcript(self, stop, spoken_text: str) -> bool:
        """Echo-robust barge-in: transcribe the mic WHILE the robot speaks and trip
        only on words that aren't the robot's own (which we know -- spoken_text).
        The robot's own voice bleeding into the mic transcribes back as its own
        words, so it's filtered out; real user speech brings NEW words."""
        import json
        import os
        import time as _t

        import numpy as np
        media = self.ctx.robot.media
        api_key = os.environ.get("CARTESIA_API_KEY")
        if not (media is not None and api_key):
            return False
        try:
            import websockets
            from scipy.signal import resample_poly
        except Exception:  # noqa: BLE001
            return False

        spoken = set(re.findall(r"[a-z']+", (spoken_text or "").lower()))
        mic_sr = media.get_input_audio_samplerate() or 16000
        try:
            media.start_recording()
        except Exception:  # noqa: BLE001
            pass
        url = ("wss://api.cartesia.ai/stt/websocket?model=ink-whisper"
               "&language=en&encoding=pcm_f32le&sample_rate=16000")
        headers = {"X-API-Key": api_key, "Cartesia-Version": "2024-11-13"}
        grace_until = _t.monotonic() + self.barge_in["grace_ms"] / 1000.0
        try:
            async with websockets.connect(url, additional_headers=headers,
                                          max_size=4_000_000) as ws:
                async def pump():
                    batch, n, target = [], 0, 1600
                    while not stop.is_set():
                        frame = await asyncio.to_thread(media.get_audio_sample)
                        if frame is None or getattr(frame, "size", 0) == 0:
                            await asyncio.sleep(0.005)
                            continue
                        if _t.monotonic() < grace_until:
                            continue
                        if getattr(frame, "ndim", 1) == 2:
                            frame = frame.mean(axis=1)
                        frame = frame.astype(np.float32)
                        if mic_sr != 16000:
                            frame = resample_poly(frame, 16000, mic_sr).astype(np.float32)
                        batch.append(frame)
                        n += frame.shape[0]
                        if n >= target:
                            try:
                                await ws.send(np.concatenate(batch).astype(np.float32).tobytes())
                            except Exception:  # noqa: BLE001
                                return
                            batch, n = [], 0

                async def consume():
                    while not stop.is_set():
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=0.5)
                        except asyncio.TimeoutError:
                            continue
                        except Exception:  # noqa: BLE001
                            return
                        if isinstance(msg, bytes):
                            continue
                        try:
                            ev = json.loads(msg)
                        except json.JSONDecodeError:
                            continue
                        if not (ev.get("type") == "transcript" and ev.get("is_final")):
                            continue
                        words = re.findall(r"[a-z']+", (ev.get("text") or "").lower())
                        novel = [w for w in words if w not in spoken and len(w) > 1]
                        # Real interruption = enough genuinely new words (raised so
                        # a stray "bye" / "what" in a noisy room won't cut us off).
                        if len(novel) >= 3 and len(novel) / max(1, len(words)) >= 0.6:
                            self._interrupted = True
                            self._interrupt_text = " ".join(words)
                            stop.set()
                            log.info("🤚 barge-in: heard %r (new words: %s)",
                                     " ".join(words), novel)
                            return

                await asyncio.gather(pump(), consume())
        except Exception as e:  # noqa: BLE001
            log.warning("barge-in transcript watcher error: %s", e)
        return self._interrupted

    async def _express(self, text: str) -> None:
        """Pick a body gesture that fits the reply's tone and play it (fire-and-
        forget, overlaps the speech) so the robot is always expressive -- even on
        instant replies where the brain never called an emotion tool."""
        if not self._emotions:
            return
        low = text.lower()
        if any(w in low for w in ("sorry", "can't", "cannot", "don't know",
                                  "not sure", "didn't catch", "say that again", "hmm")):
            name = "express_confused"
        elif any(w in low for w in ("wow", "yay", "awesome", "amazing", "cool",
                                    "hooray", "woohoo", "love it", "so fun")):
            name = random.choice(("express_excited", "celebrate"))
        elif text.strip().endswith("?"):
            name = "express_curious"
        elif any(w in low for w in ("yes", "yep", "sure", "absolutely", "of course", "okay")):
            name = "nod_yes"
        elif low.startswith(("no", "nope")) or "no, " in low:
            name = "shake_no"
        else:
            name = random.choice(("express_happy", "express_happy", "express_curious"))
        act = self._emotions.get(name)
        if act:
            try:
                await act.handler({})        # async, returns immediately
            except Exception:  # noqa: BLE001
                pass

    async def _proactive_loop(self) -> None:
        """Speak module-initiated utterances (a timer firing, a due reminder)
        unprompted, whenever the robot isn't already talking."""
        while True:
            try:
                text = await self.proactive.get()
                if text and self._running:
                    log.info("proactive: %s", text)
                    await self._say(text)
            except asyncio.CancelledError:
                return
            except Exception:
                log.exception("proactive speech failed")

    def _addressed(self, text: str):
        """When a trigger name is configured (always-listening mode), only treat
        an utterance as meant for us if it names the robot. Returns
        (is_addressed, cleaned_text) -- the trigger word is stripped so the brain
        sees just the request.

        Matching is deliberately forgiving: STT mangles an uncommon name into
        endless variants ("erichy", "reachie", "richy"...), and a *missed* address
        means the robot stays silent -- the one thing we never want. So beyond the
        known-variant list we fuzzy-match the first few tokens against the name."""
        if not self.trigger_variants:
            return True, text                      # no gate -> everything is for us
        low = text.lower()
        matched = next((v for v in self.trigger_variants
                        if re.search(rf"\b{re.escape(v)}\b", low)), None)
        if matched is None:
            # Fuzzy fallback: the name almost always leads the utterance, so only
            # the first few tokens are candidates -- keeps false positives low
            # while still catching never-before-seen mis-hearings of the name.
            for tok in re.findall(r"[a-z']+", low)[:3]:
                if len(tok) >= 4 and difflib.SequenceMatcher(
                        None, tok, self.trigger).ratio() >= 0.6:
                    matched = tok
                    break
        if matched is None:
            return False, text
        cleaned = re.sub(rf"\b{re.escape(matched)}\b[\s,!.?-]*", "", text, flags=re.IGNORECASE)
        for v in self.trigger_variants:            # also strip any known variants
            cleaned = re.sub(rf"\b{re.escape(v)}\b[\s,!.?-]*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^\s*hey[\s,]+", "", cleaned, flags=re.IGNORECASE)  # drop leading "hey"
        return True, (cleaned.strip() or text)

    async def _be_alive(self) -> None:
        """Idle micro-movements + speaker tracking, until cancelled (see _cycle)."""
        try:
            while True:
                await self.registry.run_ticks()
                await asyncio.sleep(0.25)
        except asyncio.CancelledError:
            return

    async def _handle(self, text: str):
        """Returns (reply, brain_seconds). Times the router and agent legs."""
        ctx_text = self.ctx.memory.context()
        perceptions = await self.registry.gather_perceptions()
        if perceptions:
            ctx_text = (ctx_text + "\n\nWhat your senses report:\n" + perceptions).strip()
        t0 = time.monotonic()
        try:
            decision = await self.router.decide(text, ctx_text)
        except Exception:
            log.exception("router failed -- escalating to agent")
            decision = Decision("agent")
        log.info("router: %s (%.2fs)", decision.action, time.monotonic() - t0)
        # Stay quiet on noise/fragments the router judged weren't meant for us --
        # but a real question never routes to 'ignore', so this won't silence a
        # genuine request. A usable instant reply short-circuits; an EMPTY instant
        # still falls through to the brain so we never go silent on a real ask.
        if decision.action == "ignore":
            return "", time.monotonic() - t0
        if decision.action == "instant" and (decision.reply or "").strip():
            return decision.reply, time.monotonic() - t0
        try:
            reply = await self.agent_brain.run(text, ctx_text)   # "agent"
        except Exception:
            log.exception("agent brain failed")
            reply = "Sorry, my brain glitched for a second. Mind trying again?"
        return reply, time.monotonic() - t0

    # ---- methods the settings webpage calls (run from its HTTP thread) ----
    def web_status(self) -> dict:
        cfg = self.config
        brain = cfg.get("brain", {}).get("agent", {})
        tts = cfg.get("express", {}).get("tts", {})
        persona = getattr(self.agent_brain, "personality", "") or ""
        return {
            "online": self._running,
            "trigger": self.trigger,
            "modules": cfg.get("modules", []),
            "agent_backend": brain.get("backend", "mock"),
            "agent_model": brain.get("model", ""),
            "tts_backend": tts.get("backend", "mock"),
            "tts_model": tts.get("model", ""),
            "voice_id": tts.get("voice_id", ""),
            "weather_location": cfg.get("weather", {}).get("default_location", ""),
            "persona": persona,
            "transcript": list(self.transcript),
        }

    def web_say(self, text: str) -> None:
        """Speak text now (voice preview / push-to-talk from the webpage). Marshals
        onto the agent's event loop since we're called from the HTTP thread."""
        text = (text or "").strip()
        if text and getattr(self, "_loop", None):
            asyncio.run_coroutine_threadsafe(self._say(text), self._loop)

    def web_apply(self, changes: dict) -> dict:
        """Apply settings from the webpage. Trigger + persona take effect live;
        backend/model/voice/module changes are written to the config file and need
        a restart. Returns {ok, restart_needed, error?}."""
        import os

        live, restart_needed = False, False
        # --- live: trigger name ---
        if "trigger" in changes:
            trig = (changes["trigger"] or "").strip().lower()
            self.trigger = trig
            self.trigger_variants = sorted(
                {trig, *_TRIGGER_MISHEARS.get(trig, [])}, key=len, reverse=True
            ) if trig else []
            live = True
        # --- live: persona ---
        if "persona" in changes and hasattr(self.agent_brain, "personality"):
            self.agent_brain.personality = changes["persona"] or self.agent_brain.personality
            live = True

        # --- persisted (needs restart): write into the active config file ---
        path = os.environ.get("EVERYTHING_AGENT_CONFIG", "config.yaml")
        try:
            import yaml
            with open(path) as f:
                doc = yaml.safe_load(f) or {}
            if "trigger" in changes:
                doc.setdefault("hearing", {})["trigger"] = changes["trigger"]
            if "persona" in changes:
                doc.setdefault("brain", {}).setdefault("agent", {})["personality"] = changes["persona"]
            for key, (section, field) in {
                "agent_backend": (("brain", "agent"), "backend"),
                "agent_model": (("brain", "agent"), "model"),
                "tts_model": (("express", "tts"), "model"),
                "voice_id": (("express", "tts"), "voice_id"),
            }.items():
                if key in changes:
                    node = doc
                    for s in section:
                        node = node.setdefault(s, {})
                    node[field] = changes[key]
                    restart_needed = True
            if "weather_location" in changes:
                doc.setdefault("weather", {})["default_location"] = changes["weather_location"]
                restart_needed = True
            if "modules" in changes:
                doc["modules"] = changes["modules"]
                restart_needed = True
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False)
            os.replace(tmp, path)
        except Exception as e:  # noqa: BLE001
            log.exception("web_apply failed to write config")
            return {"ok": False, "error": str(e)}
        return {"ok": True, "live": live, "restart_needed": restart_needed}
