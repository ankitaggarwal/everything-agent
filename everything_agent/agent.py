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
import logging
import re
import time

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
    "reachy": ["reachy", "reachie", "reachee", "richie", "ritchie"],
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

        # Address gate: when there's no acoustic wake word (always-listening),
        # only act on utterances that name the robot. `trigger` is the name to
        # listen for; we also match common mis-hearings of it. Empty = act on all.
        trigger = (cfg.get("hearing", {}).get("trigger") or "").strip().lower()
        self.trigger = trigger
        self.trigger_variants = sorted(
            {trigger, *_TRIGGER_MISHEARS.get(trigger, [])}, key=len, reverse=True
        ) if trigger else []

        self._running = True
        log.info("everything-agent online. Say something (or 'quit').")

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
        alive = asyncio.create_task(self._be_alive())
        try:
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

        # ADDRESS GATE -- only act when spoken to by name (cheap, no LLM call) ---
        addressed, text = self._addressed(text)
        if not addressed:
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
            await self._say(reply)
            self.ctx.memory.add_turn(text, reply)
        t_tts = time.monotonic() - t1
        log.info("⏱  stt=%.2fs  brain=%.2fs  tts=%.2fs  total=%.2fs",
                 t_stt, t_brain, t_tts, t_stt + t_brain + t_tts)

    async def _say(self, text: str) -> None:
        """speak() blocks until the audio finishes playing, so run it in a worker
        thread -- the event loop stays free and emotion gestures (background
        tasks) keep animating DURING speech instead of freezing until it ends."""
        await asyncio.to_thread(self.voice.speak, text)

    def _addressed(self, text: str):
        """When a trigger name is configured (always-listening mode), only treat
        an utterance as meant for us if it names the robot. Returns
        (is_addressed, cleaned_text) -- the trigger word is stripped so the brain
        sees just the request."""
        if not self.trigger_variants:
            return True, text                      # no gate -> everything is for us
        low = text.lower()
        if not any(re.search(rf"\b{re.escape(v)}\b", low) for v in self.trigger_variants):
            return False, text
        cleaned = text
        for v in self.trigger_variants:            # longest-first (set in start())
            cleaned = re.sub(rf"\b{re.escape(v)}\b[\s,!.?-]*", "", cleaned, flags=re.IGNORECASE)
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
        if decision.action == "ignore":
            return "", time.monotonic() - t0
        if decision.action == "instant":
            return decision.reply, time.monotonic() - t0
        try:
            reply = await self.agent_brain.run(text, ctx_text)   # "agent"
        except Exception:
            log.exception("agent brain failed")
            reply = "Sorry, my brain glitched for a second. Mind trying again?"
        return reply, time.monotonic() - t0
