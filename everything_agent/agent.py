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
import time

from .brain.agent import build as build_agent
from .brain.router import build as build_router
from .core.approval import Approval
from .core.context import AgentContext
from .core.registry import Registry
from .expressing.tts import build as build_tts
from .hearing.stt import build as build_stt
from .hearing.wakeword import build as build_wakeword
from .memory import build as build_memory
from .providers.cartesia import build_providers
from .robot import build as build_robot

log = logging.getLogger("everything_agent")

_QUIT = {"quit", "exit", "bye", "goodbye"}


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

        self._running = True
        log.info("everything-agent online. Say something (or 'quit').")

    async def run(self) -> None:
        await self.start()
        try:
            while self._running:
                await self._cycle()
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            log.info("shutting down...")
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
        if text.strip().lower() in _QUIT:
            self.voice.speak("Bye!")
            self._running = False
            return

        # DECIDE + ACT -----------------------------------------------
        reply, t_brain = await self._handle(text)

        # EXPRESS + MEMORY -------------------------------------------
        t1 = time.monotonic()
        self.voice.speak(reply)
        t_tts = time.monotonic() - t1
        self.ctx.memory.add_turn(text, reply)
        log.info("⏱  stt=%.2fs  brain=%.2fs  tts=%.2fs  total=%.2fs",
                 t_stt, t_brain, t_tts, t_stt + t_brain + t_tts)

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
        t0 = time.monotonic()
        decision = await self.router.decide(text, ctx_text)
        log.info("router: %s (%.2fs)", decision.action, time.monotonic() - t0)
        if decision.action == "ignore":
            return "", time.monotonic() - t0
        if decision.action == "instant":
            return decision.reply, time.monotonic() - t0
        reply = await self.agent_brain.run(text, ctx_text)   # "agent"
        return reply, time.monotonic() - t0
