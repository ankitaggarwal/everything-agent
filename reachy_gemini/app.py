"""The whole agent, as one simple turn-based loop.

  listen  -> a local VAD gate buffers your utterance from the mic
  STT     -> Cartesia Ink-Whisper turns it into text
  brain   -> the fastest Gemini Flash model returns a short reply (text only)
  TTS     -> Cartesia Sonic speaks the reply through the robot

It's deliberately turn-based: we only listen between turns, so the robot never
hears itself. Every stage is timed and emitted so the diagram can show exactly
where the milliseconds go (where the bottleneck is).
"""
from __future__ import annotations

import asyncio
import collections
import time

import numpy as np

from . import events, stt, tts
from .body import make_body
from .brain import Brain


def _ms(a: float, b: float) -> int:
    return int((b - a) * 1000)


class Agent:
    def __init__(self, cfg: dict, body=None):
        self.cfg = cfg
        self.body = body if body is not None else make_body(cfg)
        self.brain = Brain(cfg, self.body)
        c = cfg.get("cartesia", {})
        self._cart = dict(api_key=c.get("api_key", ""), language=c.get("language", "en"))
        self._stt_model = c.get("stt_model", "ink-whisper")
        self._tts_model = c.get("tts_model", "sonic-2")
        self._voice_id = c.get("voice_id", "")

    # --- the pipeline ------------------------------------------------------ #
    async def _greet(self) -> None:
        greeting = self.cfg.get("gemini", {}).get("greeting")
        if greeting:
            await tts.speak(greeting, self.body, model=self._tts_model,
                            voice_id=self._voice_id, **self._cart)

    async def _listen(self, stop_event) -> bytes:
        """Block until you speak an utterance; return its PCM (16k mono int16)."""
        loop = asyncio.get_running_loop()
        vad = self.cfg.get("vad", {})
        start_thr = int(vad.get("start_level", 3000))
        stop_thr = int(vad.get("stop_level", 1200))
        hang_s = float(vad.get("hangover_s", 1.0))
        preroll = collections.deque(maxlen=int(vad.get("preroll_chunks", 6)))

        events.emit(events.LISTENING, text="say something…")
        buf = bytearray()
        open_gate = False
        last_voice = 0.0
        while not (stop_event is not None and stop_event.is_set()):
            pcm = await loop.run_in_executor(None, self.body.read_mic)
            if not pcm:
                await asyncio.sleep(0.01)
                continue
            samples = np.frombuffer(pcm, dtype=np.int16)
            peak = int(np.abs(samples).max()) if samples.size else 0
            now = time.monotonic()
            if not open_gate:
                if peak >= start_thr:
                    open_gate = True
                    events.emit(events.HEARING, text="● mic picked up your voice")
                    buf.extend(b"".join(preroll))
                    preroll.clear()
                    buf.extend(pcm)
                    last_voice = now
                else:
                    preroll.append(pcm)
            else:
                buf.extend(pcm)
                if peak >= stop_thr:
                    last_voice = now
                elif now - last_voice > hang_s:
                    return bytes(buf)  # end of utterance
        return b""

    async def _handle(self, utterance: bytes) -> None:
        t0 = time.monotonic()  # the moment you stopped talking

        events.emit(events.TRANSCRIBING, text="Cartesia STT…")
        text = await stt.transcribe(utterance, api_key=self._cart["api_key"],
                                    model=self._stt_model, language=self._cart["language"])
        t_stt = time.monotonic()
        if not text:
            events.emit(events.ERROR, text="couldn't transcribe that", ms=_ms(t0, t_stt))
            return
        events.emit(events.TRANSCRIBED, text=text, ms=_ms(t0, t_stt))

        events.emit(events.THINKING, text="Gemini Flash…")
        reply = await self.brain.ask(text)
        t_llm = time.monotonic()
        if self.brain.ignored:
            events.emit(events.DONE, text="🙊 ignored — stayed silent", ms=_ms(t0, t_llm))
            return
        if not reply:
            events.emit(events.ERROR, text="no reply from the brain", ms=_ms(t_stt, t_llm))
            return
        events.emit(events.REPLY, text=reply, ms=_ms(t_stt, t_llm))

        first_audio = [0.0]

        def _first_audio():
            first_audio[0] = time.monotonic()
            events.emit(events.SPEAKING, text=reply, ms=_ms(t_llm, first_audio[0]))

        await tts.speak(reply, self.body, model=self._tts_model, voice_id=self._voice_id,
                        on_first_audio=_first_audio, **self._cart)
        # Headline latency = you stopped talking -> the robot STARTED speaking
        # (time to first audio), NOT counting how long the reply takes to play out.
        ttf = first_audio[0] or time.monotonic()
        events.emit(events.DONE, text="you stopped → robot spoke", ms=_ms(t0, ttf))

    # --- lifecycle --------------------------------------------------------- #
    def _run_tool(self, name: str, **kwargs):
        """Fire a brain tool by name (used by the webview's /tool trigger)."""
        for t in self.brain.tools:
            if t.__name__ == name:
                return t(**kwargs)
        return f"no tool named {name}"

    def _start_webview(self) -> None:
        port = self.cfg.get("web", {}).get("port")
        if not port:
            return
        try:
            from . import webview
            webview.start(int(port))
            webview.register_trigger(self._run_tool)  # GET /tool?name=dance&move=zoo
        except Exception as e:
            print(f"[webview] disabled: {e}", flush=True)

    async def run(self, stop_event=None) -> None:
        self.body.start()
        self._start_webview()
        try:
            await self._greet()
            while not (stop_event is not None and stop_event.is_set()):
                utterance = await self._listen(stop_event)
                if utterance:
                    await self._handle(utterance)
        finally:
            self.body.stop()
