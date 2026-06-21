"""The whole agent, as one simple turn-based loop.

  listen  -> a local VAD gate buffers your whole utterance from the mic
  STT     -> transcribe it as one piece (Gemini = accurate, or Cartesia ink-whisper)
  brain   -> the fastest Gemini Flash model returns a short reply (text only)
  TTS     -> Cartesia Sonic speaks the reply through the robot

It's deliberately turn-based: we only listen between turns, so the robot never
hears itself. Every stage is timed and emitted so the diagram can show exactly
where the milliseconds go (where the bottleneck is).
"""
from __future__ import annotations

import asyncio
import collections
import json
import os
import random
import time
import wave

import numpy as np

from . import events, stt, tts, wake
from .body import make_body
from .brain import Brain

_ACKS = ["Mm-hm?", "Yeah?", "I'm listening.", "What's up?", "Yes?"]


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
        self._stt_backend = c.get("stt_backend", "gemini")  # gemini (better) | cartesia (faster)
        self.gate = wake.WakeGate(cfg)            # only act when addressed by name + sleep
        self._debug = cfg.get("debug", {})        # log_audio / compare_stt

    # --- the pipeline ------------------------------------------------------ #
    async def _greet(self) -> None:
        greeting = self.cfg.get("gemini", {}).get("greeting")
        if greeting:
            await tts.speak(greeting, self.body, model=self._tts_model,
                            voice_id=self._voice_id, **self._cart)

    async def _listen_and_transcribe(self, stop_event):
        """Block until you speak; stream the audio to STT live; return (text, t_end).

        The STT WebSocket opens before you speak and each mic chunk is pushed as it
        arrives, so transcription overlaps with talking -- when you stop we just
        finalize and the transcript is ready almost immediately. Returns (None, None)
        on no speech / unintelligible.
        """
        loop = asyncio.get_running_loop()
        vad = self.cfg.get("vad", {})
        start_thr = int(vad.get("start_level", 3000))
        stop_thr = int(vad.get("stop_level", 1200))
        hang_s = float(vad.get("hangover_s", 1.0))
        preroll = collections.deque(maxlen=int(vad.get("preroll_chunks", 6)))

        events.emit(events.LISTENING, text="say something…")
        # Buffer the whole utterance, then transcribe it as one piece (regular, not
        # streaming -- live streaming fragmented the transcript). Gemini = accurate
        # default; cartesia = faster, lower accuracy.
        use_cartesia = self._stt_backend == "cartesia"
        buf = bytearray()
        voice_end = 0  # buffer length up to the last loud frame (to trim trailing silence)
        tail = int(0.25 * 16000) * 2  # keep ~250 ms after speech, drop the rest of the hangover
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
                    buf.extend(b"".join(preroll))  # un-clip the onset
                    preroll.clear()
                    buf.extend(pcm)
                    last_voice = now
                    voice_end = len(buf)
                else:
                    preroll.append(pcm)
            else:
                buf.extend(pcm)
                if peak >= stop_thr:
                    last_voice = now
                    voice_end = len(buf)
                elif now - last_voice > hang_s:
                    t_end = time.monotonic()  # you stopped talking
                    audio = bytes(buf[:min(len(buf), voice_end + tail)])  # trim trailing silence
                    events.emit(events.TRANSCRIBING,
                                text="Cartesia STT…" if use_cartesia else "Gemini STT…")
                    if use_cartesia:
                        text = await stt.transcribe(audio, api_key=self._cart["api_key"],
                                                    model=self._stt_model,
                                                    language=self._cart["language"])
                    else:
                        text = await stt.transcribe_gemini(audio, client=self.brain.client,
                                                           model=self.brain.model)
                    t_stt = time.monotonic()
                    if not text:
                        events.emit(events.ERROR, text="couldn't transcribe that",
                                    ms=_ms(t_end, t_stt))
                        return None, None
                    events.emit(events.TRANSCRIBED, text=text, ms=_ms(t_end, t_stt))
                    if self._debug.get("compare_stt") or self._debug.get("log_audio"):
                        # Capture the raw audio + run the OTHER STT model, off the hot path,
                        # so you can SEE what each model heard for the exact same utterance.
                        asyncio.create_task(self._debug_capture(audio, use_cartesia, text))
                    return text, t_end
        return None, None

    # --- debug: audio capture + model comparison --------------------------- #
    def _debug_capture(self, audio: bytes, used_cartesia: bool, live_text: str):
        async def _run():
            wav_path = self._save_wav(audio) if self._debug.get("log_audio") else None
            gemini_text = cartesia_text = None
            live = "cartesia" if used_cartesia else "gemini"
            if used_cartesia:
                cartesia_text = live_text
            else:
                gemini_text = live_text
            if self._debug.get("compare_stt"):
                try:  # transcribe the SAME audio with the other model
                    if used_cartesia:
                        gemini_text = await stt.transcribe_gemini(
                            audio, client=self.brain.client, model=self.brain.model)
                    else:
                        cartesia_text = await stt.transcribe(
                            audio, api_key=self._cart["api_key"], model=self._stt_model,
                            language=self._cart["language"])
                except Exception as e:
                    print(f"[compare] other STT failed: {e}", flush=True)
                events.emit(events.COMPARE,
                            text=f"Gemini: «{gemini_text or '—'}»   |   Cartesia: «{cartesia_text or '—'}»")
            self._log_compare({"gemini": gemini_text, "cartesia": cartesia_text,
                               "live": live, "wav": wav_path})
        return _run()

    def _save_wav(self, audio: bytes):
        try:
            d = self._debug.get("audio_dir", "logs/audio")
            os.makedirs(d, exist_ok=True)
            fn = os.path.join(d, time.strftime("%Y%m%d-%H%M%S") +
                              f"-{int(time.time() * 1000) % 1000:03d}.wav")
            with wave.open(fn, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(audio)
            return fn
        except Exception as e:
            print(f"[compare] save wav failed: {e}", flush=True)
            return None

    def _log_compare(self, row: dict):
        try:
            os.makedirs("logs", exist_ok=True)
            row = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), **row}
            with open("logs/stt_compare.jsonl", "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[compare] log failed: {e}", flush=True)

    async def _respond(self, text: str, t_end: float) -> None:
        t_stt_done = time.monotonic()  # transcript just came back
        events.emit(events.THINKING, text="Gemini Flash…")
        reply = await self.brain.ask(text)
        t_llm = time.monotonic()
        if self.brain.ignored:
            events.emit(events.DONE, text="🙊 ignored — stayed silent", ms=_ms(t_end, t_llm))
            return
        if not reply:
            events.emit(events.ERROR, text="no reply from the brain", ms=_ms(t_stt_done, t_llm))
            return
        events.emit(events.REPLY, text=reply, ms=_ms(t_stt_done, t_llm))

        first_audio = [0.0]

        def _first_audio():
            first_audio[0] = time.monotonic()
            events.emit(events.SPEAKING, text=reply, ms=_ms(t_llm, first_audio[0]))

        await tts.speak(reply, self.body, model=self._tts_model, voice_id=self._voice_id,
                        on_first_audio=_first_audio, **self._cart)
        # Headline latency = you stopped talking -> the robot STARTED speaking.
        ttf = first_audio[0] or time.monotonic()
        events.emit(events.DONE, text="you stopped → robot spoke", ms=_ms(t_end, ttf))

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

    async def _ack(self) -> None:
        """Bare attention call (just the name / 'wake up') -> a quick spoken acknowledgement."""
        line = random.choice(_ACKS)
        events.emit(events.REPLY, text=line)
        await tts.speak(line, self.body, model=self._tts_model, voice_id=self._voice_id,
                        **self._cart)
        self.gate.note_replied()

    async def run(self, stop_event=None) -> None:
        self.body.start()
        self._start_webview()
        try:
            await self._greet()
            while not (stop_event is not None and stop_event.is_set()):
                text, t_end = await self._listen_and_transcribe(stop_event)
                if not text:
                    continue
                # Only act when addressed (its name) or within the follow-up window;
                # background chatter is shown as SKIPPED, never silently swallowed.
                addressed, reason, clean = self.gate.decide(text)
                if not addressed:
                    events.emit(events.SKIPPED, text=f"{reason} · “{text[:48]}”")
                    continue
                if not clean.strip():
                    await self._ack()           # they only said the name -> acknowledge
                    continue
                await self._respond(clean, t_end)
                self.gate.note_replied()         # keep the follow-up window open
                if self.brain.slept:             # the model chose to go dormant
                    self.gate.sleep()
                    events.emit(events.SKIPPED,
                                text="asleep — say “Reachy” or “wake up” to wake me")
        finally:
            self.body.stop()
