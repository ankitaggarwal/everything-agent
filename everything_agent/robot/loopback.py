"""Loopback robot -- a headless test body whose "mic" and "speaker" are buffers.

This is how Claude (or any developer) drives and observes the *whole* agent with
no hardware: feed synthesized audio into the loopback mic, run one cycle, and the
real Cartesia STT/TTS adapters run unchanged -- STT transcribes what we fed,
the brain replies, and TTS audio lands in a capture buffer we can write to a WAV.

It implements the same MediaManager surface the Reachy Mini exposes via `.media`
(get_input_audio_samplerate / start_recording / get_audio_sample /
get_output_audio_samplerate / push_audio_sample / start_playing / stop_playing),
so the adapters can't tell it apart from real hardware. Movement calls just log.
"""
from __future__ import annotations

import logging
import queue
import threading

import numpy as np

from ..core.ports import Robot

log = logging.getLogger("everything_agent.robot")

_IN_SR = 16000   # what we feed the "mic" (matches Cartesia STT's target rate)
_OUT_SR = 24000  # what the "speaker" runs at (matches Cartesia TTS raw rate)


class LoopbackMedia:
    """In-memory stand-in for the robot's shared MediaManager."""

    def __init__(self) -> None:
        self._mic: "queue.Queue[np.ndarray]" = queue.Queue()
        self._out: list[np.ndarray] = []
        self._out_lock = threading.Lock()

    # ---- input side (the adapters' STT reads these) ----
    def get_input_audio_samplerate(self) -> int:
        return _IN_SR

    def start_recording(self) -> None:
        pass

    def get_audio_sample(self):
        try:
            return self._mic.get_nowait()
        except queue.Empty:
            return None

    # ---- output side (the adapters' TTS writes these) ----
    def get_output_audio_samplerate(self) -> int:
        return _OUT_SR

    def push_audio_sample(self, frame: np.ndarray) -> None:
        with self._out_lock:
            self._out.append(np.asarray(frame, dtype=np.float32))

    def start_playing(self) -> None:
        pass

    def stop_playing(self) -> None:
        pass

    # ---- harness helpers (not part of the hardware surface) ----
    def feed(self, pcm: np.ndarray, trailing_silence_s: float = 1.6) -> None:
        """Enqueue `pcm` (float32 mono @ _IN_SR) as ~100 ms mic frames, then a
        tail of silence so Cartesia's endpointing finalizes the utterance --
        exactly what a real mic delivers after you stop speaking."""
        step = _IN_SR // 10
        pcm = np.asarray(pcm, dtype=np.float32)
        for i in range(0, len(pcm), step):
            self._mic.put(pcm[i:i + step].copy())
        for _ in range(int(trailing_silence_s * 10)):
            self._mic.put(np.zeros(step, dtype=np.float32))

    def drain_output(self) -> np.ndarray:
        """Return everything TTS pushed since the last drain, and reset."""
        with self._out_lock:
            out = np.concatenate(self._out) if self._out else np.zeros(0, np.float32)
            self._out = []
        return out


class LoopbackRobot(Robot):
    def __init__(self, config, ctx):
        self.config = config or {}
        self._media = LoopbackMedia()

    @property
    def media(self):
        return self._media

    def connect(self):
        log.info("[loopback-robot] connected (buffer mic + speaker)")

    def disconnect(self):
        log.info("[loopback-robot] disconnected")

    def move_head(self, yaw=0.0, pitch=0.0, roll=0.0, duration=0.5):
        log.info("[loopback-robot] move_head yaw=%.2f pitch=%.2f roll=%.2f", yaw, pitch, roll)

    def set_antennas(self, left=0.0, right=0.0):
        log.info("[loopback-robot] set_antennas left=%.2f right=%.2f", left, right)

    def reset(self):
        log.info("[loopback-robot] reset to neutral pose")
