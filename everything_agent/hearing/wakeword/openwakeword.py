"""openWakeWord adapter -- hands-free "Hey Reachy" detection on-device.

Reads the robot mic (via the shared MediaManager at `ctx.robot.media`), resamples
to 16 kHz, and runs openWakeWord locally (ONNX, no cloud, no key). `wait()`
blocks until the wake phrase scores above `threshold`, then returns True so the
loop hands off to STT.

Model selection (config):
  - `model`: a built-in openWakeWord name ("hey_jarvis", "alexa", ...), OR
  - `model_path`: a path to a custom model (e.g. models/hey_reachy.onnx).
A custom "Hey Reachy" model is the real goal; until one is trained you can run a
built-in phrase or fall back to the `mock` backend (always-listening). See
docs/wakeword.md for training a custom model.
"""
from __future__ import annotations

import asyncio
import logging

import numpy as np

from ...core.ports import WakeWord

log = logging.getLogger("everything_agent.hearing.wakeword")

_OWW_SR = 16000
_FRAME = 1280  # openWakeWord expects 80 ms frames at 16 kHz


class OpenWakeWord(WakeWord):
    def __init__(self, config, ctx):
        self.robot = ctx.robot
        cfg = config or {}
        self.phrase = cfg.get("phrase", "hey reachy")
        self.model_ref = cfg.get("model_path") or cfg.get("model", "hey_jarvis")
        self.threshold = float(cfg.get("threshold", 0.5))
        self._model = None
        self._key = None

    def _resolve_model_path(self) -> str:
        """Turn the config ref into a concrete .onnx path.

        Accepts a direct path (custom "hey reachy" model) or a built-in name
        like "hey_jarvis", which we match against openWakeWord's bundled models.
        """
        import os
        import openwakeword

        if os.path.exists(self.model_ref):
            return self.model_ref
        for path in openwakeword.get_pretrained_model_paths():
            if os.path.basename(path).startswith(self.model_ref):
                return path
        return self.model_ref   # let Model() surface a clear error if unknown

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        from openwakeword.model import Model

        path = self._resolve_model_path()
        try:                                   # API name varies across versions
            self._model = Model(wakeword_model_paths=[path])
        except TypeError:
            self._model = Model(wakeword_models=[path])
        self._key = list(self._model.models.keys())[0]
        log.info("openWakeWord ready (model=%s, threshold=%.2f)", self._key, self.threshold)
        return self._model

    async def wait(self) -> bool:
        media = self.robot.media if self.robot else None
        if media is None:                       # no mic (mock robot) -> always awake
            await asyncio.sleep(0.05)
            return True

        from scipy.signal import resample_poly

        model = self._ensure_model()
        mic_sr = media.get_input_audio_samplerate()
        try:
            media.start_recording()
        except Exception:  # noqa: BLE001
            pass

        buf = np.zeros(0, dtype=np.float32)
        while True:
            frame = await asyncio.to_thread(media.get_audio_sample)
            if frame is None or frame.size == 0:
                await asyncio.sleep(0.005)
                continue
            if frame.ndim == 2:
                frame = frame.mean(axis=1).astype(np.float32)
            if mic_sr != _OWW_SR:
                frame = resample_poly(frame, _OWW_SR, mic_sr).astype(np.float32)
            buf = np.concatenate([buf, frame])
            while buf.shape[0] >= _FRAME:
                chunk, buf = buf[:_FRAME], buf[_FRAME:]
                pcm = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)
                scores = await asyncio.to_thread(model.predict, pcm)
                if scores.get(self._key, 0.0) >= self.threshold:
                    log.info("wake word '%s' detected (%.2f)", self.phrase,
                             scores[self._key])
                    try:
                        model.reset()
                    except Exception:  # noqa: BLE001
                        pass
                    return True
