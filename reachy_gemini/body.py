"""The 'body': where the voice physically goes in and comes out.

Two backends behind one tiny interface:
  - LaptopBody  : your computer's mic + speakers (sounddevice). No motion. For dev.
  - ReachyBody  : the Reachy Mini robot's mic + speaker + head/antenna motion.

Audio contract (so the rest of the app never thinks about hardware):
  read_mic()        -> 16 kHz mono 16-bit PCM bytes (or b"" if nothing yet)
  play(pcm_24k)     <- 24 kHz mono 16-bit PCM bytes straight from Gemini
  clear_playback()  -> drop anything still queued (used for barge-in)
  set_speaking(b)   -> motion hook: come alive while talking, settle when done
"""
from __future__ import annotations

import queue
import threading

import numpy as np

MIC_RATE = 16000   # what Gemini wants in
TTS_RATE = 24000   # what Gemini sends out


def make_body(cfg: dict) -> "Body":
    backend = cfg.get("body", {}).get("backend", "local")
    if backend == "reachy":
        return ReachyBody(cfg)
    if backend == "local":
        return LaptopBody(cfg)
    raise SystemExit(f"Unknown body.backend: {backend!r} (use 'local' or 'reachy')")


class Body:
    name = "base"

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def read_mic(self) -> bytes: return b""
    def play(self, pcm_24k: bytes) -> None: ...
    def clear_playback(self) -> None: ...
    def set_speaking(self, speaking: bool) -> None: ...


# --------------------------------------------------------------------------- #
# Laptop: sounddevice mic in @16k, speakers out @24k. Callback-driven so we can
# clear the output buffer instantly for barge-in.
# --------------------------------------------------------------------------- #
class LaptopBody(Body):
    name = "laptop"

    def __init__(self, cfg: dict):
        self.chunk_ms = cfg.get("audio", {}).get("chunk_ms", 50)
        self._in_q: "queue.Queue[bytes]" = queue.Queue()
        self._out_buf = bytearray()
        self._out_lock = threading.Lock()
        self._in_stream = None
        self._out_stream = None

    def start(self) -> None:
        import sounddevice as sd  # imported lazily: only the laptop backend needs it

        block = int(MIC_RATE * self.chunk_ms / 1000)

        def on_mic(indata, frames, time_info, status):
            self._in_q.put(bytes(indata))

        def on_speaker(outdata, frames, time_info, status):
            need = len(outdata)
            with self._out_lock:
                have = min(need, len(self._out_buf))
                outdata[:have] = bytes(self._out_buf[:have])
                del self._out_buf[:have]
            if have < need:
                outdata[have:] = b"\x00" * (need - have)

        self._in_stream = sd.RawInputStream(
            samplerate=MIC_RATE, channels=1, dtype="int16",
            blocksize=block, callback=on_mic,
        )
        self._out_stream = sd.RawOutputStream(
            samplerate=TTS_RATE, channels=1, dtype="int16",
            blocksize=block, callback=on_speaker,
        )
        self._in_stream.start()
        self._out_stream.start()

    def stop(self) -> None:
        for s in (self._in_stream, self._out_stream):
            if s is not None:
                s.stop()
                s.close()

    def read_mic(self) -> bytes:
        chunks = []
        try:
            while True:
                chunks.append(self._in_q.get_nowait())
        except queue.Empty:
            pass
        return b"".join(chunks)

    def play(self, pcm_24k: bytes) -> None:
        with self._out_lock:
            self._out_buf.extend(pcm_24k)

    def clear_playback(self) -> None:
        with self._out_lock:
            self._out_buf.clear()


# --------------------------------------------------------------------------- #
# Reachy Mini: SDK mic/speaker (both @16k float32 stereo) + head/antenna motion.
# Gemini's 24k output is resampled down to the robot's 16k before playback.
# --------------------------------------------------------------------------- #
class ReachyBody(Body):
    name = "reachy"

    def __init__(self, cfg: dict):
        self._cfg = cfg
        self.robot = None

    def start(self) -> None:
        from reachy_mini import ReachyMini  # only the robot has this SDK

        self.robot = ReachyMini(connection_mode="localhost_only", media_backend="local")
        # Take the audio device away from the daemon's WebRTC pipeline, then open it.
        try:
            self.robot.acquire_media()
        except Exception:
            pass
        self.robot.media.start_recording()
        self.robot.media.start_playing()
        try:
            self.robot.wake_up()
        except Exception:
            pass

    def stop(self) -> None:
        if self.robot is None:
            return
        try:
            self.robot.media.stop_recording()
            self.robot.media.stop_playing()
        except Exception:
            pass
        try:
            self.robot.goto_sleep()
        except Exception:
            pass
        try:
            self.robot.release_media()
        except Exception:
            pass

    def read_mic(self) -> bytes:
        sample = self.robot.media.get_audio_sample()  # (N, 2) float32 @16k, or None
        if sample is None or len(sample) == 0:
            return b""
        mono = sample[:, 0] if sample.ndim == 2 else sample
        return (np.clip(mono, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

    def play(self, pcm_24k: bytes) -> None:
        audio = np.frombuffer(pcm_24k, dtype=np.int16).astype(np.float32) / 32767.0
        # 24k -> 16k by simple ratio resample (good enough for speech).
        n_out = max(1, round(len(audio) * MIC_RATE / TTS_RATE))
        idx = (np.arange(n_out) * (len(audio) - 1) / max(1, n_out - 1)).astype(np.int64)
        self.robot.media.push_audio_sample(audio[idx])

    def clear_playback(self) -> None:
        try:
            self.robot.media.clear_player()
        except Exception:
            pass

    def set_speaking(self, speaking: bool) -> None:
        # Best-effort liveliness: wobble while talking, settle when quiet.
        try:
            if speaking:
                self.robot.enable_wobbling()
            else:
                self.robot.disable_wobbling()
                self.robot.reset()
        except Exception:
            pass
