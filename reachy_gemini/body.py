"""The 'body': where the voice physically goes in and comes out.

One audio engine (sounddevice) for both backends -- the laptop and the robot only
differ in which audio device they point at and whether there's a head to move:

  - LaptopBody : default mic + speakers.            No motion.
  - ReachyBody : the robot's pipewire audio nodes   + head/antenna motion.
                 (reachymini_audio_src / _sink, which are shareable, so we never
                  fight the daemon for the raw ALSA device.)

Audio contract the rest of the app relies on:
  read_mic()        -> 16 kHz MONO 16-bit PCM bytes (or b"" if nothing yet)
  play(pcm_24k)     <- 24 kHz MONO 16-bit PCM bytes straight from Gemini
  clear_playback()  -> drop anything still queued (barge-in)
  set_speaking(b)   -> motion hook: come alive while talking, settle when done
"""
from __future__ import annotations

import queue
import threading
import time

import numpy as np

GEMINI_IN_RATE = 16000    # what we must send up to Gemini
GEMINI_OUT_RATE = 24000   # what Gemini streams back down


def make_body(cfg: dict) -> "Body":
    backend = cfg.get("body", {}).get("backend", "local")
    if backend == "reachy":
        return ReachyBody(cfg)
    if backend == "local":
        return LaptopBody(cfg)
    raise SystemExit(f"Unknown body.backend: {backend!r} (use 'local' or 'reachy')")


def _resample_mono(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or len(audio) == 0:
        return audio
    n_out = max(1, round(len(audio) * dst_rate / src_rate))
    idx = (np.arange(n_out) * (len(audio) - 1) / max(1, n_out - 1)).astype(np.int64)
    return audio[idx]


class Body:
    """A sounddevice mic+speaker. Subclasses pick the devices and add motion."""

    name = "base"

    def __init__(self, cfg: dict, *, in_device=None, out_device=None,
                 in_channels=1, out_rate=GEMINI_OUT_RATE, out_channels=1):
        self.chunk_ms = cfg.get("audio", {}).get("chunk_ms", 50)
        self._in_device = in_device
        self._out_device = out_device
        self._in_channels = in_channels
        self._out_rate = out_rate
        self._out_channels = out_channels
        self._in_q: "queue.Queue[bytes]" = queue.Queue()
        self._out_buf = bytearray()
        self._out_lock = threading.Lock()
        self._in_stream = None
        self._out_stream = None

    def start(self) -> None:
        import sounddevice as sd  # lazy: import only when a real body starts

        block = int(GEMINI_IN_RATE * self.chunk_ms / 1000)

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
            samplerate=GEMINI_IN_RATE, channels=self._in_channels, dtype="int16",
            blocksize=block, device=self._in_device, callback=on_mic,
        )
        self._out_stream = sd.RawOutputStream(
            samplerate=self._out_rate, channels=self._out_channels, dtype="int16",
            blocksize=block, device=self._out_device, callback=on_speaker,
        )
        self._in_stream.start()
        self._out_stream.start()

    def stop(self) -> None:
        for s in (self._in_stream, self._out_stream):
            if s is not None:
                try:
                    s.stop(); s.close()
                except Exception:
                    pass

    def read_mic(self) -> bytes:
        chunks = []
        try:
            while True:
                chunks.append(self._in_q.get_nowait())
        except queue.Empty:
            pass
        if not chunks:
            return b""
        raw = b"".join(chunks)
        if self._in_channels == 1:
            return raw
        # de-interleave: keep the left channel as mono
        frame = np.frombuffer(raw, dtype=np.int16).reshape(-1, self._in_channels)
        return frame[:, 0].tobytes()

    def play(self, pcm_24k: bytes) -> None:
        audio = np.frombuffer(pcm_24k, dtype=np.int16)
        audio = _resample_mono(audio, GEMINI_OUT_RATE, self._out_rate)
        if self._out_channels > 1:
            audio = np.repeat(audio[:, None], self._out_channels, axis=1).reshape(-1)
        with self._out_lock:
            self._out_buf.extend(audio.astype(np.int16).tobytes())

    def clear_playback(self) -> None:
        with self._out_lock:
            self._out_buf.clear()

    def set_speaking(self, speaking: bool) -> None:  # no body to move by default
        pass


# --------------------------------------------------------------------------- #
class LaptopBody(Body):
    name = "laptop"

    def __init__(self, cfg: dict):
        # default devices; play back at Gemini's native 24 kHz mono
        super().__init__(cfg, out_rate=GEMINI_OUT_RATE, out_channels=1)


# --------------------------------------------------------------------------- #
class ReachyBody(Body):
    """Robot audio via pipewire nodes + a head that wiggles while it talks."""

    name = "reachy"

    def __init__(self, cfg: dict):
        # The robot's pipewire nodes run at 16 kHz stereo (both directions).
        super().__init__(
            cfg,
            in_device="reachymini_audio_src", in_channels=2,
            out_device="reachymini_audio_sink", out_rate=16000, out_channels=2,
        )
        self.robot = None
        self._speaking = False
        self._motion = None

    def start(self) -> None:
        super().start()  # audio first -- it must work even if motion doesn't
        try:
            from reachy_mini import ReachyMini

            # no_media: we own audio via sounddevice; the SDK is only for motion.
            self.robot = ReachyMini(connection_mode="localhost_only", media_backend="no_media")
        except Exception as exc:
            print(f"(motion disabled: {exc})", flush=True)
            self.robot = None

    def stop(self) -> None:
        self._speaking = False
        if self._motion is not None:
            self._motion.join(timeout=1.0)
        if self.robot is not None:
            try:
                self.robot.set_target_antenna_joint_positions([0.0, 0.0])
            except Exception:
                pass
            # The SDK client spawns non-daemon threads; disconnect so we can exit.
            try:
                self.robot.client.disconnect()
            except Exception:
                pass
            self.robot = None
        super().stop()

    def set_speaking(self, speaking: bool) -> None:
        if self.robot is None or speaking == self._speaking:
            return
        self._speaking = speaking
        if speaking and (self._motion is None or not self._motion.is_alive()):
            self._motion = threading.Thread(target=self._wiggle, daemon=True)
            self._motion.start()

    def _wiggle(self) -> None:
        """Gentle antenna life while talking. Antennas only -- safe for wifi."""
        phase = 0.0
        while self._speaking and self.robot is not None:
            phase += 0.6
            r = 0.18 * np.sin(phase)
            try:
                self.robot.set_target_antenna_joint_positions([float(r), float(-r)])
            except Exception:
                break
            time.sleep(0.12)
        try:
            if self.robot is not None:
                self.robot.set_target_antenna_joint_positions([0.0, 0.0])
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Robot via the daemon's AppManager: we are handed a ready ReachyMini with the
# LOCAL media backend, so we use the SDK's own audio (which routes through the
# daemon to the real speaker) -- no sounddevice, no device tug-of-war.
# --------------------------------------------------------------------------- #
class RobotMediaBody:
    name = "reachy-app"

    def __init__(self, reachy_mini, cfg: dict):
        self.robot = reachy_mini
        self._speaking = False
        self._motion = None

    def start(self) -> None:
        self.robot.media.start_recording()
        self.robot.media.start_playing()
        try:
            self.robot.wake_up()
        except Exception:
            pass

    def stop(self) -> None:
        self._speaking = False
        if self._motion is not None:
            self._motion.join(timeout=1.0)
        try:
            self.robot.media.stop_recording()
            self.robot.media.stop_playing()
        except Exception:
            pass

    def read_mic(self) -> bytes:
        sample = self.robot.media.get_audio_sample()  # (N,2) float32 @16k or None
        if sample is None or len(sample) == 0:
            return b""
        mono = sample[:, 0] if sample.ndim == 2 else sample
        return (np.clip(mono, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

    def play(self, pcm_24k: bytes) -> None:
        audio = np.frombuffer(pcm_24k, dtype=np.int16).astype(np.float32) / 32767.0
        audio = _resample_mono(audio, GEMINI_OUT_RATE, 16000)
        self.robot.media.push_audio_sample(audio)

    def clear_playback(self) -> None:
        try:
            self.robot.media.clear_player()
        except Exception:
            pass

    def set_speaking(self, speaking: bool) -> None:
        if speaking == self._speaking:
            return
        self._speaking = speaking
        if speaking and (self._motion is None or not self._motion.is_alive()):
            self._motion = threading.Thread(target=self._wiggle, daemon=True)
            self._motion.start()

    def _wiggle(self) -> None:
        phase = 0.0
        while self._speaking:
            phase += 0.6
            r = 0.18 * np.sin(phase)
            try:
                self.robot.set_target_antenna_joint_positions([float(r), float(-r)])
            except Exception:
                break
            time.sleep(0.12)
        try:
            self.robot.set_target_antenna_joint_positions([0.0, 0.0])
        except Exception:
            pass
