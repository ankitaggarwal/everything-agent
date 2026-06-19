"""Reachy Kid Buddy -- a real-time, SEEING voice playmate, powered by Gemini Live.

Runs ON the robot, continuously. Streams the mic + the Pi camera to Gemini's Live
API (native audio + vision -- no STT to mangle a child's speech) and plays Gemini's
spoken reply back through the speaker, in real time. Gemini can also call a
`turn_head` tool to rotate Reachy's head/camera to look around.

- Audio is half-duplex (mic muted while Reachy talks) so the no-AEC mic doesn't
  feed the speaker back and make it talk to itself.
- Camera frames come straight from rpicam (the daemon's local IPC pipeline is
  broken, but the camera device itself is free).
- Reconnects automatically if the Live session ends, so it's always listening.

Runs as the kid-buddy systemd service (see deploy/kid-buddy.service).
"""
from __future__ import annotations

import asyncio
import logging
import os
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("kid_buddy")

GEM_IN_SR = 16000     # Gemini Live wants 16 kHz PCM in
GEM_OUT_SR = 24000    # Gemini Live sends 24 kHz PCM out
MODEL = "gemini-2.5-flash-native-audio-latest"
VOICE = "Puck"        # upbeat, playful Gemini voice

PERSONA = (
    "You are Reachy, a cheerful, playful little desk robot and the best buddy of a "
    "young child (around 5). Talk like a warm, fun, slightly silly friend. Keep every "
    "reply SHORT -- one or two simple sentences. Be excited and curious, ask little "
    "questions, tell tiny jokes, play along with whatever they say. Use simple kid "
    "words, never anything scary, mean, or inappropriate. If you didn't catch "
    "something, cheerfully ask them to say it again. Never give long lectures.\n"
    "You CAN SEE through your camera: when the child shows you a toy or points at "
    "something, react to what you actually see -- name it, its colors, what it does.\n"
    "You can TURN YOUR HEAD with the turn_head tool. Use it when the child asks you "
    "to look somewhere, or when what they're talking about isn't in your view -- turn "
    "left/right/up/down to look around, then tell them what you see. Keep it playful "
    "('Ooh, let me look over here!')."
)


async def main() -> int:
    from google import genai
    from google.genai import types
    from scipy.signal import resample_poly
    from scipy.spatial.transform import Rotation as R
    from reachy_mini import ReachyMini

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        log.error("GEMINI_API_KEY not set"); return 2

    log.info("connecting to robot...")
    mini = ReachyMini(connection_mode="localhost_only", media_backend="local")
    for fn in (mini.acquire_media, mini.wake_up, lambda: mini.media.start_playing()):
        try: fn()
        except Exception as e: log.warning("setup step failed: %s", e)
    media = mini.media
    mic_sr = media.get_input_audio_samplerate()
    out_sr = media.get_output_audio_samplerate()
    log.info("robot mic=%dHz speaker=%dHz", mic_sr, out_sr)

    def move_head_sync(direction: str) -> None:
        yaw = pitch = 0.0
        d = (direction or "center").lower()
        if d == "left": yaw = 0.6
        elif d == "right": yaw = -0.6
        elif d == "up": pitch = -0.35
        elif d == "down": pitch = 0.35
        try:
            pose = np.eye(4)
            pose[:3, :3] = R.from_euler("ZYX", [yaw, pitch, 0.0]).as_matrix()
            mini.goto_target(head=pose, duration=0.6, body_yaw=None)
        except Exception as e:  # noqa: BLE001
            log.warning("head move failed: %s", e)

    client = genai.Client(api_key=api_key)
    turn_head = types.Tool(function_declarations=[types.FunctionDeclaration(
        name="turn_head",
        description=("Turn your head and camera to look in a direction. Use it when the "
                     "child asks you to look somewhere or when what they mean isn't in "
                     "view. After turning you'll see a new view."),
        parameters=types.Schema(type=types.Type.OBJECT, properties={
            "direction": types.Schema(type=types.Type.STRING,
                                      enum=["left", "right", "up", "down", "center"])},
            required=["direction"]))])
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=PERSONA,
        speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE))),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        tools=[turn_head],
    )

    speaking_until = {"t": 0.0}

    async def run_session(session):
        async def send_mic():
            try: media.start_recording()
            except Exception: pass
            batch, n, target = [], 0, GEM_IN_SR // 10
            while True:
                frame = await asyncio.to_thread(media.get_audio_sample)
                if frame is None or getattr(frame, "size", 0) == 0:
                    await asyncio.sleep(0.005); continue
                if frame.ndim == 2:
                    frame = frame.mean(axis=1).astype(np.float32)
                if time.monotonic() < speaking_until["t"]:
                    continue
                if mic_sr != GEM_IN_SR:
                    frame = resample_poly(frame, GEM_IN_SR, mic_sr).astype(np.float32)
                batch.append(frame); n += frame.shape[0]
                if n >= target:
                    pcm16 = (np.clip(np.concatenate(batch), -1, 1) * 32767).astype("<i2").tobytes()
                    await session.send_realtime_input(
                        audio=types.Blob(data=pcm16, mime_type=f"audio/pcm;rate={GEM_IN_SR}"))
                    batch, n = [], 0

        async def recv():
            while True:
                async for msg in session.receive():
                    sc = getattr(msg, "server_content", None)
                    if sc is not None:
                        it = getattr(sc, "input_transcription", None)
                        ot = getattr(sc, "output_transcription", None)
                        if it and getattr(it, "text", None): log.info("🧒 heard: %s", it.text)
                        if ot and getattr(ot, "text", None): log.info("🤖 says : %s", ot.text)
                    tc = getattr(msg, "tool_call", None)
                    if tc is not None:
                        for fc in (tc.function_calls or []):
                            if fc.name == "turn_head":
                                d = (fc.args or {}).get("direction", "center")
                                log.info("👀 turning head %s", d)
                                await asyncio.to_thread(move_head_sync, d)
                            await session.send_tool_response(function_responses=[
                                types.FunctionResponse(id=fc.id, name=fc.name,
                                                       response={"ok": True})])
                    data = getattr(msg, "data", None)
                    if data:
                        frame = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
                        if out_sr != GEM_OUT_SR:
                            frame = resample_poly(frame, out_sr, GEM_OUT_SR).astype(np.float32)
                        speaking_until["t"] = max(speaking_until["t"], time.monotonic()) + \
                            frame.shape[0] / float(out_sr) + 0.3
                        await asyncio.to_thread(media.push_audio_sample, frame)

        async def send_video():
            try:
                proc = await asyncio.create_subprocess_exec(
                    "rpicam-vid", "-t", "0", "--codec", "mjpeg", "--width", "640",
                    "--height", "480", "--framerate", "5", "--nopreview", "-o", "-",
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
            except Exception as e:  # noqa: BLE001
                log.warning("camera unavailable (%s) -- voice only", e); return
            log.info("👁  camera streaming to Gemini")
            buf, last = b"", 0.0
            try:
                while True:
                    chunk = await proc.stdout.read(65536)
                    if not chunk:
                        break
                    buf += chunk
                    while True:
                        soi = buf.find(b"\xff\xd8")
                        eoi = buf.find(b"\xff\xd9", soi + 2) if soi >= 0 else -1
                        if soi < 0 or eoi < 0:
                            break
                        jpeg = buf[soi:eoi + 2]; buf = buf[eoi + 2:]
                        if time.monotonic() - last >= 1.0:
                            last = time.monotonic()
                            await session.send_realtime_input(
                                video=types.Blob(data=jpeg, mime_type="image/jpeg"))
            finally:
                try: proc.kill()
                except Exception: pass

        await asyncio.gather(send_mic(), recv(), send_video())

    # Reconnect loop -- keep Reachy always listening.
    try:
        while True:
            try:
                async with client.aio.live.connect(model=MODEL, config=config) as session:
                    log.info("Gemini Live connected. Reachy is listening + watching! (voice=%s)", VOICE)
                    await run_session(session)
            except (KeyboardInterrupt, asyncio.CancelledError):
                raise
            except Exception as e:  # noqa: BLE001
                log.warning("session ended (%s) -- reconnecting in 2s", str(e)[:120])
                await asyncio.sleep(2)
    finally:
        try: mini.goto_sleep(); mini.release_media()
        except Exception: pass
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
