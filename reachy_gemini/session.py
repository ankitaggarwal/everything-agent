"""Thin wrapper around the Gemini Live realtime session.

This is the 'black box' in the diagram: we open one bidirectional connection,
stream raw mic PCM up, and stream audio (and transcripts) back down.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from google import genai
from google.genai import types


def build_config(cfg: dict) -> types.LiveConnectConfig:
    g = cfg["gemini"]
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=g.get("persona", "You are a friendly robot."),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=g.get("voice", "Kore")
                )
            )
        ),
        # Give us text alongside the audio so we can show what's being said.
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        # No tools defined yet -> every turn takes the "just talk" path.
    )


@asynccontextmanager
async def open_session(cfg: dict):
    g = cfg["gemini"]
    client = genai.Client(api_key=g["api_key"])
    async with client.aio.live.connect(model=g["model"], config=build_config(cfg)) as session:
        yield session
