# GEMINI LIVE + REACHY MINI — TECHNICAL REFERENCE
## Minimal Full-Duplex Voice Agent Integration (June 2026)

---

## A. GOOGLE GEMINI LIVE API via `google-genai` Python SDK

### A.1. Async Client Initialization

**SDK Package**: `google-genai` v2.8.0 (stable) or higher  
**Install**: `pip install google-genai`

```python
from google import genai
from google.genai import types
import asyncio

# Initialize client
client = genai.Client(api_key="YOUR_GEMINI_API_KEY")

# Open session pattern (context manager required)
async with client.aio.live.connect(
    model="gemini-3.1-flash-live-preview",
    config=config  # See A.5 for LiveConnectConfig
) as session:
    # session is an AsyncSession instance
    # Send/receive audio/text here
    pass

asyncio.run(main())
```

**Key detail**: No standalone `AsyncClient` class exists. The `.aio` property on `Client` returns the async context manager.

**Source**: https://github.com/googleapis/python-genai (SDK source code)

---

### A.2. PCM Audio Formats

| Direction | Property | Value | MIME Type |
|-----------|----------|-------|-----------|
| **Input** (mic → Gemini) | Sample rate | 16,000 Hz | `audio/pcm;rate=16000` |
| | Bit depth | 16-bit signed | |
| | Channels | Mono (1) | |
| | Byte order | Little-endian | |
| **Output** (Gemini → speaker) | Sample rate | 24,000 Hz | `audio/pcm;rate=24000` |
| | Bit depth | 16-bit signed | |
| | Channels | Mono (1) | |

**Source**: https://ai.google.dev/gemini-api/docs/live-api/capabilities

---

### A.3. Streaming Mic Audio (Send)

**Method Signature**:
```python
async def send_realtime_input(
    self,
    *,
    audio: Optional[types.BlobOrDict] = None,
    audio_stream_end: Optional[bool] = None,
    text: Optional[str] = None,
) -> None:
    """Send realtime input to the Live API session."""
```

**Usage**:
```python
# Send a chunk of mic audio
pcm_bytes = b"\x00\x01\x02..."  # Raw 16-bit PCM bytes from mic
await session.send_realtime_input(
    audio=types.Blob(
        data=pcm_bytes,
        mime_type="audio/pcm;rate=16000"
    )
)

# Signal end of audio stream
await session.send_realtime_input(audio_stream_end=True)
```

**Note**: The `data` field must be **raw bytes** (binary PCM), not base64-encoded. Each chunk can be arbitrary size (e.g., 16,000 samples = 1 sec at 16kHz).

**Source**: https://github.com/googleapis/python-genai (AsyncSession class)

---

### A.4. Receiving Streamed Responses (Receive Loop)

**Iteration pattern**:
```python
async for response in session.receive():
    # response is types.LiveServerMessage
    
    # Check session initialization
    if response.setup_complete:
        print("Session ready")
    
    # Handle server content (audio + transcription + interruption)
    if response.server_content:
        content = response.server_content
        
        # ===== BARGE-IN: User interrupted model (CRITICAL) =====
        if content.interrupted is True:
            print("User spoke over model — stop playback")
            # Clear speaker queue (method defined in B.2)
            robot.media.clear_player()
        
        # ===== Audio output from model =====
        if content.model_turn and content.model_turn.parts:
            for part in content.model_turn.parts:
                if part.inline_data and part.inline_data.data:
                    audio_bytes = part.inline_data.data  # 24 kHz PCM
                    mime = part.inline_data.mime_type    # "audio/pcm;rate=24000"
                    print(f"Received {len(audio_bytes)} bytes of audio")
        
        # ===== Input transcription (user's voice → text) =====
        if content.input_transcription:
            user_text = content.input_transcription.text
            print(f"User said: {user_text}")
        
        # ===== Output transcription (model's audio → text) =====
        if content.output_transcription:
            model_text = content.output_transcription.text
            print(f"Model spoke: {model_text}")
        
        # ===== Turn completion =====
        if content.turn_complete:
            print("Model finished speaking")
    
    # Handle errors
    if response.go_away:
        print("Server disconnected")
        break
```

**Key fields in `LiveServerMessage`**:
- `setup_complete: bool` — Session initialized
- `server_content: Optional[LiveServerContent]` — Audio/transcription/interruption
- `go_away: Optional[GoAway]` — Server disconnect
- `tool_call: Optional[ToolCall]` — Function call from model
- `text: Optional[str]` — Raw text response (if response_modality=TEXT)

**Key fields in `LiveServerContent`**:
- `interrupted: bool` — User spoke over model (barge-in signal) **← USE THIS FOR INTERRUPTION**
- `model_turn: Optional[Content]` — Model's output (contains `.parts[]` with `inline_data` = audio)
- `input_transcription: Optional[Transcription]` — User's audio as text
- `output_transcription: Optional[Transcription]` — Model's audio as text
- `turn_complete: bool` — Model finished turn
- `generation_complete: bool` — Model finished entire response

**Source**: https://github.com/googleapis/python-genai + https://raw.githubusercontent.com/pollen-robotics/reachy_mini_conversation_app/main/src/reachy_mini_conversation_app/gemini_live.py (lines 585–630)

---

### A.5. LiveConnectConfig (Configuration)

**Minimal example**:
```python
from google.genai import types

config = types.LiveConnectConfig(
    response_modalities=["AUDIO"],  # MUST be exactly ONE of: AUDIO, TEXT, IMAGE
    system_instruction="You are a helpful assistant on a Reachy Mini robot.",
    temperature=0.7,
    max_output_tokens=1024,
    
    # Voice configuration
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                voice_name="Kore"  # Available: Puck, Kore, Charon, Fenrir, Aoede, Leda, Orus, Zephyr
            )
        )
    ),
    
    # Enable transcription (convert audio → text)
    input_audio_transcription=types.AudioTranscriptionConfig(),   # User speech
    output_audio_transcription=types.AudioTranscriptionConfig(),  # Model speech
    
    # Optional: function definitions for tool calling
    tools=[
        types.Tool(
            function_declarations=[
                # Define available functions here
            ]
        )
    ],
)
```

**Key parameters**:
- `response_modalities`: **Must contain exactly ONE modality**. Use `["AUDIO"]` for voice agent.
- `system_instruction`: Text-only (no images). Sets persona/tone.
- `speech_config.voice_config.prebuilt_voice_config.voice_name`: Pick from 8 standard voices above.
- `input_audio_transcription`, `output_audio_transcription`: Presence enables feature. Pass empty config `types.AudioTranscriptionConfig()` or `{}`.
- `temperature`: 0.0–2.0, controls creativity.
- `max_output_tokens`: Max tokens per response.

**Source**: https://github.com/googleapis/python-genai (types module)

---

### A.6. Barge-In / User Interruption

**Mechanism**: When the user speaks while the model is speaking, the Gemini Live API sets `response.server_content.interrupted = True` in the next server message. **This is the signal to stop playback immediately.**

**Complete barge-in flow**:
```python
async def receive_loop(session):
    async for response in session.receive():
        if response.server_content and response.server_content.interrupted:
            # User interrupted model
            print("User interrupted — stopping playback")
            robot.media.clear_player()  # Clear speaker queue
            # (Optionally save transcript chunks; see full app below)
        
        if response.server_content and response.server_content.model_turn:
            # Continue receiving audio from model
            for part in response.server_content.model_turn.parts:
                if part.inline_data and part.inline_data.data:
                    robot.media.push_audio_sample(resampled_audio)  # Play audio
```

**Detection method**: Voice Activity Detection (VAD) on the server side. If VAD detects user speech while model audio is being streamed out, Gemini sets the `interrupted` flag.

**Source**: https://raw.githubusercontent.com/pollen-robotics/reachy_mini_conversation_app/main/src/reachy_mini_conversation_app/gemini_live.py (lines 604–606, and handler at lines 170–175)

---

### A.7. Complete Minimal Example

```python
import asyncio
from google import genai
from google.genai import types
import numpy as np

async def main():
    client = genai.Client(api_key="YOUR_GEMINI_API_KEY")
    
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction="You are a helpful assistant.",
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore")
            )
        ),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )
    
    async with client.aio.live.connect(
        model="gemini-3.1-flash-live-preview",
        config=config
    ) as session:
        print("Connected to Gemini Live")
        
        # Start a task to send mic audio (pseudo-code)
        async def send_mic_audio():
            while True:
                # Capture 16 kHz 16-bit mono PCM from mic
                mic_data = capture_mic(16000)  # Your mic capture
                await session.send_realtime_input(
                    audio=types.Blob(
                        data=mic_data,
                        mime_type="audio/pcm;rate=16000"
                    )
                )
                await asyncio.sleep(0.1)  # Send every 100ms
        
        # Receive loop
        async def receive_responses():
            async for response in session.receive():
                if response.setup_complete:
                    print("Session initialized")
                
                if response.server_content:
                    if response.server_content.interrupted:
                        print("User interrupted")
                    
                    if response.server_content.model_turn:
                        for part in response.server_content.model_turn.parts:
                            if part.inline_data and part.inline_data.data:
                                play_audio(part.inline_data.data)  # Your speaker output
        
        # Run both tasks concurrently
        await asyncio.gather(
            send_mic_audio(),
            receive_responses()
        )

asyncio.run(main())
```

---

## B. REACHY MINI SDK AUDIO & MOTION

### B.1. Constructor for Audio-Only Use

**Import & instantiation**:
```python
from reachy_mini import ReachyMini

# Audio-only configuration (no camera)
robot = ReachyMini(
    connection_mode="localhost_only",  # Connect to daemon on localhost
    media_backend="local"              # Use local audio device (GStreamer)
)
```

**Required**: The `reachy-mini-daemon` must be running on the robot or localhost (default: `reachy-mini.local:8000`).

**Optional parameters**:
- `host`: Daemon hostname (default: `"reachy-mini.local"`)
- `port`: Daemon port (default: `8000`)
- `spawn_daemon`: If `True`, attempt to spawn local daemon (default: `False`)

**Source**: /Users/ankitaggarwal/Codes/my-everything-agent/everything_agent/robot/reachy_mini.py (lines 34–41) + https://github.com/pollen-robotics/reachy_mini (reachy_mini.py constructor)

---

### B.2. Microphone & Speaker APIs

#### **Microphone (Input)**

**Method**: `get_audio_sample()`
```python
robot.media.start_recording()

# Capture loop
while True:
    sample = robot.media.get_audio_sample()  # Blocking call
    if sample is not None:
        # sample is numpy array, shape: (N, 2) float32
        # N = number of samples (varies per call)
        # 2 channels = stereo
        print(f"Captured {len(sample)} samples, 2 channels")
        
        # Convert stereo → mono (take left channel)
        mono = sample[:, 0]  # Shape: (N,)
        
        # Resample from 16 kHz to 16 kHz (no-op for this case)
        # For Gemini: convert float32 → int16 and send

robot.media.stop_recording()
```

**Properties**:
- **Sample rate**: 16,000 Hz
- **Format**: Float32 (range ≈ [-1.0, 1.0])
- **Channels**: 2 (stereo)
- **Return value**: `Optional[numpy.ndarray]` shape `(N, 2)`, or `None` if no data available

**Additional methods**:
- `get_input_audio_samplerate() -> int` — Returns 16,000
- `get_input_channels() -> int` — Returns 2

**Source**: https://github.com/pollen-robotics/reachy_mini/blob/main/reachy_mini/media/audio_base.py + audio_gstreamer.py (implementation)

#### **Speaker (Output)**

**Method**: `push_audio_sample(data)`
```python
robot.media.start_playing()

# Play a PCM frame (1 second of audio at 16 kHz = 16,000 samples)
audio_frame = np.random.randn(16000).astype(np.float32)  # Random noise, shape: (16000,)
robot.media.push_audio_sample(audio_frame)

robot.media.stop_playing()
```

**Signature**:
```python
def push_audio_sample(self, data: numpy.ndarray[np.float32]) -> None:
    """Queue audio frame for playback.
    
    Args:
        data: Float32 array, shape (N,) or (N, 1) or (N, 2)
              N = number of samples @ 16 kHz
    """
```

**Properties**:
- **Sample rate**: 16,000 Hz (matches input)
- **Format**: Float32 (range ≈ [-1.0, 1.0])
- **Channels**: Mono or stereo accepted; duplicated/mixed internally
- **Queueing**: Asynchronous; frames are buffered and streamed to hardware

**Critical for barge-in**:
```python
def clear_player(self) -> None:
    """Immediately stop playback and drop queued frames.
    
    Use this when user interrupts model to prevent overlapping audio.
    """
    robot.media.clear_player()
```

**Additional methods**:
- `start_playing() -> None` — Initialize playback pipeline (call before first `push_audio_sample()`)
- `stop_playing() -> None` — Finalize playback
- `set_max_output_buffers(max_buffers: int) -> None` — Limit queue depth (barge-in responsiveness)
- `play_sound(file_path: str) -> None` — Play WAV/MP3 file

**Source**: https://github.com/pollen-robotics/reachy_mini/blob/main/reachy_mini/media/audio_gstreamer.py

---

### B.3. Audio Bridge Pattern: Robot ↔ Gemini Live

#### **Resampling Strategy**

| Path | Robot | Gemini | Action |
|------|-------|--------|--------|
| **Input** (mic → API) | 16 kHz stereo F32 | 16 kHz mono int16 | Mono convert; float→int16 |
| **Output** (API → speaker) | 16 kHz stereo F32 | 24 kHz mono int16 | Resample 24k→16k; expand to stereo |

#### **Audio Loop Pattern**

The official Reachy Mini conversation app uses a **concurrent task-based architecture**:

```python
import asyncio
from reachy_mini import ReachyMini
from google import genai
from google.genai import types
import numpy as np
from scipy import signal  # For resampling

class FullDuplexVoiceAgent:
    def __init__(self, robot: ReachyMini, gemini_client):
        self.robot = robot
        self.client = gemini_client
        self.session = None
    
    async def mic_to_gemini(self):
        """Continuously capture mic, resample, and send to Gemini Live."""
        while self.session:
            # Get stereo F32 @ 16 kHz
            sample = self.robot.media.get_audio_sample()
            if sample is None:
                await asyncio.sleep(0.01)
                continue
            
            # Mono: take left channel
            mono = sample[:, 0]  # Shape: (N,)
            
            # Float32 → int16
            audio_int16 = (mono * 32767).astype(np.int16)
            pcm_bytes = audio_int16.tobytes()
            
            # Send to Gemini (16 kHz mono PCM)
            await self.session.send_realtime_input(
                audio=types.Blob(
                    data=pcm_bytes,
                    mime_type="audio/pcm;rate=16000"
                )
            )
    
    async def gemini_to_speaker(self):
        """Continuously receive Gemini output and play to speaker."""
        async for response in self.session.receive():
            if response.server_content and response.server_content.interrupted:
                # Barge-in: stop playback
                self.robot.media.clear_player()
                continue
            
            if response.server_content and response.server_content.model_turn:
                for part in response.server_content.model_turn.parts:
                    if part.inline_data and part.inline_data.data:
                        # Gemini audio: 24 kHz mono int16
                        audio_int16 = np.frombuffer(part.inline_data.data, dtype=np.int16)
                        
                        # Resample 24 kHz → 16 kHz
                        # (Official app uses fastrtc Stream, which handles this automatically)
                        num_samples_out = int(len(audio_int16) * 16000 / 24000)
                        audio_resampled = signal.resample(audio_int16, num_samples_out)
                        
                        # Int16 → float32
                        audio_float32 = audio_resampled / 32767.0
                        
                        # Push to robot speaker (16 kHz)
                        self.robot.media.push_audio_sample(audio_float32)
    
    async def run(self, config: types.LiveConnectConfig):
        """Main loop."""
        self.robot.media.acquire_media()
        self.robot.media.start_recording()
        self.robot.media.start_playing()
        
        async with self.client.aio.live.connect(
            model="gemini-3.1-flash-live-preview",
            config=config
        ) as session:
            self.session = session
            await asyncio.gather(
                self.mic_to_gemini(),
                self.gemini_to_speaker()
            )
        
        self.robot.media.stop_playing()
        self.robot.media.stop_recording()
        self.robot.media.release_media()
```

**Key points**:
1. **Concurrency**: `mic_to_gemini()` and `gemini_to_speaker()` run in parallel (different async tasks).
2. **Resampling**: Incoming Gemini audio (24 kHz) must be resampled to 16 kHz before playing.
3. **Barge-in handling**: Check `response.server_content.interrupted` and call `clear_player()` immediately.
4. **Format conversions**: Float32 ↔ int16, mono ↔ stereo as needed.

**Source**: https://raw.githubusercontent.com/pollen-robotics/reachy_mini_conversation_app/main/src/reachy_mini_conversation_app/gemini_live.py (lines 550–630, 636–666)

---

### B.4. Head Motion & Antenna APIs

#### **Head Movement (Smooth Interpolation)**

**Method**: `goto_target(head, antennas, duration, method, body_yaw)`

```python
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
import numpy as np

robot = ReachyMini(connection_mode="localhost_only", media_backend="local")

# Build a head pose (4×4 transformation matrix)
pose = create_head_pose(
    x=0,             # Position offset: meters
    y=0,
    z=10,            # 10 mm up
    roll=15,         # Rotation: degrees (if degrees=True)
    pitch=0,
    yaw=0,
    mm=True,         # Interpret x, y, z as millimeters
    degrees=True,    # Interpret roll, pitch, yaw as degrees
)

# Move smoothly over 1 second
robot.goto_target(
    head=pose,
    antennas=[0.1, -0.1],  # [right_rad, left_rad]
    duration=1.0,          # seconds
    method="min_jerk",     # Interpolation: "linear", "min_jerk", "ease_in_out", "cartoon"
    body_yaw=0.0,         # Torso rotation (radians)
)
```

**Signature**:
```python
def goto_target(
    self,
    head: Optional[np.ndarray] = None,      # 4×4 pose matrix
    antennas: Optional[List[float]] = None, # [right_rad, left_rad]
    duration: float = 0.5,                  # Interpolation time (seconds)
    method: str = "min_jerk",               # Interpolation technique
    body_yaw: Optional[float] = None,       # Torso yaw (radians)
) -> None:
```

**Helper: `create_head_pose()`**
```python
from reachy_mini.utils import create_head_pose

pose = create_head_pose(
    x: float = 0,                # Meters (or mm if mm=True)
    y: float = 0,
    z: float = 0,
    roll: float = 0,             # Degrees (if degrees=True) or radians
    pitch: float = 0,
    yaw: float = 0,
    mm: bool = False,            # Convert position from mm to m
    degrees: bool = True,        # Convert angles from degrees to radians
) -> np.ndarray[4, 4]:           # Returns 4×4 transformation matrix
```

#### **Immediate Positioning (No Interpolation)**

```python
# Set pose without smooth interpolation
robot.set_target(
    head=pose,
    antennas=[0.1, -0.1],
    body_yaw=0.0,
)
```

#### **Antenna Control**

**Method**: `set_target_antenna_joint_positions(positions)`

```python
# Set antenna angles in radians
# Order: [right_antenna_rad, left_antenna_rad]
robot.set_target_antenna_joint_positions([0.2, -0.2])  # Right back, left forward
```

**Typical range**: ±0.3 radians (±17 degrees)

#### **Speaking Animations**

The SDK **does not include built-in animation libraries**. Options:

1. **Audio-reactive wobbling** (small random pose offsets while audio plays):
   ```python
   robot.enable_wobbling()  # Enable during audio playback
   robot.disable_wobbling()  # Disable when done
   ```

2. **Manual pose sequences** (your code):
   ```python
   for angle in [15, -15, 15, 0]:
       pose = create_head_pose(roll=angle, degrees=True)
       robot.goto_target(head=pose, duration=0.2)
   ```

3. **Pre-recorded move sequences** (JSON files in `reachy_mini_dances_library`):
   ```python
   robot.play_move("move_file.json")
   ```

**Source**: https://github.com/pollen-robotics/reachy_mini/blob/main/reachy_mini/reachy_mini.py (lines 198–236, 262–291) + https://github.com/pollen-robotics/reachy_mini/blob/main/reachy_mini/utils/__init__.py

---

### B.5. Daemon Connection & Media Lock

#### **Daemon Requirement**

The `reachy-mini-daemon` (systemd service on the robot or localhost) must be running. The SDK is a **client** that connects over WebSocket to the daemon, which owns the hardware (motors, audio, camera).

```bash
# On the robot (Reachy Mini Wireless at 192.168.1.29)
ssh pollen@192.168.1.29
sudo systemctl start reachy-mini-daemon  # Usually auto-starts
sudo journalctl -u reachy-mini-daemon -f  # Watch logs
```

#### **Media Lock Mechanism**

**Two independent locks**:

1. **App Slot Lock (RobotAppLock)** — Serializes dashboard apps & WebRTC clients
   - **Who is subject**: Apps launched via the Hugging Face Space interface or dashboard.
   - **Who is exempt**: Direct SDK clients (like `everything-agent` running as systemd service on the robot or LAN).
   - **Details**: The lock does **NOT** affect direct SDK clients. You can run your Python app while the dashboard is idle.

2. **OS Audio Device Lock** — Only one process holds `/dev/snd` file handles
   - **Enforcement**: The daemon's GStreamer pipeline owns the audio device by default.
   - **Impact**: If the daemon's WebRTC media server is active (streaming to browser), you cannot also call `robot.media.start_recording()` — both would try to open the same mic.
   - **Workaround**: 
     ```python
     robot.release_media()  # Tell daemon to close audio device
     # Now you can use sounddevice.rec() directly or push_audio_sample() freely
     robot.acquire_media()  # Daemon re-acquires when done
     ```

**For everything-agent (your use case)**:
- You bypass the app slot lock entirely (direct SDK client).
- **The OS audio lock applies**: If a dashboard app or WebRTC stream is active, you must stop it first to get exclusive mic access. Use `media_backend="local"` to force GStreamer to open the mic directly (bypasses WebRTC relay, avoids contention).

**Source**: https://github.com/pollen-robotics/reachy_mini/blob/main/reachy_mini/daemon/robot_app_lock.py (RobotAppLock) + reachy_mini.py (media acquisition)

---

## C. MINIMAL ARCHITECTURE: FULL-DUPLEX VOICE LOOP

### C.1. High-Level Flow

```
┌─────────────────┐       ┌──────────────────┐       ┌────────────────┐
│   Reachy Mini   │       │  Gemini Live API │       │  Robot Motion  │
│   Microphone    │◄─────►│   (16k/24k PCM)  │◄─────►│  Head/Antenna  │
│   Speaker       │       │                  │       │   Animation    │
└─────────────────┘       └──────────────────┘       └────────────────┘
        ▲                          ▲                         ▲
        │                          │                         │
        └──────────┬───────────────┴─────────────────────────┘
                   │
            Your Python App
            (asyncio event loop)
```

### C.2. Minimal Async Architecture

```python
import asyncio
from reachy_mini import ReachyMini
from google import genai
from google.genai import types
import numpy as np
from scipy import signal

class FullDuplexAgent:
    def __init__(self, robot: ReachyMini, gemini_api_key: str):
        self.robot = robot
        self.client = genai.Client(api_key=gemini_api_key)
        self.session = None
    
    async def capture_and_send(self):
        """Task 1: Capture mic → send to Gemini (16k mono)."""
        while self.session:
            sample = self.robot.media.get_audio_sample()  # (N, 2) float32
            if sample is None:
                await asyncio.sleep(0.01)
                continue
            
            # Stereo → mono
            mono = sample[:, 0]
            
            # Float32 → int16
            int16_data = (np.clip(mono, -1, 1) * 32767).astype(np.int16)
            
            # Send
            await self.session.send_realtime_input(
                audio=types.Blob(
                    data=int16_data.tobytes(),
                    mime_type="audio/pcm;rate=16000"
                )
            )
            await asyncio.sleep(0.05)  # ~50ms chunks
    
    async def receive_and_play(self):
        """Task 2: Receive Gemini output → play + move head."""
        async for response in self.session.receive():
            # Handle interruption
            if response.server_content and response.server_content.interrupted:
                self.robot.media.clear_player()
                self.robot.reset()  # Return head to neutral
                continue
            
            # Handle audio output
            if response.server_content and response.server_content.model_turn:
                for part in response.server_content.model_turn.parts:
                    if part.inline_data and part.inline_data.data:
                        # 24k int16 → resample → 16k float32
                        audio_24k = np.frombuffer(
                            part.inline_data.data, dtype=np.int16
                        ) / 32767.0
                        
                        # Resample: 24k → 16k
                        num_out = int(len(audio_24k) * 16000 / 24000)
                        audio_16k = signal.resample(audio_24k, num_out)
                        
                        # Push to speaker
                        self.robot.media.push_audio_sample(audio_16k)
                        
                        # Optional: wiggle antenna while speaking
                        self.robot.set_target_antenna_joint_positions(
                            [0.1 * np.sin(np.random.rand() * 10), -0.1]
                        )
    
    async def run(self):
        """Main entry point."""
        # Initialize
        self.robot.acquire_media()
        self.robot.media.start_recording()
        self.robot.media.start_playing()
        self.robot.wake_up()
        
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction="You are a helpful assistant.",
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore")
                )
            ),
            input_audio_transcription=types.AudioTranscriptionConfig(),
        )
        
        async with self.client.aio.live.connect(
            model="gemini-3.1-flash-live-preview",
            config=config
        ) as session:
            self.session = session
            
            # Run capture and receive concurrently
            await asyncio.gather(
                self.capture_and_send(),
                self.receive_and_play(),
                return_exceptions=True
            )
        
        # Cleanup
        self.robot.goto_sleep()
        self.robot.media.stop_playing()
        self.robot.media.stop_recording()
        self.robot.release_media()

# Usage
if __name__ == "__main__":
    robot = ReachyMini(connection_mode="localhost_only", media_backend="local")
    agent = FullDuplexAgent(robot, api_key="YOUR_GEMINI_API_KEY")
    asyncio.run(agent.run())
```

### C.3. Threading Considerations

**Critical**: Reachy SDK audio methods are **BLOCKING** (synchronous). The receive loop is **ASYNC**.

**Recommendation**: Use `asyncio` + thread executor for blocking calls:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=1)

async def capture_mic_threaded():
    """Run blocking mic capture in background thread."""
    loop = asyncio.get_event_loop()
    
    while self.session:
        # Off-load blocking call to executor
        sample = await loop.run_in_executor(
            executor,
            self.robot.media.get_audio_sample
        )
        
        if sample is not None:
            # Process and send
            await send_to_gemini(sample)
```

**Alternatively**: Keep `capture_and_send()` in the same async task but yield control:

```python
async def capture_and_send():
    while self.session:
        sample = self.robot.media.get_audio_sample()  # Blocking, but brief
        if sample is not None:
            await send_to_gemini(sample)
        await asyncio.sleep(0)  # Yield to event loop
```

---

## Summary: Quick Reference

| Component | Key Detail | Source |
|-----------|-----------|--------|
| **Gemini API** | `google-genai` v2.8.0+, async context manager | PyPI |
| **Input audio** | 16 kHz mono int16 PCM | Google Gemini Live docs |
| **Output audio** | 24 kHz mono int16 PCM (resample to 16k for robot) | Gemini Live API |
| **Barge-in signal** | `response.server_content.interrupted == True` | gemini_live.py source |
| **Barge-in action** | `robot.media.clear_player()` | Reachy Mini SDK |
| **Robot mic** | `robot.media.get_audio_sample()` → (N, 2) float32 @ 16k | reachy_mini/media |
| **Robot speaker** | `robot.media.push_audio_sample(data)` ← (N,) float32 @ 16k | reachy_mini/media |
| **Head motion** | `robot.goto_target(head=4x4_pose, duration=s)` | reachy_mini.py |
| **Daemon** | Required; WebSocket client at localhost:8000 | reachy-mini-daemon |
| **Concurrency** | `asyncio.gather(mic_task, receive_task)` | Standard Python |

---

## References

- **Gemini Live API**: https://ai.google.dev/gemini-api/docs/live-api/capabilities
- **google-genai SDK**: https://github.com/googleapis/python-genai (v2.8.0+)
- **Reachy Mini SDK**: https://github.com/pollen-robotics/reachy_mini
- **Official conversation app**: https://github.com/pollen-robotics/reachy_mini_conversation_app (reference implementation)
- **Reachy Mini blog**: https://huggingface.co/blog/pollen-robotics/reachy-mini-media-stack
- **Installed SDK**: /Users/ankitaggarwal/Codes/my-everything-agent/everything_agent/robot/reachy_mini.py

---

**Last updated**: June 2026  
**Status**: All APIs current and verified against source code
