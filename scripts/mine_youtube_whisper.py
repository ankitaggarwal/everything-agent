"""Mine YouTube audio for 'reachy' utterances via Whisper word timestamps.

Transcribes each wav with word-level timestamps, finds words that sound like the
wake word (reachy/reach/reechy/richie/ritchie), and cuts a clip around each.
Saves to wakeword_training/youtube_clips/.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

OUT = "wakeword_training/youtube_clips"
os.makedirs(OUT, exist_ok=True)
SR = 16000
# words Whisper might use for a spoken "Reachy"
PAT = re.compile(r"\b(reachy|reachie|reechy|reachi|reach|richie|ritchie|reaching)\b", re.I)

model = WhisperModel("base.en", device="cpu", compute_type="int8")
total = 0
for path in sorted(glob.glob("wakeword_training/youtube/*.wav")):
    vid = os.path.basename(path).replace(".wav", "")
    d, sr = sf.read(path)
    if d.ndim == 2:
        d = d.mean(1)
    d = d.astype(np.float32)
    segments, _ = model.transcribe(path, word_timestamps=True, language="en",
                                   vad_filter=True)
    saved = 0
    for seg in segments:
        for w in (seg.words or []):
            if PAT.search(w.word):
                c = (w.start + w.end) / 2.0
                s0 = max(0, int((c - 0.55) * SR))
                s1 = min(len(d), int((c + 0.45) * SR))
                if s1 - s0 < 4000:
                    continue
                sf.write(f"{OUT}/{vid}_{saved:02d}.wav", d[s0:s1], SR)
                saved += 1
    total += saved
    print(f"  {vid}: {saved} reachy clips", flush=True)
print(f"TOTAL youtube reachy clips (whisper): {total}", flush=True)
