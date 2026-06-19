# Recording "Reachy" for the custom wake word

Goal: enough varied samples of **each household voice** saying **"Reachy"** so the
trained model reliably wakes for *you* (not just generic en-US TTS voices).

## What to say
- Mostly the **single word "Reachy"** said the way you'd naturally address the robot.
- Mix in some **in-context** ones too: *"Reachy, what time is it?"*, *"Reachy, stop."*,
  *"Hey Reachy."* — the model needs to spot "Reachy" both alone and at the start of a sentence.

## How many
- **~50-100 per person**, more is better. Record **each person** who'll use it
  (you, your wife, anyone else) — name the files by person.

## Vary it (this is what makes it robust)
Across your samples, deliberately change:
- **Distance**: right up close, arm's length, across the room.
- **Tone/speed**: normal, quiet, excited, questioning, slow, fast.
- **Rooms / background**: quiet room, with TV/music, in the kitchen, etc.

## Format & delivery
- Phone voice memo is fine (**.m4a / .wav / .mp3** — I convert).
- **Leave ~1 second of silence between each "Reachy"** so I can auto-split cleanly.
  (Don't rattle them off back-to-back — that's what made the first two hard to split.)
- One long file per session is fine; just pause between each.
- Tell me **whose voice** each file is.

## Optional but valuable — "negatives"
A few minutes of **normal talk** + similar-sounding words (*reach, beachy, peachy,
richie, ritchie, region, really*) so the model learns what is **not** the wake word
(cuts false triggers). Name these clearly as negatives.

## Where to put them
Drop the files in `~/Downloads` (or anywhere) and paste the paths to me — I'll
convert, split, label, and feed them into the HF Jobs training run.
