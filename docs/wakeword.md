# Wake word: "Hey Reachy"

The wake word is a **port** (`hearing/wakeword/`) with two adapters:

| backend | what it does | needs |
|---|---|---|
| `mock` | always-listening — every utterance is treated as input | nothing |
| `openwakeword` | on-device detection of a wake phrase, then hands off to STT | `openwakeword`, `onnxruntime` |

Select it in `config.reachy.yaml` under `hearing.wakeword`.

## Running a built-in phrase (works today)

openWakeWord ships pretrained models (`hey_jarvis`, `alexa`, `hey_mycroft`, …).
To try the pipeline immediately:

```yaml
hearing:
  wakeword:
    backend: openwakeword
    model: hey_jarvis      # built-in
    threshold: 0.5
```

## Getting a real "Hey Reachy" model

There is no pretrained "Hey Reachy", so train a custom one (one-time, offline):

1. Use the openWakeWord **automatic training** notebook
   (<https://github.com/dscripka/openWakeWord>), which synthesizes "hey reachy"
   samples with TTS and trains a small model — no manual recording needed.
2. Export the `.onnx` model and copy it to `models/hey_reachy.onnx` on the robot.
3. Point the config at it and switch the backend:

```yaml
hearing:
  wakeword:
    backend: openwakeword
    model_path: models/hey_reachy.onnx
    threshold: 0.5     # raise to reduce false triggers, lower to catch more
```

Tune `threshold` against `journalctl -u everything-agent -f`: watch the logged
detection scores and pick a value just above the noise.
