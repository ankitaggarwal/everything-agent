# Release roadmap

How to grow and ship **everything-agent** in small, releasable steps. Each phase
is a working robot you can demo, tag, and open-source — not a half-thing. The
architecture (wake word → router → instant/agent → tools, see
`docs/architecture.html`) already supports every phase; you're just turning
backends from `mock` to real, one at a time, in `config.yaml`.

> Versioning suggestion: tag each phase (`v0.1`, `v0.2`, …). Keep `main` always
> runnable in mock mode so anyone can try it with zero setup.

### Current status (2026-06)
Live on a real Reachy Mini Wireless. Implemented adapters: `robot/reachy_mini`,
`hearing/stt/cartesia` (Ink-Whisper), `expressing/tts/cartesia` (Sonic to the
robot speaker), `brain/router/anthropic` (Haiku), `brain/agent/haiku`
(conversational stopgap), `hearing/wakeword/openwakeword`. Deployed via `deploy/`
(systemd, auto-start on boot). Remaining for the full vision: a custom "Hey
Reachy" wake-word model (see `docs/wakeword.md`), and the `claude_sdk` tool brain
(installed; needs `claude login` on the robot) to replace the stopgap.

---

## Phase 0 — Skeleton (done ✅)
The whole pipeline runs in **mock mode**: type to the robot, it routes and replies,
two example modules (`idle`, `system_time`) work. No hardware, no API keys.
```bash
python -m everything_agent
```
**Ship:** `v0.1` — "the shape of the thing." README + architecture doc.

## Phase 1 — Wake word + a real general LLM 🎙️
Make it *talk*. Turn on the real router and real voice; add real hearing.
- `brain.router.backend: anthropic` (Haiku) — needs `ANTHROPIC_API_KEY`.
- `hearing.stt.backend: whisper` — add `hearing/stt/whisper.py` + a map line.
- `hearing.wakeword.backend: openwakeword` — add `hearing/wakeword/openwakeword.py`.
- `express.tts.backend: cartesia` — set `providers.cartesia.enabled: true` + key.

Now: say "Hey Reachy", ask anything, hear a spoken answer. Still no tools.
**Ship:** `v0.2` — "you can talk to it."

## Phase 2 — First tools (local actions) ⏱️
Give it hands, starting tiny. `system_time` already ships as the template.
- Install the Agent SDK so escalation does real reasoning:
  `pip install -e ".[agent]"` (+ Python 3.10+, Node, `ANTHROPIC_API_KEY`).
- Add a few more local modules by copying `modules/system_time/`
  (e.g. `timer`, `notes`, `look_around`).
**Ship:** `v0.3` — "it can do simple things."

## Phase 3 — External services via MCP 🔌
The big unlock, and where your plan points: connect tools you don't write.
- **Telegram**: add a Telegram MCP server under `brain.agent.mcp_servers` (and
  set `brain.agent.backend: claude_sdk`). Sending a message is **sensitive** → it
  routes through the approval gate.
- **Clio (your other agent)**: if Clio exposes an MCP endpoint, list it the same
  way — Reachy can now delegate to Clio.
- Same pattern later for Swiggy, smart home, research.
```yaml
brain:
  agent:
    backend: claude_sdk
    mcp_servers:
      telegram: { command: npx, args: ["-y", "<telegram-mcp>"] }
      clio:     { command: python, args: ["-m", "clio.mcp"] }
```
**Ship:** `v0.4` — "it can act in the world (with your OK)."

## Phase 4 — Memory & personalization 🧠
Make it *yours*. The `simple` memory adapter already persists facts; start using it.
- Teach it facts ("remember my usual order is X").
- Add a `Mem0Memory` adapter (`memory/mem0.py` + a map line) for auto-extraction
  and semantic recall; set `memory.backend: mem0` (self-hosted, async writes off
  the hot path). The Memory port doesn't change.
**Ship:** `v0.5` — "it knows you."

## Phase 5 — Real robot body 🤖
Swap the mock for hardware. Add `robot/reachy_mini.py` (a `ReachyMiniRobot`
wrapping the `reachy_mini` SDK) + a line in `robot/__init__.py`, set
`robot.backend: reachy_mini`. Tune expression (head + antennas while listening/talking).
**Ship:** `v0.6` — "it's alive on my desk."

## Phase 6 — Polish for open source 📦
- Approval/safety hardening; document exactly what data leaves the device.
- Barge-in (interrupt while speaking).
- `pip install everything-agent`, sample configs, a 30-second demo video.
- CONTRIBUTING for module + MCP authors (already started).
**Ship:** `v1.0` — public release.

---

### The one rule that keeps this easy
Every phase = **flip a backend in `config.yaml`** or **add a module/MCP server**.
You should almost never edit `agent.py` or the core. If you find yourself
changing the loop to add a feature, that's a signal the feature wants to be a
module instead.
