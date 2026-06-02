# everything-agent

A **personal desk robot you talk to**, built on the
[Reachy Mini](https://www.pollen-robotics.com/). Say the wake word, ask for
anything — it answers instantly, or gets it done with tools — and it reacts with
its head and antennas, tracking you as you speak. You grow it by adding
**modules** and connecting **MCP servers**, one small step at a time.

It runs end-to-end on a real Reachy Mini Wireless: mic → wake word → speech-to-text
→ a fast router → a tool-using brain → expressive voice and movement.

> 📊 **See [`docs/architecture.html`](docs/architecture.html) for the interactive
> architecture diagram** (open in any browser). 🗺️ **See [ROADMAP.md](ROADMAP.md)
> for the release plan.**

## How it works

A request flows through four stages (+ memory alongside):

```
HEAR      wake word → speech-to-text
DECIDE    router LLM (fast) → answer instantly, OR escalate
ACT       agent brain → tools (local modules + MCP servers) → approval gate
EXPRESS   speak the reply + express emotion with head & antennas
```

The **router** is the trick that keeps it snappy: a small fast model handles
trivial things itself and only wakes the heavier agent when there's real work.
The agent brain has two flavours — `fast` (direct Anthropic API tool calls, low
latency) and `claude_sdk` (the full Claude Agent SDK with MCP servers). Risky
actions (spending money, sending messages) pass an **approval gate**.

## Quick start (no hardware, no API keys)

Everything defaults to **mock mode**, so the whole pipeline runs on a laptop and
you just *type* to the robot.

```bash
pip install pyyaml            # optional; without it, built-in defaults are used
python -m everything_agent    # type to it; say "quit" to stop
```

Try: `hello`, `what is the time`, `what is the date`, `look left`.

## Going live (flip backends in config)

Turn `mock` → real, one piece at a time — see [ROADMAP.md](ROADMAP.md). Install
only the extras you need, then point at a profile with `EVERYTHING_AGENT_CONFIG`
(the repo ships [`config.reachy.yaml`](config.reachy.yaml), the real-robot profile):

```bash
pip install -e ".[anthropic,cartesia]"
cp .env.example .env          # add your API keys
```

- **Real router:** `brain.router.backend: anthropic` (needs `ANTHROPIC_API_KEY`).
- **Fast tool brain:** `brain.agent.backend: fast` — direct API tool calls (Haiku), low latency.
- **Full agent brain:** `brain.agent.backend: claude_sdk` — adds MCP servers; needs Node + the Claude Code CLI.
- **Voice + hearing:** Cartesia `sonic` TTS and `ink-whisper` STT (one `CARTESIA_API_KEY`).
- **Wake word:** `hearing.wakeword.backend: openwakeword` (bundled `alexa`; train a custom "Hey Reachy" — see [docs/wakeword.md](docs/wakeword.md)).
- **Real robot:** `robot.backend: reachy_mini` (see `everything_agent/robot/reachy_mini.py`).
- **Personality & emotions:** edit [`everything_agent/persona.py`](everything_agent/persona.py); the brain expresses feelings through the `emotions` module.

### Run it on a real Reachy Mini
The agent runs on the robot itself (it speaks, listens, and moves). See
**[deploy/README.md](deploy/README.md)** for the rsync + systemd setup.

## Architecture: ports & adapters

Every subsystem is a **port** (an interface in `core/ports.py`) with one or more
**adapters** (implementations) you pick by config. Choosing Cartesia vs Parakeet,
or Mem0 vs no memory, is always the same move: set `backend:`. Adapters are
**lazily imported**, so selecting `mock` never pulls in `torch` or a cloud SDK —
you install only the dependencies for the backends you turn on.

```
everything_agent/
├── __main__.py        # entry point + config loading
├── agent.py           # the loop: HEAR → DECIDE → ACT → EXPRESS (no concrete imports)
├── persona.py         # the robot's personality (one editable source of character)
├── core/
│   ├── ports.py       #   the interfaces: WakeWord, STT, Router, AgentBrain, TTS, Robot, Memory
│   ├── plugins.py     #   load() — config name → adapter (lazy import)
│   ├── context.py     #   AgentContext shared with every adapter
│   ├── module.py      #   the Module/Action contract ← extend to add capabilities
│   ├── registry.py    #   loads modules, aggregates their actions
│   └── approval.py    #   asks before risky actions
├── hearing/
│   ├── wakeword/      #   mock | openwakeword           ← each folder = one port,
│   └── stt/           #   mock | cartesia                  one file per adapter,
├── brain/                                              #   __init__.py = BACKENDS map
│   ├── router/        #   mock | anthropic  (+ gemini)
│   └── agent/         #   mock | haiku | fast | claude_sdk
├── expressing/
│   └── tts/           #   mock | cartesia  (+ elevenlabs)
├── robot/             #   mock | reachy_mini
├── memory/            #   none | simple  (+ mem0)
├── modules/           # CAPABILITIES (copy these to add more)
│   ├── idle/          #   aliveness: speaker tracking (DoA) + organic idle + look_at
│   ├── emotions/      #   express_happy / curious / nod_yes / celebrate … (head + antennas)
│   └── system_time/   #   get_time / get_date  ← template for a new tool
└── providers/         # shared clients (one Cartesia client, reusable across ports)

# alongside the package (deployment + config, kept out of the importable library):
config.yaml            # default control panel (all mock — runs anywhere)
config.reachy.yaml     # the real-robot profile (voice + movement + Haiku)
scripts/               # run_on_robot.sh, selftest_robot.py
deploy/                # systemd service + install_service.sh (see deploy/README.md)
docs/                  # architecture.html, wakeword.md
```

## Two ways to extend

**Add a capability (a module):**
1. `cp -r everything_agent/modules/system_time everything_agent/modules/<name>`
2. Write your `actions()` (plain async functions — no SDK import needed).
3. Register the class in `core/registry.py` and add `<name>` to `config.yaml`.

**Add a backend (an adapter), e.g. ElevenLabs voice or Mem0 memory:**
1. Write one file implementing the port (e.g. `expressing/tts/elevenlabs.py`).
2. Add one line to that port's `BACKENDS` map in its `__init__.py`.
3. Select it in `config.yaml`.

For external services with their own MCP server (Telegram, Swiggy, Clio), you
don't even write a module — just add it under `brain.agent.mcp_servers`. In all
cases the loop and brains never change. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
