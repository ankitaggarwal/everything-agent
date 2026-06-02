# Contributing

everything-agent grows by **modules** and **MCP servers**. The core loop and the
brains rarely change — almost all new capability is a new module (local) or a new
MCP server (external). This guide shows both.

## Mental model

```
HEAR → DECIDE (router) → instant reply  OR  agent brain → tools/MCP → approval
                                                             ↓
                                            EXPRESS (speak) + MEMORY
```

Everything you add hangs off the **agent brain's tools** or one of the module
hooks. See `docs/architecture.html` for the picture.

## Add a local module

A module is one capability. It can contribute three independent things — implement
only what applies:

| Hook | Purpose | Example |
|------|---------|---------|
| `actions()` | capabilities the agent brain can call | `get_time()`, `look_at(dir)` |
| `perceive()` | report what you sense (fed to the router) | "heard a doorbell" |
| `tick()` | autonomous idle behavior | gentle look-around |

**Actions are plain async Python — no Claude SDK import.** The brain wraps them
into Agent-SDK tools automatically, and adds the approval gate for `sensitive`
ones. An `Action` is just:

```python
Action(
    name="send_telegram",
    description="Send a Telegram message to the user",
    handler=send,                  # async (args: dict) -> str
    params={"text": str},          # {} if no args
    sensitive=True,                # → approval gate asks first
)
```

**Steps**

1. `cp -r everything_agent/modules/system_time everything_agent/modules/<name>`
2. Rename the class, write your `actions()`.
3. Register it in `everything_agent/core/registry.py` → `_module_classes()`:
   ```python
   from ..modules.<name>.<name> import <Name>Module
   return { ..., "<name>": <Name>Module }
   ```
4. Add `"<name>"` to `modules:` in `config.yaml`.

No loop or brain changes.

## Add an external service (MCP server) — often no code at all

If a service already has an MCP server (Telegram, Swiggy, your other agents…),
just declare it in `config.yaml`:

```yaml
agent:
  mcp_servers:
    telegram: { command: npx, args: ["-y", "<telegram-mcp-server>"] }
```

The Agent SDK connects to it and its tools become available to the brain. Mark
flows that spend money or message people as sensitive at the call site / in the
service's own permissions, and lean on the approval gate.

## Add a backend (adapter) — e.g. a new voice, STT, or memory

Every subsystem is a **port** (interface in `core/ports.py`) with swappable
**adapters**. To add one (say ElevenLabs TTS):

1. Write `everything_agent/expressing/tts/elevenlabs.py` with a class that
   subclasses the port and has the standard constructor:
   ```python
   from ...core.ports import TTS
   class ElevenLabsTTS(TTS):
       def __init__(self, config, ctx):   # config = this port's config block
           self.robot = ctx.robot          # ctx = shared AgentContext
       def speak(self, text): ...
   ```
2. Add one line to the `BACKENDS` map in `expressing/tts/__init__.py`:
   ```python
   "elevenlabs": "everything_agent.expressing.tts.elevenlabs:ElevenLabsTTS",
   ```
3. Select it in `config.yaml`: `express: { tts: { backend: elevenlabs } }`.

Rules for adapters:
- **Lazy imports.** Import heavy/optional deps (torch, an SDK) *inside* the
  adapter, not at module top level — so choosing another backend never imports
  yours. The registry only imports an adapter when it's selected.
- **Same constructor everywhere:** `__init__(self, config, ctx)`. Pull what you
  need from `ctx` (robot, providers, memory, approval, actions).
- **Opting out is an adapter too** — see `memory/none.py`.

## Add a provider

A provider is a shared client for an external service that one or more *adapters*
use (e.g. one Cartesia client used by both a TTS and an STT adapter). Copy
`everything_agent/providers/cartesia.py` and construct it in `build_providers()`.
Adapters read it from `ctx.providers["<name>"]`.

## Principles

- **One module = one clear purpose.** If it does two things, it's two modules.
- **Talk to interfaces, not libraries.** Use the `Robot` interface and providers;
  never import `reachy_mini` (or the Claude SDK) inside a module.
- **Fail soft.** A module that throws in `perceive()`/`tick()` is logged and
  skipped — it must not crash the loop.
- **Sensitive by default for irreversible actions.** Spending money, sending
  messages, deleting → `sensitive=True`.
