"""Entry point: `python -m everything_agent`.

Loads config.yaml (falls back to sane defaults if PyYAML isn't installed yet),
then runs the always-on loop. In mock mode you just type to the robot.
"""
from __future__ import annotations

import asyncio
import logging
import os

# Mirrors config.yaml. Every subsystem is a {backend: ...} block, so the shape
# is identical everywhere and the registry knows how to build each one.
_DEFAULTS = {
    "hearing": {
        "wakeword": {"backend": "mock", "phrase": "hey reachy"},
        "stt": {"backend": "mock"},
    },
    "brain": {
        "router": {"backend": "mock", "model": "claude-haiku-4-5"},
        "agent": {"backend": "mock", "model": "claude-opus-4-8", "mcp_servers": {}},
    },
    "express": {"tts": {"backend": "mock"}},
    "robot": {"backend": "mock"},
    "memory": {"backend": "simple", "path": "memory.json"},
    "approval": {"auto_approve": False},
    "modules": ["idle", "system_time"],
    "providers": {"cartesia": {"enabled": False}},
}


def load_config(path: str | None = None) -> dict:
    # Allow a deployment to point at a different config without touching the
    # committed default (e.g. EVERYTHING_AGENT_CONFIG=config.reachy.yaml on the robot).
    explicit = path or os.environ.get("EVERYTHING_AGENT_CONFIG")
    path = explicit or "config.yaml"
    if not os.path.exists(path):
        if explicit:
            # A config asked for by name that doesn't exist is a deploy bug;
            # silently booting with mock defaults would only hide it.
            raise FileNotFoundError(f"config not found: {path}")
        return _DEFAULTS
    try:
        import yaml
    except ImportError:
        logging.getLogger("everything_agent").warning(
            "PyYAML not installed -- ignoring %s and using built-in defaults", path)
        return _DEFAULTS
    with open(path) as f:                       # malformed YAML crashes loudly here
        return yaml.safe_load(f) or _DEFAULTS   # rather than silently going mock


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # Load a local .env (API keys) if python-dotenv is installed. Optional: in
    # mock mode there are no keys to load, so this stays a soft dependency.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    from .agent import EverythingAgent
    asyncio.run(EverythingAgent(load_config()).run())


if __name__ == "__main__":
    main()
