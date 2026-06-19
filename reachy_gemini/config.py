"""Load config.yaml, overlay the gitignored config.local.yaml, fall back to env.

The whole secret story lives here: the tracked config.yaml never holds a key;
config.local.yaml (gitignored) or $GEMINI_API_KEY does.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _deep_merge(base: dict, over: dict) -> dict:
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | os.PathLike | None = None) -> dict:
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) or {}

    local = cfg_path.parent / "config.local.yaml"
    if local.exists():
        _deep_merge(cfg, yaml.safe_load(local.read_text()) or {})

    gemini = cfg.setdefault("gemini", {})
    if not gemini.get("api_key"):
        gemini["api_key"] = os.environ.get("GEMINI_API_KEY", "")

    if not gemini["api_key"]:
        raise SystemExit(
            "No Gemini API key. Put it in config.local.yaml under gemini.api_key, "
            "or set the GEMINI_API_KEY environment variable.\n"
            "Get one at https://aistudio.google.com/apikey"
        )
    return cfg
