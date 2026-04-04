"""Load YAML configuration and merge with environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def load_config(config_path: str | Path = "config/settings.yaml") -> dict[str, Any]:
    """Load YAML config and inject relevant env vars."""
    load_dotenv()

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Inject OpenAI key from environment
    cfg.setdefault("agent", {})
    cfg["agent"]["openai_api_key"] = os.environ.get("OPENAI_API_KEY", "")

    # Keep output directories as relative paths (relative to project root)
    # They are used as-is; scripts are expected to run from the project root.

    return cfg
