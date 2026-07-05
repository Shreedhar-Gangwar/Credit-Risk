"""Central configuration access for the credit-risk pipeline.

Usage
-----
    from config import load_config, ROOT, resolve_path

    cfg = load_config()
    seed = cfg["seed"]
    train_path = resolve_path(cfg["data"]["raw_train"])  # absolute Path

All paths in config.yaml are stored relative to the repository root; use
``resolve_path`` to turn them into absolute paths that work from any cwd.
This is the single source of truth for parameters (CLAUDE.md §7).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Repository root = parent of this ``config/`` package directory.
ROOT: Path = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG: Path = ROOT / "config" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and parse the YAML configuration file.

    Parameters
    ----------
    path : optional
        Override the config location; defaults to ``config/config.yaml``.
    """
    cfg_path = Path(path) if path is not None else _DEFAULT_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_path(relative: str | Path) -> Path:
    """Resolve a config-relative path against the repository root.

    Absolute inputs are returned unchanged.
    """
    p = Path(relative)
    return p if p.is_absolute() else (ROOT / p)


__all__ = ["load_config", "resolve_path", "ROOT"]
