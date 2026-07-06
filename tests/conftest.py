"""Shared pytest fixtures. Run from the repo root: `python -m pytest`.

Fixtures load only a small sample of the raw data (fast) and, where needed, the saved
artifact (skipping those tests if it hasn't been trained yet).
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import load_config, resolve_path  # noqa: E402

SAMPLE_ROWS = 6000


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def raw_sample(cfg):
    """A small slice of the raw training table (keeps tests fast)."""
    path = resolve_path(cfg["data"]["raw_train"])
    if not path.exists():
        pytest.skip(f"raw data not present at {path}")
    return pd.read_csv(path, nrows=SAMPLE_ROWS)


@pytest.fixture(scope="session")
def bundle(cfg):
    """The saved model artifact; skip artifact-dependent tests if not trained yet."""
    path = resolve_path(cfg["artifacts"]["pipeline_file"])
    if not path.exists():
        pytest.skip("model artifact not found — run `python -m models.train` first")
    return joblib.load(path)
