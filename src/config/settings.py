from __future__ import annotations

import os
from pathlib import Path


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


ROOT_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = Path(os.getenv("PROJECT_ENV_FILE", ROOT_DIR / ".env"))
_load_env_file(ENV_FILE)


def _path_from_env(var_name: str, default_relative: str) -> Path:
    raw = os.getenv(var_name)
    path = Path(raw) if raw else (ROOT_DIR / default_relative)
    if not path.is_absolute():
        path = (ROOT_DIR / path).resolve()
    return path


DATASET_PATH = _path_from_env("DATASET_PATH", "data/financial_data.csv")
PROCESSED_DIR = _path_from_env("PROCESSED_DIR", "data/processed")
MODELS_DIR = _path_from_env("MODELS_DIR", "data/models")
REPORTS_DIR = _path_from_env("REPORTS_DIR", "data/reports")

ML_SAMPLE_SIZE = int(os.getenv("ML_SAMPLE_SIZE", "200000"))
PREDICTION_THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))
