from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_markdown(path: str | Path, max_chars: int = 80_000) -> str:
    text = Path(path).read_text(encoding="utf-8")
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n[truncated]"
    return text


def read_parquet_preview(path: str | Path, limit: int = 200) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    return frame.head(limit)


def parquet_schema(path: str | Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    return pd.DataFrame({"column": list(frame.columns), "dtype": [str(dtype) for dtype in frame.dtypes]})


def resolve_existing_path(path: str | Path, root: str | Path = ".") -> Path:
    raw = Path(path)
    if raw.exists():
        return raw
    candidate = Path(root) / raw
    if candidate.exists():
        return candidate
    raise FileNotFoundError(path)
