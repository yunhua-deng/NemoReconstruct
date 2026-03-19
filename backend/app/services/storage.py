from __future__ import annotations

import shutil
from pathlib import Path


def ensure_workspace(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def remove_workspace(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
