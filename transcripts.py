"""Per-chat transcript file logging.

A `TranscriptStore` owns one root directory; each chat lands in its own
sub-directory with zero-padded sequential `NNN.txt` files. `start()` rotates
to a fresh file (used by /clear and the first turn of any chat).
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)


class TranscriptStore:
    def __init__(
        self,
        root: Path,
        *,
        now: Callable[[], datetime.datetime] = datetime.datetime.now,
    ) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._paths: dict[int, Path] = {}
        self._now = now

    def start(self, chat_id: int) -> Path:
        chat_dir = self.root / str(chat_id)
        chat_dir.mkdir(parents=True, exist_ok=True)
        existing = [int(p.stem) for p in chat_dir.glob("*.txt") if p.stem.isdigit()]
        path = chat_dir / f"{max(existing, default=0) + 1:03d}.txt"
        path.touch()
        self._paths[chat_id] = path
        log.info("New transcript for chat %s: %s", chat_id, path)
        return path

    def append(self, chat_id: int, role: str, content: str) -> None:
        path = self._paths.get(chat_id) or self.start(chat_id)
        ts = self._now().strftime("%Y-%m-%d %H:%M:%S")
        with path.open("a", encoding="utf-8") as f:
            f.write(f"[{ts}] {role}: {content}\n\n")
