from __future__ import annotations

from contextlib import closing, contextmanager
import fcntl
import json
import os
from pathlib import Path
import sqlite3
from uuid import uuid4


def cache_root() -> Path:
    configured = os.environ.get("INTENT_CACHE_DIR")
    if configured is not None:
        if not configured:
            raise ValueError("INTENT_CACHE_DIR must name a directory")
        return Path(configured).expanduser().resolve()
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return (base / "intentdsl").expanduser().resolve()


def _compilation_key(executable: Path, module_text: str,
                     options: tuple[str, ...]) -> str:
    status = executable.stat()
    profiles = executable.parent / "profiles"
    dependencies = [(str(path), path.read_text(encoding="utf-8"))
                    for path in sorted(profiles.glob("*.json"))]
    for option in options:
        if option.startswith("--tuning-config="):
            path = Path(option.split("=", 1)[1]).resolve()
            dependencies.append((str(path), path.read_text(encoding="utf-8")))
    # Compare the complete compile inputs. File identity invalidates entries
    # when the compiler is rebuilt or replaced, without checksumming artifacts.
    return json.dumps((str(executable),
                       (status.st_dev, status.st_ino, status.st_size,
                        status.st_mtime_ns, status.st_ctime_ns),
                       os.environ.get("LD_LIBRARY_PATH", ""),
                       options, dependencies, module_text), ensure_ascii=False)


@contextmanager
def compilation_directory(executable: Path, module_text: str,
                          options: tuple[str, ...]):
    root = cache_root() / "compiler"
    root.mkdir(parents=True, exist_ok=True)
    key = _compilation_key(executable, module_text, options)
    with closing(sqlite3.connect(root / "entries.sqlite3", timeout=60)) as database:
        with database:
            database.execute(
                "CREATE TABLE IF NOT EXISTS entries "
                "(directory TEXT PRIMARY KEY, compile_inputs TEXT NOT NULL UNIQUE)"
            )
            database.execute("INSERT OR IGNORE INTO entries (directory, compile_inputs) VALUES (?, ?)",
                             (uuid4().hex, key))
            identity, = database.execute(
                "SELECT directory FROM entries WHERE compile_inputs = ?", (key,)
            ).fetchone()
    directory = root / identity
    directory.mkdir(exist_ok=True)
    # Independent specializations compile concurrently; the same entry is
    # published only after all outputs have been read and validated.
    with (directory / ".lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield directory, key
