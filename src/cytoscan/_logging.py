"""
centralized logging setup for cytoscan.

conventions:
- log = logging.getLogger(__name__)  in every module that emits output.
- logs go to stderr, colored, prefixed with [cytoscan].
- data goes to stdout (CSV paths, JSON), so piping works:
    cytoscan run exp/ > results.csv 2> run.log
- per-frame progress uses tqdm (cytoscan._logging.progress), not prints.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterable, Optional

from tqdm import tqdm

_ROOT_NAME = "cytoscan"

class _ColorFormatter(logging.Formatter):
    """color-codes the [cytoscan] prefix by level. ANSI only — no deps."""

    _COLORS = {
        logging.DEBUG:    "\033[2m",       # dim
        logging.INFO:     "\033[36m",      # cyan
        logging.WARNING:  "\033[33m",      # yellow
        logging.ERROR:    "\033[31m",      # red
        logging.CRITICAL: "\033[1;31m",    # bold red
    }
    _RESET = "\033[0m"

    def __init__(self, use_color: bool = True):
        super().__init__()
        self.use_color = use_color and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        if record.exc_info:
            msg = f"{msg}\n{self.formatException(record.exc_info)}"
        if self.use_color:
            color = self._COLORS.get(record.levelno, "")
            return f"{color}[cytoscan]{self._RESET} {msg}"
        return f"[cytoscan] {msg}"


"""
configure the cytoscan logger. Call once from CLI / GUI entrypoint.

verbose: enables DEBUG level (algorithm internals, per-frame diagnostics).
quiet:   raises threshold to WARNING (suppresses INFO progress messages).
log_file: also write all DEBUG-and-up records to this file, uncolored.
"""
def setup(verbose: bool = False, quiet: bool = False, log_file: Optional[Path] = None) -> None:
    if verbose and quiet:
        raise ValueError("--verbose and --quiet are mutually exclusive")

    level = logging.DEBUG if verbose else (logging.WARNING if quiet else logging.INFO)

    root = logging.getLogger(_ROOT_NAME)
    root.setLevel(logging.DEBUG)   # let handlers do the filtering
    root.handlers.clear()
    root.propagate = False         # don't double-print via the global root logger

    # stderr handler: colored, level-gated by --verbose/--quiet
    stream = logging.StreamHandler(sys.stderr)
    stream.setLevel(level)
    stream.setFormatter(_ColorFormatter())
    root.addHandler(stream)

    # file handler: always DEBUG, full timestamps, no color
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        root.addHandler(fh)


"""
tqdm wrapper — auto-disables when stderr isn't a TTY (clean batch logs).

Usage:
    for fi, fd in progress(detections.items(), "running detections"):
        ...
"""
def progress(iterable: Iterable, desc: str, **kwargs):
    return tqdm(
        iterable,
        desc=f"[cytoscan] {desc}",
        unit="frame",
        leave=False,                       # don't litter the terminal post-loop
        disable=not sys.stderr.isatty(),   # silent in batch / CI

        ncols=95,
        **kwargs,
    )
