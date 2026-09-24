"""Logging setup: console at INFO (the live trace), ``run.log`` at DEBUG, and a per-run context prefix."""

from __future__ import annotations

import logging
import time
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

PACKAGE_LOGGER = "pso_rf"
_HANDLER_MARK = "_pso_rf_handler"


class _ConsoleFormatter(logging.Formatter):
    """Plain ``HH:MM:SS message`` lines for INFO/DEBUG; a ``LEVEL:`` prefix for warnings and errors."""

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(message)s", datefmt="%H:%M:%S")
        self._warning = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno >= logging.WARNING:
            return self._warning.format(record)
        return super().format(record)


def close_logging() -> None:
    """Remove and close every handler installed by :func:`setup_logging` (releases ``run.log`` on Windows)."""
    logger = logging.getLogger(PACKAGE_LOGGER)
    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_MARK, False):
            logger.removeHandler(handler)
            handler.close()


def setup_logging(
    run_log_path: Path | str | None = None, console_level: int = logging.INFO
) -> logging.Logger:
    """Configure the ``pso_rf`` logger: console at ``console_level``, plus ``run_log_path`` at DEBUG if given.

    Calling it again replaces the handlers from the previous call. File timestamps are UTC.
    """
    close_logging()
    logger = logging.getLogger(PACKAGE_LOGGER)
    logger.setLevel(logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(_ConsoleFormatter())
    setattr(console, _HANDLER_MARK, True)
    logger.addHandler(console)

    if run_log_path is not None:
        path = Path(run_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s.%(msecs)03dZ %(levelname)-7s %(name)s: %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
        )
        formatter.converter = time.gmtime
        file_handler.setFormatter(formatter)
        setattr(file_handler, _HANDLER_MARK, True)
        logger.addHandler(file_handler)
    return logger


def run_prefix(dataset: str, fold: int | None, method: str, seed: int) -> str:
    """Return the run context ``[dataset|fold k|method|seed s]``; the fold reads ``deployment`` when None."""
    fold_label = "deployment" if fold is None else f"fold {fold}"
    return f"[{dataset}|{fold_label}|{method}|seed {seed}]"


class RunLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that prefixes every message with the run context (CONTEXT §15)."""

    def __init__(
        self, logger: logging.Logger, dataset: str, fold: int | None, method: str, seed: int
    ) -> None:
        super().__init__(logger, {"dataset": dataset, "fold": fold, "method": method, "seed": seed})
        self.prefix = run_prefix(dataset, fold, method, seed)

    def process(self, msg: Any, kwargs: MutableMapping[str, Any]) -> tuple[Any, MutableMapping[str, Any]]:
        """Prefix the message with the run context and attach the context as ``extra``."""
        kwargs.setdefault("extra", self.extra)
        return f"{self.prefix} {msg}", kwargs
