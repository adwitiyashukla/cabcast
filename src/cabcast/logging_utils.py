from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_CONFIGURED = False


class _KeyValueFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        stamp = self.formatTime(record, "%H:%M:%S")
        base = f"{stamp} {record.levelname:<7} {record.name:<30} {record.getMessage()}"
        extras = getattr(record, "extra_fields", None)
        if extras:
            base = f"{base} | {json.dumps(extras, default=str, sort_keys=True)}"
        if record.exc_info:
            base = f"{base}\n{self.formatException(record.exc_info)}"
        return base


def setup_logging(level: int | str = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_KeyValueFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


def log_event(logger: logging.Logger, message: str, /, **fields: Any) -> None:
    logger.info(message, extra={"extra_fields": fields})


@contextmanager
def stage(logger: logging.Logger, name: str, **fields: Any) -> Iterator[dict[str, Any]]:
    payload: dict[str, Any] = dict(fields)
    log_event(logger, f"START {name}", **payload)
    started = time.perf_counter()
    try:
        yield payload
    except Exception as exc:
        elapsed = time.perf_counter() - started
        log_event(logger, f"FAIL  {name}", error=repr(exc), seconds=round(elapsed, 2), **payload)
        raise
    elapsed = time.perf_counter() - started
    log_event(logger, f"DONE  {name}", seconds=round(elapsed, 2), **payload)
