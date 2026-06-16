"""Structured logging shared across components.

Provides a single ``get_logger`` so all components log with a consistent format.
Never pass secret values to the logger.
"""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
_configured = False


def _configure_root() -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT))
    root = logging.getLogger("peakpredict")
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the ``peakpredict`` root."""
    _configure_root()
    return logging.getLogger(f"peakpredict.{name}")
