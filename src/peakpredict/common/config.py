"""Configuration and secret loading.

Secrets are read from the gitignored ``.secrets`` file at the repo root or from
the process environment (environment wins). Secret VALUES are never logged or
included in exceptions — only key names and the file name are surfaced.
"""

from __future__ import annotations

import os
from pathlib import Path

# src/peakpredict/common/config.py -> parents[3] == repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
SECRETS_FILE = REPO_ROOT / ".secrets"

# A value wrapped in double underscores (e.g. "__set_me__") is a template
# placeholder and is treated as "not set".
_PLACEHOLDER_PREFIX = "__"
_PLACEHOLDER_SUFFIX = "__"


class MissingSecretError(RuntimeError):
    """Raised when a required secret is absent or still a placeholder."""


def _parse_dotfile(path: Path) -> dict[str, str]:
    """Parse a simple ``KEY=VALUE`` dotfile. Blank lines and ``#`` comments ignored."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip()
    return out


def _is_placeholder(value: str) -> bool:
    return value.startswith(_PLACEHOLDER_PREFIX) and value.endswith(_PLACEHOLDER_SUFFIX)


def load_secrets(path: Path | None = None) -> dict[str, str]:
    """Return all secrets, merging the dotfile with the environment (env wins)."""
    secrets = _parse_dotfile(path or SECRETS_FILE)
    for key in list(secrets):
        if key in os.environ:
            secrets[key] = os.environ[key]
    return secrets


def get_secret(key: str, *, required: bool = True, path: Path | None = None) -> str | None:
    """Return a single secret, or ``None`` if absent and not required.

    Raises ``MissingSecretError`` (without echoing any value) if a required
    secret is missing or is still a template placeholder.
    """
    value = os.environ.get(key) or _parse_dotfile(path or SECRETS_FILE).get(key)
    if value and _is_placeholder(value):
        value = None
    if required and not value:
        raise MissingSecretError(
            f"Secret '{key}' is not set. Add it to the gitignored "
            f"'{SECRETS_FILE.name}' file or the environment."
        )
    return value
