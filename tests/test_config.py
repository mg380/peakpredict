import pytest

from peakpredict.common.config import MissingSecretError, get_secret, load_secrets


def _write_secrets(tmp_path, text):
    p = tmp_path / ".secrets"
    p.write_text(text, encoding="utf-8")
    return p


def test_load_secrets_parses_and_ignores_comments(tmp_path):
    p = _write_secrets(tmp_path, "# comment\nPP_USER=alice\n\nPP_PASS=secret-value\n")
    secrets = load_secrets(p)
    assert secrets == {"PP_USER": "alice", "PP_PASS": "secret-value"}


def test_env_overrides_file(tmp_path, monkeypatch):
    p = _write_secrets(tmp_path, "PP_USER=from_file\n")
    monkeypatch.setenv("PP_USER", "from_env")
    assert load_secrets(p)["PP_USER"] == "from_env"


def test_get_secret_present(tmp_path):
    p = _write_secrets(tmp_path, "TILASTOPAJA_PASS=abc123\n")
    assert get_secret("TILASTOPAJA_PASS", path=p) == "abc123"


def test_get_secret_placeholder_treated_missing(tmp_path):
    p = _write_secrets(tmp_path, "TILASTOPAJA_PASS=__set_me__\n")
    with pytest.raises(MissingSecretError):
        get_secret("TILASTOPAJA_PASS", path=p)


def test_missing_required_secret_does_not_leak_value(tmp_path):
    p = _write_secrets(tmp_path, "OTHER=x\n")
    with pytest.raises(MissingSecretError) as exc:
        get_secret("TILASTOPAJA_PASS", path=p)
    # error names the key + file but never a value
    assert "TILASTOPAJA_PASS" in str(exc.value)
    assert "x" not in str(exc.value).replace("TILASTOPAJA_PASS", "")


def test_optional_secret_returns_none(tmp_path):
    p = _write_secrets(tmp_path, "")
    assert get_secret("NOPE", required=False, path=p) is None
