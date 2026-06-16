"""C6 — credential gate.

If ``dashboard_password`` is set in Streamlit secrets, the app requires it before
rendering. With no password configured (local/dev) the gate is open.
"""

from __future__ import annotations

import streamlit as st


def _expected_password() -> str | None:
    try:
        return st.secrets.get("dashboard_password")
    except Exception:
        return None


def require_auth() -> None:
    """Block the app until the configured password is entered (no-op if unset)."""
    expected = _expected_password()
    if not expected or st.session_state.get("pp_authed"):
        return
    st.title("Sign in")
    pw = st.text_input("Password", type="password")
    if pw == expected:
        st.session_state["pp_authed"] = True
        st.rerun()
    elif pw:
        st.error("Incorrect password")
    st.stop()
