"""#68545: configurable journal_mode (env + config.yaml) + centralized DB openers."""

from __future__ import annotations

import sqlite3

import pytest


def test_resolve_journal_mode_defaults_to_wal(monkeypatch):
    from hermes_state import resolve_journal_mode

    monkeypatch.delenv("HERMES_JOURNAL_MODE", raising=False)
    assert resolve_journal_mode() == "wal"


def test_resolve_journal_mode_env_override(monkeypatch):
    from hermes_state import resolve_journal_mode

    monkeypatch.setenv("HERMES_JOURNAL_MODE", "delete")
    assert resolve_journal_mode() == "delete"


def test_resolve_journal_mode_env_truncase(monkeypatch):
    from hermes_state import resolve_journal_mode

    monkeypatch.setenv("HERMES_JOURNAL_MODE", "DELETE")
    assert resolve_journal_mode() == "delete"


def test_resolve_journal_mode_config_override(monkeypatch, tmp_path):
    from hermes_state import resolve_journal_mode

    monkeypatch.delenv("HERMES_JOURNAL_MODE", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "database:\n  journal_mode: delete\n",
        encoding="utf-8",
    )
    assert resolve_journal_mode() == "delete"


def test_resolve_journal_mode_invalid_falls_back_to_wal(monkeypatch):
    from hermes_state import resolve_journal_mode

    monkeypatch.setenv("HERMES_JOURNAL_MODE", "bogus")
    assert resolve_journal_mode() == "wal"


def test_apply_wal_with_fallback_honors_delete_mode(monkeypatch, tmp_path):
    """When HERMES_JOURNAL_MODE=delete, apply_wal_with_fallback must NOT set WAL."""
    from hermes_state import apply_wal_with_fallback

    monkeypatch.setenv("HERMES_JOURNAL_MODE", "delete")
    db = tmp_path / "test.db"
    conn = sqlite3.connect(str(db))
    mode = apply_wal_with_fallback(conn, db_label="test.db")
    assert mode == "delete"
    actual = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert actual.lower() == "delete"
    conn.close()


def test_apply_wal_with_fallback_switches_existing_wal_to_delete(monkeypatch, tmp_path):
    """A forced mode is authoritative even when the DB header already says WAL."""
    from hermes_state import apply_wal_with_fallback

    db = tmp_path / "existing-wal.db"
    conn = sqlite3.connect(str(db))
    assert conn.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    conn.close()

    monkeypatch.setenv("HERMES_JOURNAL_MODE", "delete")
    conn = sqlite3.connect(str(db))
    assert apply_wal_with_fallback(conn, db_label="existing-wal.db") == "delete"
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    conn.close()


def test_apply_wal_with_fallback_does_not_hide_forced_mode_failure(monkeypatch):
    """Never claim DELETE is active when SQLite rejected the transition."""
    from hermes_state import apply_wal_with_fallback

    class RejectingConnection:
        def execute(self, _sql):
            raise sqlite3.OperationalError("database is locked")

    monkeypatch.setenv("HERMES_JOURNAL_MODE", "delete")
    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        apply_wal_with_fallback(RejectingConnection(), db_label="locked.db")


def test_apply_wal_with_fallback_defaults_to_wal(monkeypatch, tmp_path):
    """Without override, apply_wal_with_fallback still sets WAL."""
    from hermes_state import apply_wal_with_fallback

    monkeypatch.delenv("HERMES_JOURNAL_MODE", raising=False)
    db = tmp_path / "test2.db"
    conn = sqlite3.connect(str(db))
    mode = apply_wal_with_fallback(conn, db_label="test2.db")
    assert mode == "wal"
    conn.close()


def test_direct_db_openers_honor_forced_delete(monkeypatch, tmp_path):
    """Exercise every former direct-WAL opener against real SQLite files."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_JOURNAL_MODE", "delete")

    from agent import verification_evidence
    from cron import executions
    from gateway import delivery_ledger
    from plugins.platforms.discord.recovery import DiscordRecoveryStore
    from tools import async_delegation

    connections = [
        async_delegation._connect(),
        delivery_ledger._connect(),
        verification_evidence._connect(),
    ]
    try:
        assert [
            conn.execute("PRAGMA journal_mode").fetchone()[0] for conn in connections
        ] == ["delete", "delete", "delete"]
    finally:
        for conn in connections:
            conn.close()

    monkeypatch.setattr(
        executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db"
    )
    conn = executions._connect()
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    finally:
        conn.close()

    store = DiscordRecoveryStore(tmp_path)
    assert (
        store.call(lambda conn: conn.execute("PRAGMA journal_mode").fetchone()[0])
        == "delete"
    )
