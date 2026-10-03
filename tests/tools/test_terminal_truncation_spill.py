"""Tests for terminal truncation spill + metadata (deferred retrieval)."""

import json
import os
import time
from pathlib import Path

import pytest

from tools.terminal_tool import terminal_tool


@pytest.fixture
def small_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_constants import hermes_home_key
    import tools.tool_output_limits as lim
    monkeypatch.setattr(lim, "_cached_limits", {hermes_home_key(): {
        "max_bytes": 2000, "max_lines": 2000, "max_line_length": 2000,
    }})
    return tmp_path


class TestTruncationSpill:
    def test_truncated_output_has_metadata_and_spill(self, small_cap):
        r = json.loads(terminal_tool(
            "python3 -c \"print('marker_head'); [print(f'row_{i}', 'x'*80) for i in range(200)]; print('marker_tail')\"",
            task_id="t-spill-1"))
        assert r["exit_code"] == 0
        assert "OUTPUT TRUNCATED" in r["output"]
        assert r["output_total_chars"] > 2000
        p = Path(r["full_output_path"])
        assert p.exists()
        full = p.read_text()
        assert "marker_head" in full and "marker_tail" in full
        # The spill contains rows that were cut from the visible window.
        assert "row_100 " in full
        assert "read_file" in r["truncation_note"]

    def test_small_output_has_no_metadata(self, small_cap):
        r = json.loads(terminal_tool("echo tiny", task_id="t-spill-2"))
        assert r["exit_code"] == 0
        assert "full_output_path" not in r
        assert "output_total_chars" not in r

    def test_spill_is_redacted(self, small_cap):
        r = json.loads(terminal_tool(
            "python3 -c \"print('sk-proj-' + 'a1B2c3D4e5F6g7H8i9J0' * 3); [print('pad', 'y'*90) for i in range(200)]\"",
            task_id="t-spill-3"))
        p = Path(r["full_output_path"])
        full = p.read_text()
        assert "a1B2c3D4e5F6g7H8i9J0a1B2c3D4e5F6g7H8i9J0" not in full

    def test_old_spills_cleaned(self, small_cap, tmp_path):
        spill_dir = tmp_path / ".hermes" / "cache" / "terminal-output"
        spill_dir.mkdir(parents=True, exist_ok=True)
        stale = spill_dir / "out-1-2-dead.log"
        stale.write_text("old")
        os.utime(stale, (1, 1))
        json.loads(terminal_tool(
            "python3 -c \"[print('z'*90) for i in range(200)]\"", task_id="t-spill-4"))
        assert not stale.exists()

    def test_failed_command_still_gets_spill(self, small_cap):
        r = json.loads(terminal_tool(
            "python3 -c \"[print('e'*90) for i in range(200)]; import sys; sys.exit(3)\"",
            task_id="t-spill-5"))
        assert r["exit_code"] == 3
        assert Path(r["full_output_path"]).exists()


class _RecordingEnv:
    """Non-local backend stand-in: records what reaches its filesystem."""

    def __init__(self, returncode=0):
        self.returncode = returncode
        self.calls = []

    def execute(self, command, cwd="", **kwargs):
        self.calls.append((command, kwargs.get("stdin_data")))
        return {"output": "", "returncode": self.returncode}


class TestSpillOnNonLocalBackend:
    def _host_spill(self, tmp_path, text):
        p = tmp_path / "out-1-2-beef.log"
        p.write_text(text)
        return p

    def test_spill_is_written_inside_the_environment(self, tmp_path):
        from tools.terminal_tool_result import _redact_spill_file, _SANDBOX_SPILL_DIR
        host = self._host_spill(tmp_path, "head\n" + "sk-proj-" + "a1B2c3D4e5F6g7H8i9J0" * 3 + "\ntail\n")
        env = _RecordingEnv()
        fields = dict(_redact_spill_file(str(host), 99, "cat x", env=env, env_type="docker"))
        assert fields["full_output_path"] == f"{_SANDBOX_SPILL_DIR}/out-1-2-beef.log"
        assert fields["full_output_path"] in fields["truncation_note"]
        command, stdin = env.calls[0]
        assert fields["full_output_path"] in command
        assert "head" in stdin and "tail" in stdin
        assert "a1B2c3D4e5F6g7H8i9J0a1B2c3D4e5F6g7H8i9J0" not in stdin
        assert not host.exists()

    def test_failed_delivery_advertises_no_path(self, tmp_path):
        from tools.terminal_tool_result import _redact_spill_file
        host = self._host_spill(tmp_path, "data\n")
        fields = dict(_redact_spill_file(str(host), 99, "cat x", env=_RecordingEnv(returncode=1),
                                         env_type="docker"))
        assert "full_output_path" not in fields
        assert fields["output_total_chars"] == 99
        assert "redirected to a file" in fields["truncation_note"]
        assert not host.exists()

    def test_local_backend_keeps_host_path(self, tmp_path):
        from tools.terminal_tool_result import _redact_spill_file
        host = self._host_spill(tmp_path, "data\n")
        env = _RecordingEnv()
        fields = dict(_redact_spill_file(str(host), 99, "cat x", env=env, env_type="local"))
        assert fields["full_output_path"] == str(host)
        assert host.exists() and not env.calls


class TestSpillDeliveryThroughARealShell:
    """The delivery command runs in a real shell; only the target directory moves."""

    @pytest.fixture
    def env(self, tmp_path, monkeypatch):
        import tools.terminal_tool_result as ttr
        from tools.environments.local import LocalEnvironment
        monkeypatch.setattr(ttr, "_SANDBOX_SPILL_DIR", str(tmp_path / "spills"))
        return LocalEnvironment(cwd=str(tmp_path), timeout=30)

    def test_delivered_file_is_private_and_readable(self, env, tmp_path):
        from tools.terminal_tool_result import _deliver_spill_to_env
        target = _deliver_spill_to_env(env, Path("out-1-2-aa.log"), "line\n" * 1000)
        assert Path(target).read_text() == "line\n" * 1000
        assert (os.stat(target).st_mode & 0o777) == 0o600

    def test_keeps_only_the_newest_spills(self, env, tmp_path):
        import tools.terminal_tool_result as ttr
        spills = tmp_path / "spills"
        spills.mkdir(mode=0o700)
        for i in range(ttr._SANDBOX_SPILL_KEEP + 3):
            old = spills / f"out-{i}-0-0.log"
            old.write_text("old")
            stamp = time.time() - 600 + i
            os.utime(old, (stamp, stamp))
        ttr._deliver_spill_to_env(env, Path("out-new-0-0.log"), "new")
        kept = sorted(p.name for p in spills.iterdir())
        assert len(kept) == ttr._SANDBOX_SPILL_KEEP
        assert "out-new-0-0.log" in kept and "out-0-0-0.log" not in kept

    def test_refuses_to_write_through_a_planted_file(self, env, tmp_path):
        from tools.terminal_tool_result import _deliver_spill_to_env
        spills = tmp_path / "spills"
        spills.mkdir(mode=0o700)
        victim = tmp_path / "victim"
        victim.write_text("keep")
        (spills / "out-9-9-99.log").symlink_to(victim)
        assert _deliver_spill_to_env(env, Path("out-9-9-99.log"), "overwrite") is None
        assert victim.read_text() == "keep"
