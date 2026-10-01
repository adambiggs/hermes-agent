#!/usr/bin/env python3
"""
Tests for file staleness detection in write_file and patch.

write_file refuses (before any disk mutation) to overwrite an existing file the
task never read in full or that changed on disk since that read; patch stays
warning-only for stale reads.

Run with:  python -m pytest tests/tools/test_file_staleness.py -v
"""

import json
import os
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock

from tools import file_state
from tools.file_tools import read_file_tool, write_file_tool, patch_tool
from tools.file_tools_read_tracking import _read_tracker, reset_file_dedup
from tools.file_operations import ShellFileOperations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeReadResult:
    def __init__(self, content="line1\nline2\n", total_lines=2, file_size=100):
        self.content = content
        self._total_lines = total_lines
        self._file_size = file_size

    def to_dict(self):
        return {
            "content": self.content,
            "total_lines": self._total_lines,
            "file_size": self._file_size,
        }


class _FakeWriteResult:
    def __init__(self):
        self.bytes_written = 10

    def to_dict(self):
        return {"bytes_written": self.bytes_written}


class _FakePatchResult:
    def __init__(self):
        self.success = True

    def to_dict(self):
        return {"success": True, "diff": "--- a\n+++ b\n@@ ...\n"}


def _make_fake_ops(read_content="hello\n", file_size=6):
    fake = MagicMock(env=None)
    fake.read_file = lambda path, offset=1, limit=500: _FakeReadResult(
        content=read_content, total_lines=1, file_size=file_size,
    )
    fake.write_file = lambda path, content: _FakeWriteResult()
    fake.patch_replace = lambda path, old, new, replace_all=False: _FakePatchResult()
    return fake


def _real_shell_ops(cwd: str, path_map: tuple[str, str] | None = None,
                    sha256sum_available: bool = True, backend_home: str | None = None,
                    portable_only: bool = False):
    env = MagicMock()
    env.cwd = cwd

    def execute(command, cwd=None, timeout=None, stdin_data=None):
        if path_map is not None:
            command = command.replace(*path_map)
        if portable_only and command.startswith(("realpath -m ", "python3 -c ", "python -c ")):
            return {"returncode": 127, "output": "command not found"}
        if (portable_only or not sha256sum_available) and command.startswith("sha256sum "):
            return {"returncode": 127, "output": "sha256sum: command not found"}
        result = subprocess.run(command, shell=True, cwd=cwd, timeout=timeout,
                                input=stdin_data, text=True, capture_output=True,
                                env={**os.environ, "HOME": backend_home} if backend_home else None)
        return {"returncode": result.returncode, "output": result.stdout}

    env.execute.side_effect = execute
    return ShellFileOperations(env)


def _modify_externally(path: str, content: str) -> None:
    """Rewrite *path* so its mtime provably differs from the pre-write stamp."""
    before = os.path.getmtime(path)
    with open(path, "w") as f:
        f.write(content)
    if os.path.getmtime(path) == before:
        os.utime(path, (before + 1.0, before + 1.0))


# ---------------------------------------------------------------------------
# write_file: refuse stale / unread overwrites before touching the disk
# ---------------------------------------------------------------------------

class TestStalenessCheck(unittest.TestCase):

    def setUp(self):
        _read_tracker.clear()
        file_state.get_registry().clear()
        self._tmpdir = tempfile.mkdtemp()
        self._tmpfile = os.path.join(self._tmpdir, "stale_test.txt")
        with open(self._tmpfile, "w") as f:
            f.write("original content\n")

    def tearDown(self):
        _read_tracker.clear()
        file_state.get_registry().clear()
        try:
            os.unlink(self._tmpfile)
            os.rmdir(self._tmpdir)
        except OSError:
            pass

    @patch("tools.file_tools._get_file_ops")
    def test_no_warning_when_file_unchanged(self, mock_ops):
        """Read then write with no external modification — no warning."""
        mock_ops.return_value = _make_fake_ops("original content\n", 18)
        read_file_tool(self._tmpfile, task_id="t1")

        result = json.loads(write_file_tool(self._tmpfile, "new content", task_id="t1"))
        self.assertNotIn("_warning", result)
        self.assertNotIn("error", result)

    def test_write_file_refuses_before_mutation_when_modified_externally(self):
        """read → external edit → write_file: refused, external edit preserved;
        a full re-read heals the baseline and the next write lands."""
        self.assertNotIn("error", json.loads(read_file_tool(self._tmpfile, task_id="t1")))
        _modify_externally(self._tmpfile, "someone else changed this\n")

        refused = json.loads(write_file_tool(self._tmpfile, "new content\n", task_id="t1"))
        self.assertTrue(refused.get("stale_write_blocked"), refused)
        with open(self._tmpfile) as f:
            self.assertEqual(f.read(), "someone else changed this\n")

        self.assertNotIn("error", json.loads(read_file_tool(self._tmpfile, task_id="t1")))
        written = json.loads(write_file_tool(self._tmpfile, "merged\n", task_id="t1"))
        self.assertNotIn("error", written)
        with open(self._tmpfile) as f:
            self.assertEqual(f.read(), "merged\n")

    @patch("tools.file_tools._get_file_ops")
    def test_complete_shell_backend_read_allows_write_but_changed_file_does_not(self, mock_ops):
        """The backend namespace supplies the read and version proof for writes."""
        mock_ops.return_value = _real_shell_ops(self._tmpdir)
        with open(self._tmpfile, "w") as stream:
            stream.write("one\ntwo\nthree\n")

        partial = json.loads(read_file_tool(self._tmpfile, limit=1, task_id="remote"))
        self.assertTrue(partial["truncated"])
        self.assertTrue(json.loads(write_file_tool(self._tmpfile, "stale\n", task_id="remote"))
                        .get("stale_write_blocked"))
        read = json.loads(read_file_tool(self._tmpfile, limit=10, task_id="remote"))
        self.assertEqual(read["total_lines"], 3)
        self.assertFalse(read["truncated"])
        written = json.loads(write_file_tool(self._tmpfile, "merged\n", task_id="remote"))
        self.assertNotIn("error", written, written)
        with open(self._tmpfile) as stream:
            self.assertEqual(stream.read(), "merged\n")

        json.loads(read_file_tool(self._tmpfile, task_id="remote"))
        _modify_externally(self._tmpfile, "changed elsewhere\n")
        reset_file_dedup("remote")
        refused = json.loads(write_file_tool(self._tmpfile, "stale\n", task_id="remote"))
        self.assertTrue(refused.get("stale_write_blocked"), refused)
        with open(self._tmpfile) as stream:
            self.assertEqual(stream.read(), "changed elsewhere\n")

    @patch("tools.file_tools._get_file_ops")
    def test_backend_only_existing_file_requires_current_full_read(self, mock_ops):
        """A remote file with no host counterpart is still an existing file."""
        remote_path = self._tmpfile + "-backend"
        self.assertFalse(os.path.exists(remote_path))
        mock_ops.return_value = _real_shell_ops(
            self._tmpdir, (remote_path, self._tmpfile), portable_only=True)
        with open(self._tmpfile, "w") as stream:
            stream.write("one\ntwo\nthree\n")

        partial = json.loads(read_file_tool(remote_path, limit=1, task_id="remote-only"))
        self.assertTrue(partial["truncated"])
        refused = json.loads(write_file_tool(remote_path, "stale\n", task_id="remote-only"))
        self.assertTrue(refused.get("stale_write_blocked"), refused)

        full = json.loads(read_file_tool(remote_path, task_id="remote-only"))
        self.assertFalse(full["truncated"])
        written = json.loads(write_file_tool(remote_path, "merged\n", task_id="remote-only"))
        self.assertNotIn("error", written, written)
        rewritten = json.loads(write_file_tool(remote_path, "merged again\n", task_id="remote-only"))
        self.assertNotIn("error", rewritten, rewritten)

        json.loads(read_file_tool(remote_path, task_id="remote-only"))
        _modify_externally(self._tmpfile, "changed elsewhere\n")
        refused = json.loads(write_file_tool(remote_path, "stale\n", task_id="remote-only"))
        self.assertTrue(refused.get("stale_write_blocked"), refused)
        with open(self._tmpfile) as stream:
            self.assertEqual(stream.read(), "changed elsewhere\n")

        new_remote = os.path.join(self._tmpdir, "portable-new-dir", "new.txt")
        created = json.loads(write_file_tool(new_remote, "portable\n", task_id="portable-new"))
        self.assertNotIn("error", created, created)
        with open(new_remote) as stream:
            self.assertEqual(stream.read(), "portable\n")
        os.unlink(new_remote)
        os.rmdir(os.path.dirname(new_remote))
        missing_top = f"/__hermes_missing_{os.path.basename(self._tmpdir)}/file.txt"
        self.assertEqual(mock_ops.return_value.resolve_backend_path(missing_top), missing_top)

        remote_notebook = remote_path + ".ipynb"
        backend_notebook = self._tmpfile + ".ipynb"
        notebook = ('{"cells":[{"cell_type":"markdown","metadata":{},'
                    '"source":["hello notebook"]}],"metadata":{},'
                    '"nbformat":4,"nbformat_minor":5}')
        with open(backend_notebook, "w") as stream:
            stream.write(notebook)
        mock_ops.return_value = _real_shell_ops(self._tmpdir, (remote_notebook, backend_notebook))
        extracted = json.loads(read_file_tool(remote_notebook, task_id="remote-notebook"))
        self.assertTrue(extracted["extracted_document"], extracted)
        self.assertNotIn("error", json.loads(write_file_tool(
            remote_notebook, notebook.replace("hello notebook", "updated notebook"),
            task_id="remote-notebook")))
        with open(backend_notebook) as stream:
            self.assertIn("updated notebook", stream.read())
        os.unlink(backend_notebook)

        from tools import terminal_tool
        workspace = os.path.join(self._tmpdir, "workspace")
        decoy = os.path.join(self._tmpdir, "backend-cwd")
        os.makedirs(workspace)
        os.makedirs(decoy)
        workspace_file = os.path.join(workspace, "target.txt")
        decoy_file = os.path.join(decoy, "target.txt")
        with open(workspace_file, "w") as stream:
            stream.write("workspace version\n")
        with open(decoy_file, "w") as stream:
            stream.write("backend cwd version\n")
        mock_ops.return_value = _real_shell_ops(decoy)
        fresh = json.loads(read_file_tool("target.txt", task_id="remote-first-read"))
        self.assertIn("backend cwd version", fresh["content"])
        terminal_tool.record_session_cwd("remote-relative", workspace)
        try:
            read = json.loads(read_file_tool("target.txt", task_id="remote-relative"))
            self.assertIn("workspace version", read["content"])
            written = json.loads(write_file_tool("target.txt", "updated\n", task_id="remote-relative"))
            self.assertNotIn("error", written, written)
        finally:
            terminal_tool.clear_session_cwd("remote-relative")
        with open(workspace_file) as stream:
            self.assertEqual(stream.read(), "updated\n")
        with open(decoy_file) as stream:
            self.assertEqual(stream.read(), "backend cwd version\n")
        os.unlink(workspace_file)
        os.unlink(decoy_file)
        os.rmdir(workspace)
        os.rmdir(decoy)

        backend_home = os.path.join(self._tmpdir, "backend-home")
        os.makedirs(backend_home)
        home_file = os.path.join(backend_home, "prefs.txt")
        with open(home_file, "w") as stream:
            stream.write("remote home\n")
        mock_ops.return_value = _real_shell_ops(self._tmpdir, backend_home=backend_home)
        self.assertIn("remote home", json.loads(read_file_tool("~/prefs.txt", task_id="remote-home"))["content"])
        self.assertNotIn("error", json.loads(write_file_tool("~/prefs.txt", "updated home\n", task_id="remote-home")))
        with open(home_file) as stream:
            self.assertEqual(stream.read(), "updated home\n")
        ssh_dir = os.path.join(backend_home, ".ssh")
        os.mkdir(ssh_dir)
        ssh_key = os.path.join(ssh_dir, "authorized_keys")
        ssh_config = os.path.join(ssh_dir, "config")
        for guarded_path in ("~/.ssh/authorized_keys", ssh_key, "~/.ssh/config", ssh_config):
            denied = json.loads(write_file_tool(guarded_path, "unsafe\n", task_id="remote-home"))
            self.assertIn("error", denied, denied)
        self.assertFalse(os.path.exists(ssh_key))
        self.assertFalse(os.path.exists(ssh_config))
        with open(ssh_key, "w") as stream:
            stream.write("old key\n")
        remote_link = os.path.join(self._tmpdir, "innocent-link.txt")
        os.symlink(ssh_key, remote_link)
        linked_read = json.loads(read_file_tool(remote_link, task_id="remote-home"))
        self.assertIn("old key", linked_read["content"])
        linked_write = json.loads(write_file_tool(remote_link, "unsafe\n", task_id="remote-home"))
        self.assertIn("error", linked_write, linked_write)
        with open(ssh_key) as stream:
            self.assertEqual(stream.read(), "old key\n")
        os.unlink(remote_link)
        os.unlink(ssh_key)
        os.symlink(home_file, ssh_config)
        self.assertIn("error", json.loads(write_file_tool(
            "~/.ssh/config", "unsafe\n", task_id="remote-home")))
        with open(home_file) as stream:
            self.assertEqual(stream.read(), "updated home\n")
        os.unlink(ssh_config)
        hermes_dir = os.path.join(backend_home, ".hermes")
        os.mkdir(hermes_dir)
        auth_file = os.path.join(hermes_dir, "auth.json")
        with open(auth_file, "w") as stream:
            stream.write('{"token":"secret"}\n')
        auth_link = os.path.join(self._tmpdir, "innocent-auth.txt")
        os.symlink(auth_file, auth_link)
        for guarded_path in ("~/.hermes/auth.json", auth_file, auth_link):
            denied = json.loads(read_file_tool(guarded_path, task_id="remote-home"))
            self.assertIn("error", denied, denied)
            self.assertNotIn("secret", denied.get("content", ""))
        os.unlink(auth_link)
        os.unlink(auth_file)
        os.rmdir(hermes_dir)
        os.rmdir(ssh_dir)
        os.unlink(home_file)
        os.rmdir(backend_home)

        backend_file = os.path.join(self._tmpdir, "backend-file.txt")
        host_link = os.path.join(self._tmpdir, "linked.txt")
        with open(backend_file, "w") as stream:
            stream.write("backend file\n")
        os.symlink("/etc/passwd", host_link)
        mock_ops.return_value = _real_shell_ops(self._tmpdir, (host_link, backend_file))
        self.assertIn("backend file", json.loads(read_file_tool(host_link, task_id="remote-link"))["content"])
        self.assertNotIn("error", json.loads(write_file_tool(host_link, "updated file\n", task_id="remote-link")))
        with open(backend_file) as stream:
            self.assertEqual(stream.read(), "updated file\n")
        os.unlink(host_link)
        os.unlink(backend_file)

    def test_write_file_requires_full_unredacted_read_of_existing_file(self):
        """Existing file with no baseline is refused untouched: never read, only
        patched, read partially, or read redacted (the «redacted:…» sentinel must
        never be persisted). A net-new file needs no baseline and the task's own
        write is a baseline for its next write."""
        refused = json.loads(write_file_tool(self._tmpfile, "x\n", task_id="t2"))
        self.assertTrue(refused.get("stale_write_blocked"), refused)

        patched = json.loads(patch_tool(mode="replace", path=self._tmpfile,
                                        old_string="original", new_string="patched", task_id="t2"))
        self.assertNotIn("error", patched)
        self.assertTrue(json.loads(write_file_tool(self._tmpfile, "x\n", task_id="t2")).get("stale_write_blocked"))

        with open(self._tmpfile, "w") as f:
            f.write("one\ntwo\nthree\n")
        self.assertNotIn("error", json.loads(read_file_tool(self._tmpfile, offset=1, limit=1, task_id="t2")))
        self.assertTrue(json.loads(write_file_tool(self._tmpfile, "x\n", task_id="t2")).get("stale_write_blocked"))

        secret = "ghp_" + "A" * 40
        with open(self._tmpfile, "w") as f:
            f.write(f"token={secret}\n")
        with patch("agent.redact._REDACT_ENABLED", True):
            read = json.loads(read_file_tool(self._tmpfile, task_id="t2"))
            self.assertNotIn(secret, read["content"])
            refused = json.loads(write_file_tool(self._tmpfile, "token=«redacted:ghp_…»\n", task_id="t2"))
        self.assertTrue(refused.get("stale_write_blocked"), refused)
        with open(self._tmpfile) as f:
            self.assertEqual(f.read(), f"token={secret}\n")

        new_path = os.path.join(self._tmpdir, "brand_new.txt")
        self.assertNotIn("error", json.loads(write_file_tool(new_path, "one\n", task_id="t2")))
        self.assertNotIn("error", json.loads(write_file_tool(new_path, "two\n", task_id="t2")))
        with open(new_path) as f:
            self.assertEqual(f.read(), "two\n")
        os.unlink(new_path)

    def test_paged_read_of_large_file_is_a_full_baseline_that_survives_compaction(self):
        """A file too big for one read_file page (>2000 lines) can only be seen by
        paging; contiguous pages reaching the last line at one mtime count as a full
        read, so write_file is not permanently refused. A compaction reset keeps that
        baseline while the file is unchanged, and an edit between pages voids it."""
        with open(self._tmpfile, "w") as f:
            f.write("".join(f"line {i}\n" for i in range(1, 2501)))
        first = json.loads(read_file_tool(self._tmpfile, task_id="t3"))
        self.assertTrue(first.get("truncated"), first)
        self.assertTrue(json.loads(write_file_tool(self._tmpfile, "x\n", task_id="t3")).get("stale_write_blocked"))

        self.assertNotIn("error", json.loads(read_file_tool(self._tmpfile, offset=2001, task_id="t3")))
        reset_file_dedup("t3")
        written = json.loads(write_file_tool(self._tmpfile, "merged\n", task_id="t3"))
        self.assertNotIn("error", written, written)
        with open(self._tmpfile) as f:
            self.assertEqual(f.read(), "merged\n")

        with open(self._tmpfile, "w") as f:
            f.write("".join(f"line {i}\n" for i in range(1, 2501)))
        json.loads(read_file_tool(self._tmpfile, task_id="t3"))
        _modify_externally(self._tmpfile, "".join(f"other {i}\n" for i in range(1, 2501)))
        json.loads(read_file_tool(self._tmpfile, offset=2001, task_id="t3"))
        refused = json.loads(write_file_tool(self._tmpfile, "x\n", task_id="t3"))
        self.assertTrue(refused.get("stale_write_blocked"), refused)
        self.assertNotIn("Warning:", refused["error"])


    @patch("tools.file_tools._get_file_ops")
    def test_relative_path_uses_recorded_session_cwd_for_staleness_tracking(self, mock_ops):
        """Relative-path stale tracking must follow the session's recorded cwd."""
        start_dir = os.path.join(self._tmpdir, "start")
        live_dir = os.path.join(self._tmpdir, "worktree")
        os.makedirs(start_dir, exist_ok=True)
        os.makedirs(live_dir, exist_ok=True)

        start_file = os.path.join(start_dir, "shared.txt")
        live_file = os.path.join(live_dir, "shared.txt")
        with open(start_file, "w") as f:
            f.write("start copy\n")
        with open(live_file, "w") as f:
            f.write("live copy\n")

        fake_ops = _make_fake_ops("live copy\n", 10)
        fake_ops.write_file = MagicMock(side_effect=AssertionError("must not write stale content"))
        mock_ops.return_value = fake_ops

        from tools import terminal_tool

        # The session cd'd into the worktree (recorded by the completed command).
        terminal_tool.record_session_cwd("live_task", live_dir)

        try:
            with patch.dict(os.environ, {"TERMINAL_CWD": start_dir}, clear=False):
                read_file_tool("shared.txt", task_id="live_task")

                time.sleep(0.05)
                with open(live_file, "w") as f:
                    f.write("live copy modified elsewhere\n")

                result = json.loads(
                    write_file_tool("shared.txt", "replacement", task_id="live_task")
                )
        finally:
            terminal_tool.clear_session_cwd("live_task")

        self.assertTrue(result.get("stale_write_blocked"), result)
        fake_ops.write_file.assert_not_called()


# ---------------------------------------------------------------------------
# Staleness in patch
# ---------------------------------------------------------------------------

class TestPatchStaleness(unittest.TestCase):

    def setUp(self):
        _read_tracker.clear()
        file_state.get_registry().clear()
        self._tmpdir = tempfile.mkdtemp()
        self._tmpfile = os.path.join(self._tmpdir, "patch_test.txt")
        with open(self._tmpfile, "w") as f:
            f.write("original line\n")

    def tearDown(self):
        _read_tracker.clear()
        file_state.get_registry().clear()
        try:
            os.unlink(self._tmpfile)
            os.rmdir(self._tmpdir)
        except OSError:
            pass

    @patch("tools.file_tools._get_file_ops")
    def test_patch_warns_on_stale_file(self, mock_ops):
        """Patch should warn if the target file changed since last read."""
        mock_ops.return_value = _make_fake_ops("original line\n", 15)
        read_file_tool(self._tmpfile, task_id="p1")

        time.sleep(0.05)
        with open(self._tmpfile, "w") as f:
            f.write("externally modified\n")

        result = json.loads(patch_tool(
            mode="replace", path=self._tmpfile,
            old_string="original", new_string="patched",
            task_id="p1",
        ))
        self.assertIn("_warning", result)
        self.assertIn("modified since you last read", result["_warning"])

    @patch("tools.file_tools._get_file_ops")
    def test_patch_no_warning_when_fresh(self, mock_ops):
        """Patch with no external changes — no warning."""
        mock_ops.return_value = _make_fake_ops("original line\n", 15)
        read_file_tool(self._tmpfile, task_id="p2")

        result = json.loads(patch_tool(
            mode="replace", path=self._tmpfile,
            old_string="original", new_string="patched",
            task_id="p2",
        ))
        self.assertNotIn("_warning", result)


# ---------------------------------------------------------------------------
# Unit test for the helper
# ---------------------------------------------------------------------------



if __name__ == "__main__":
    unittest.main()
