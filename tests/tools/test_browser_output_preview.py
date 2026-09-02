"""Confined file:///output browser preview behavior."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

import pytest

from tools import browser_tool, output_preview


@pytest.fixture()
def output_root(tmp_path, monkeypatch):
    output_preview.stop_output_preview_server()
    root = tmp_path / "output"
    (root / "apps" / "demo").mkdir(parents=True)
    (root / "apps" / "_assets").mkdir()
    (root / "apps" / "demo" / "index.html").write_text(
        '<h1>preview</h1><script src="../_assets/app.js"></script>',
        encoding="utf-8",
    )
    (root / "apps" / "_assets" / "app.js").write_text(
        'document.body.dataset.loaded = "yes";', encoding="utf-8"
    )
    monkeypatch.setattr(output_preview, "OUTPUT_PREVIEW_ROOT", root)
    yield root
    output_preview.stop_output_preview_server()


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "file:/output/apps/demo/index.html",
        "FILE:///output/apps/demo/index.html",
        "file:///%6futput/apps/demo/index.html",
        "file:///output/../etc/passwd",
        "file:///output/%2e%2e/etc/passwd",
        "file:///output/apps//index.html",
        "file:///output/apps/demo/index.html?mode=preview",
        "file:///output/apps/demo/index.html#fragment",
        "file:///output/apps/demo/index.html?",
        "file:///output/apps/demo/index.html#",
        "file:///output/apps/demo/index.html?#",
        "file:///output/apps/demo/%0aname.html",
        "file:///output/apps/demo/%7fname.html",
        "file:///output/apps/demo/%C2%80name.html",
        "file://localhost/output/apps/demo/index.html",
        "file:///output/apps/demo\\index.html",
        " file:///output/apps/demo/index.html",
        "file:///output/apps/demo/index.html ",
        "file:// /output/apps/demo/index.html",
    ],
)
def test_decode_rejects_every_noncanonical_file_url(url):
    with pytest.raises(ValueError):
        output_preview.decode_output_preview_segments(url)


def test_decode_ignores_non_file_urls():
    assert output_preview.decode_output_preview_segments("https://example.com") is None


def test_decode_accepts_percent_encoded_filename_space():
    assert output_preview.decode_output_preview_segments(
        "file:///output/apps/demo/my%20file.html"
    ) == ["apps", "demo", "my file.html"]


@pytest.mark.parametrize(
    "url",
    [
        " file:///output/apps/demo/index.html",
        "file:///output/apps/demo/index.html ",
        "file:// /output/apps/demo/index.html",
        "file:///output/apps/demo/index.html?",
        "file:///output/apps/demo/index.html#",
        "file:///output/apps/demo/index.html?#",
        "file:///output/apps/demo/%0aname.html",
        "file:///output/apps/demo/%7fname.html",
        "file:///output/apps/demo/%C2%80name.html",
    ],
)
def test_browser_rejects_ambiguous_file_input_before_url_normalization(url):
    result = json.loads(browser_tool.browser_navigate(url, task_id="task"))
    assert result["success"] is False
    assert "local file" in result["error"] or "file:///output/" in result["error"]


def test_loopback_server_serves_main_file_and_relative_asset(output_root):
    preview_url = output_preview.output_preview_url(
        "file:///output/apps/demo/index.html"
    )
    assert preview_url is not None
    parsed = urllib.parse.urlsplit(preview_url)
    assert parsed.hostname == "127.0.0.1"
    assert parsed.query == output_preview.OUTPUT_PREVIEW_QUERY

    with urllib.request.urlopen(preview_url, timeout=2) as response:
        assert response.read().decode() == (
            '<h1>preview</h1><script src="../_assets/app.js"></script>'
        )
        assert response.headers["Cache-Control"] == "no-store"

    asset_url = urllib.parse.urljoin(preview_url, "../_assets/app.js")
    with urllib.request.urlopen(asset_url, timeout=2) as response:
        assert response.read().decode() == 'document.body.dataset.loaded = "yes";'


def test_loopback_server_preserves_preview_only_state_contract(output_root):
    preview_url = output_preview.output_preview_url(
        "file:///output/apps/demo/index.html"
    )
    state_url = urllib.parse.urljoin(preview_url, "./api/v1/state/preferences")
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(state_url, timeout=2)
    assert caught.value.code == 409
    payload = json.loads(caught.value.read())
    assert payload["error"]["code"] == "preview_only"


def test_symlink_cannot_escape_output_root(output_root, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("must not be served", encoding="utf-8")
    (output_root / "apps" / "demo" / "leak.txt").symlink_to(outside)

    with pytest.raises(ValueError, match="unavailable or unsafe"):
        output_preview.output_preview_url("file:///output/apps/demo/leak.txt")


def test_symlinked_directory_cannot_escape_output_root(output_root, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "leak.txt").write_text("must not be served", encoding="utf-8")
    (output_root / "apps" / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="unavailable or unsafe"):
        output_preview.output_preview_url("file:///output/apps/linked/leak.txt")


def test_loopback_request_cannot_traverse_output_root(output_root):
    preview_url = output_preview.output_preview_url(
        "file:///output/apps/demo/index.html"
    )
    parsed = urllib.parse.urlsplit(preview_url)
    traversal_url = f"http://{parsed.netloc}/%2e%2e/outside.txt"

    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(traversal_url, timeout=2)
    assert caught.value.code == 404


def test_directory_is_not_a_preview_target(output_root):
    with pytest.raises(ValueError, match="unavailable or unsafe"):
        output_preview.output_preview_url("file:///output/apps/demo")


def test_browser_navigate_forces_local_sidecar_and_hides_proxy_url(
    output_root, monkeypatch
):
    requested_url = "file:///output/apps/demo/index.html"
    calls = []

    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_is_always_blocked_url", lambda _url: False)
    monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: True)
    monkeypatch.setattr(browser_tool, "check_website_access", lambda _url: None)
    monkeypatch.setattr(browser_tool, "_maybe_start_recording", lambda _task: None)
    monkeypatch.setattr(
        browser_tool,
        "_get_session_info",
        lambda _task: {
            "session_name": "preview",
            "cdp_url": None,
            "_first_nav": False,
        },
    )

    def fake_run(task_id, command, args=None, **_kwargs):
        calls.append((task_id, command, args or []))
        if command == "open":
            return {
                "success": True,
                "data": {"title": "Preview", "url": (args or [""])[0]},
            }
        if command == "snapshot":
            return {
                "success": True,
                "data": {"snapshot": "preview", "refs": {}},
            }
        if command == "eval":
            return {
                "success": True,
                "data": {"result": calls[0][2][0]},
            }
        raise AssertionError(command)

    monkeypatch.setattr(browser_tool, "_run_browser_command", fake_run)

    result = json.loads(browser_tool.browser_navigate(requested_url, task_id="task"))

    assert result["success"] is True
    assert result["url"] == requested_url
    task_id, command, args = calls[0]
    assert task_id == f"task{browser_tool._LOCAL_SUFFIX}"
    assert command == "open"
    assert args[0].startswith("http://127.0.0.1:")
    assert "file:" not in args[0]
    browser_tool._last_active_session_key.pop("task", None)
    with browser_tool._output_preview_lock:
        browser_tool._output_preview_session_keys.discard(task_id)


def test_unconfined_current_file_is_blanked_before_content_return(monkeypatch):
    calls = []
    with browser_tool._output_preview_lock:
        browser_tool._output_preview_session_keys.add("task")

    def fake_run(task_id, command, args=None, **_kwargs):
        calls.append((task_id, command, args or []))
        if command == "eval":
            return {
                "success": True,
                "data": {"result": "file:///etc/passwd"},
            }
        if command == "open":
            return {"success": True}
        raise AssertionError(command)

    monkeypatch.setattr(browser_tool, "_run_browser_command", fake_run)

    blocked = browser_tool._blocked_local_file_page("task", "return content")

    assert blocked is not None
    assert json.loads(blocked)["success"] is False
    assert calls[-1] == ("task", "open", ["about:blank"])
    with browser_tool._output_preview_lock:
        browser_tool._output_preview_session_keys.discard("task")


@pytest.mark.parametrize("probe_raises", [False, True])
def test_unavailable_current_url_probe_blanks_and_blocks(monkeypatch, probe_raises):
    calls = []
    with browser_tool._output_preview_lock:
        browser_tool._output_preview_session_keys.add("task")

    def fake_run(task_id, command, args=None, **_kwargs):
        calls.append((task_id, command, args or []))
        if command == "eval":
            if probe_raises:
                raise RuntimeError("synthetic probe failure")
            return {"success": False, "error": "synthetic probe failure"}
        if command == "open":
            return {"success": True}
        raise AssertionError(command)

    monkeypatch.setattr(browser_tool, "_run_browser_command", fake_run)

    blocked = browser_tool._blocked_local_file_page("task", "return content")

    assert blocked is not None
    payload = json.loads(blocked)
    assert payload["success"] is False
    assert "could not verify" in payload["error"]
    assert calls[-1] == ("task", "open", ["about:blank"])
    with browser_tool._output_preview_lock:
        browser_tool._output_preview_session_keys.discard("task")


@pytest.mark.parametrize("probe_raises", [False, True])
def test_snapshot_withholds_content_when_current_url_probe_fails(
    monkeypatch, probe_raises
):
    with browser_tool._output_preview_lock:
        browser_tool._output_preview_session_keys.add("task")
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)

    def fake_run(_task_id, command, _args=None, **_kwargs):
        if command == "snapshot":
            return {
                "success": True,
                "data": {"snapshot": "FILE PAGE CONTENT", "refs": {}},
            }
        if command == "eval":
            if probe_raises:
                raise RuntimeError("synthetic probe failure")
            return {"success": False, "error": "synthetic probe failure"}
        if command == "open":
            return {"success": True}
        raise AssertionError(command)

    monkeypatch.setattr(browser_tool, "_run_browser_command", fake_run)

    result = json.loads(browser_tool.browser_snapshot(task_id="task"))

    assert result["success"] is False
    assert "FILE PAGE CONTENT" not in json.dumps(result)
    assert "could not verify" in result["error"]
    with browser_tool._output_preview_lock:
        browser_tool._output_preview_session_keys.discard("task")
