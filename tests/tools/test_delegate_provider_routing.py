"""Configured delegation routes and tool ceilings survive real child construction."""

import copy
import json
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import yaml

from agent.secret_scope import (
    build_profile_secret_scope, is_multiplex_active, reset_secret_scope,
    set_multiplex_active, set_secret_scope,
)
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import delegate_tool
from tools.registry import registry
from toolsets import resolve_toolset


@contextmanager
def _profile(home):
    home_token = set_hermes_home_override(home)
    secret_token = set_secret_scope(build_profile_secret_scope(home), profile_home=str(home))
    try:
        yield
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


def _parent():
    return SimpleNamespace(
        model="parent-model", provider="custom", requested_provider="custom:parent",
        base_url="http://127.0.0.1:1/v1", api_key="parent-key", api_mode="chat_completions",
        enabled_toolsets=["hermes-cli"], disabled_toolsets=[], session_id=None,
        _session_db=None, _delegate_depth=0, _active_children=[], _active_children_lock=threading.Lock(),
        tool_progress_callback=None, thinking_callback=None, _print_fn=None, request_overrides={},
    )


def _capture_batch(batch, background):
    entries = []
    for _, _, child in batch.children:
        try:
            entries.append({
                "provider": child.requested_provider, "model": child.model,
                "base_url": child.base_url, "api_key": child.api_key,
                "tools": sorted(child.valid_tool_names), "role": child._delegate_role,
            })
        finally:
            child.close()
    return json.dumps({"children": entries})


def _write_profile(home, key, cap, *, legacy=False):
    home.mkdir()
    (home / ".env").write_text(f"LOCAL_ROUTING_KEY={key}\n", encoding="utf-8")
    entry = {"name": "Local Worker", "api_key_env": "LOCAL_ROUTING_KEY", "delegation_toolsets": cap}
    entry["base_url" if legacy else "api"] = f"http://127.0.0.1:{8001 if legacy else 8002}/v1"
    config = {
        "delegation": {"max_spawn_depth": 3, "orchestrator_enabled": True, "worktree_isolation": False},
        "custom_providers" if legacy else "providers": [entry] if legacy else {"local-worker": entry},
    }
    (home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return entry


def test_routes_and_operator_ceiling_follow_profile_through_dispatch(tmp_path, monkeypatch):
    from run_agent import AIAgent

    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    monkeypatch.setattr(delegate_tool, "_run_batch", _capture_batch)
    a, b = tmp_path / "a", tmp_path / "b"
    entries = {a: _write_profile(a, "key-a", ["file"]), b: _write_profile(b, "key-b", [], legacy=True)}
    was_multiplex = is_multiplex_active()
    set_multiplex_active(True)
    try:
        for home in (a, b, a):
            with _profile(home):
                args = {
                    "provider": "custom:local-worker", "model": "call-model",
                    "tasks": [
                        {"goal": "Inspect the project", "toolsets": ["terminal", "delegation"]},
                        {"goal": "Review the findings", "provider": "Local Worker", "model": "task-model"},
                    ],
                }
                for dispatch in (
                    lambda: registry.dispatch("delegate_task", args, parent_agent=_parent()),
                    lambda: AIAgent._dispatch_delegate_task(_parent(), args),
                ):
                    children = json.loads(dispatch())["children"]
                    assert [child["model"] for child in children] == ["call-model", "task-model"]
                    for child in children:
                        assert child["provider"].lower().replace(" ", "-") in {"custom:local-worker", "local-worker"}
                        assert child["base_url"] == entries[home].get("api", entries[home].get("base_url"))
                        assert child["api_key"] == ("key-a" if home == a else "key-b")
                        allowed = set(resolve_toolset("file")) if home == a else set()
                        assert set(child["tools"]) <= allowed
                        assert bool(child["tools"]) == bool(allowed)
                        assert child["role"] == "leaf"
    finally:
        set_multiplex_active(was_multiplex)


def test_provider_schema_is_profile_scoped_copy_safe_and_caps_direct_routes(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    monkeypatch.setattr(delegate_tool, "_run_batch", _capture_batch)
    a, b = tmp_path / "a", tmp_path / "b"
    _write_profile(a, "key-a", ["file"])
    _write_profile(b, "key-b", [], legacy=True)
    # Config editing/migration must preserve even an empty or invalid ceiling.
    from hermes_cli.config_providers import _custom_provider_entry_to_provider_config, _normalize_custom_provider_entry
    for cap in (["file"], [], False):
        raw = {"name": "worker", "base_url": "http://127.0.0.1:8002/v1", "delegation_toolsets": cap}
        normalized = _normalize_custom_provider_entry(raw)
        converted = _custom_provider_entry_to_provider_config(normalized)
        assert converted["delegation_toolsets"] == cap
        if isinstance(cap, list):
            normalized["delegation_toolsets"].append("terminal")
            assert raw["delegation_toolsets"] == cap
    static = copy.deepcopy(delegate_tool.DELEGATE_TASK_SCHEMA)
    was_multiplex = is_multiplex_active()
    set_multiplex_active(True)
    try:
        for home in (a, b, a):
            with _profile(home):
                definitions = registry.get_definitions(["delegate_task"])
                schema = definitions[0]["function"]["parameters"]["properties"]
                top = schema["provider"]["description"]
                per_task = schema["tasks"]["items"]["properties"]["provider"]["description"]
                assert top == per_task
                assert "custom:local-worker" in top
                assert ("toolsets: file" if home == a else "toolsets: none") in top
                assert delegate_tool.DELEGATE_TASK_SCHEMA == static
        with _profile(a):
            config_path = a / "config.yaml"
            config = yaml.safe_load(config_path.read_text())
            config["delegation"].update({
                "base_url": "http://127.0.0.1:8003/v1", "api_key": "direct-key", "delegation_toolsets": [],
            })
            config_path.write_text(yaml.safe_dump(config))
            result = json.loads(registry.dispatch("delegate_task", {"goal": "Inspect the project"}, parent_agent=_parent()))
            assert result["children"][0]["tools"] == []
            config["delegation"]["delegation_toolsets"] = False
            config_path.write_text(yaml.safe_dump(config))
            result = json.loads(registry.dispatch("delegate_task", {"goal": "Inspect the project"}, parent_agent=_parent()))
            assert "refusing uncapped delegation" in result["error"]
            config["delegation"].pop("delegation_toolsets")
            config["providers"]["invalid-worker"] = {
                "api": "http://127.0.0.1:8004/v1", "delegation_toolsets": False,
            }
            config_path.write_text(yaml.safe_dump(config))
            parent = _parent()
            result = json.loads(registry.dispatch("delegate_task", {"tasks": [
                {"goal": "Inspect the project", "provider": "custom:local-worker"},
                {"goal": "Review the findings", "provider": "custom:invalid-worker"},
            ]}, parent_agent=parent))
            assert "refusing uncapped delegation" in result["error"]
            assert parent._active_children == []
    finally:
        set_multiplex_active(was_multiplex)
