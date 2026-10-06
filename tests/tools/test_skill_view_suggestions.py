"""Error suggestions obey the same session tool gates as discovery (toby #24)."""
import json
from types import SimpleNamespace

import pytest

from tools import skills_tool


@pytest.fixture(params=[False, True], ids=["explicit-metadata", "file-metadata"])
def skills(monkeypatch, tmp_path, request):
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    import hermes_cli.plugins as plugins

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(skills_tool, "_skills_dir", lambda: tmp_path / "skills")
    manager = PluginManager()
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    monkeypatch.setattr(plugins, "discover_plugins", lambda: None)
    ctx = PluginContext(PluginManifest(name="music", version="1.0.0"), manager)
    for name, extra in (("jukebox", "prerequisites:\n  tools: [spotify_search]\n"), ("plain", "")):
        directory = tmp_path / "skills" / "media" / name
        directory.mkdir(parents=True)
        path = directory / "SKILL.md"
        path.write_text(f"---\nname: {name}\ndescription: Test skill\n{extra}---\n\nInstructions.\n", encoding="utf-8")
        metadata = {} if request.param else {"frontmatter": skills_tool._safe_frontmatter(path)}
        ctx.register_skill(name, path, **metadata)
    skills_tool._SKILLS_CACHE.clear()
    yield
    skills_tool._SKILLS_CACHE.clear()


@pytest.mark.parametrize("requested,prefix", [("missing", ""), ("music:missing", "music:")])
@pytest.mark.parametrize("bridge", [False, True])
def test_missing_skill_suggestions_follow_discovery(skills, requested, prefix, bridge, monkeypatch):
    import model_tools
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    # Explicitly defer skill_view to exercise the real tool_call dispatch.
    from hermes_cli.config import load_config, save_config
    config = load_config()
    config.setdefault("tools", {}).setdefault("tool_search", {})["defer"] = ["skill_view"]
    save_config(config)

    def call(name, held):
        from agent.system_prompt import _skill_gate_tools_for_call
        tool = "tool_call" if bridge else "skill_view"
        scoped = None if held is None else _skill_gate_tools_for_call(
            SimpleNamespace(valid_tool_names=held | {tool}, enabled_toolsets=["skills"], disabled_toolsets=None), tool)
        args = {"name": name}
        return json.loads(model_tools.handle_function_call(
            "tool_call" if bridge else "skill_view",
            {"calls": [{"name": "skill_view", "arguments": args}]} if bridge else args,
            skill_gate_tools=scoped, enabled_toolsets=["skills"], skip_pre_tool_call_hook=True))

    for held in ({"skill_view"}, {"skill_view", "spotify_search"}, None, {"skill_view"}):
        expected = {prefix + "plain"}
        if held is None or "spotify_search" in held:
            expected.add(prefix + "jukebox")
        result = call(requested, held)
        assert result.get("success") is False, result
        # Unqualified suggestions can include plugin skills too; compare the requested namespace.
        suggested = {s for s in result["available_skills"] if s.startswith(prefix) and (prefix or ":" not in s)}
        assert suggested == expected


@pytest.mark.parametrize("name", ["jukebox", "music:jukebox", "music:plain"])
def test_explicit_load_preserved_and_siblings_follow_discovery(skills, name):
    import model_tools
    from agent.system_prompt import _skill_gate_tools_for_call
    for held in ({"skill_view"}, {"skill_view", "spotify_search"}, None):
        scoped = None if held is None else _skill_gate_tools_for_call(SimpleNamespace(valid_tool_names=held), "skill_view")
        result = json.loads(model_tools.handle_function_call(
            "skill_view", {"name": name}, skill_gate_tools=scoped, skip_pre_tool_call_hook=True))
        assert result["success"] is True
        assert result["name"] == name
        if name == "music:plain":
            banner = result["content"].split("\n\n", 1)[0]
            assert ("jukebox" in banner) is (held is None or "spotify_search" in held)
