"""Skill index tool gates: ``prerequisites.tools`` hides a skill like
``requires_tools``, and a tool deferred behind ``tool_call`` counts as held."""
import json
from types import SimpleNamespace

import pytest

from agent import system_prompt
from agent.prompt_builder import build_skills_system_prompt, clear_skills_system_prompt_cache
from agent.skill_utils import extract_skill_conditions


@pytest.fixture(autouse=True)
def _clear_skills_cache():
    clear_skills_system_prompt_cache(clear_snapshot=True)
    yield
    clear_skills_system_prompt_cache(clear_snapshot=True)


def _write_skill(root, name, frontmatter_extra):
    skill_dir = root / "skills" / "media" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\ndescription: Test skill\n{frontmatter_extra}---\n")


class TestPrerequisitesTools:
    def test_merged_into_requires_tools(self):
        conditions = extract_skill_conditions({
            "prerequisites": {"tools": ["spotify_search", "spotify_playback"]},
            "metadata": {"hermes": {"requires_tools": ["spotify_search"]}},
        })
        assert conditions["requires_tools"] == ["spotify_search", "spotify_playback"]

    def test_malformed_prerequisites_ignored(self):
        assert extract_skill_conditions({"prerequisites": {"tools": "spotify_search"}})["requires_tools"] == []
        assert extract_skill_conditions({"prerequisites": ["spotify_search"]})["requires_tools"] == []

    def test_index_hides_skill_without_its_prerequisite_tools(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _write_skill(tmp_path, "jukebox", "prerequisites:\n  tools: [spotify_search]\n")
        assert "jukebox" not in build_skills_system_prompt(available_tools={"terminal"}, available_toolsets=set())

    def test_index_shows_skill_with_its_prerequisite_tools(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _write_skill(tmp_path, "jukebox", "prerequisites:\n  tools: [spotify_search]\n")
        assert "jukebox" in build_skills_system_prompt(available_tools={"spotify_search"}, available_toolsets=set())

    def test_snapshot_written_before_prerequisites_gating_is_rebuilt(self, monkeypatch, tmp_path):
        # A v3 snapshot stored conditions without prerequisites.tools; trusting it would keep the skill listed.
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _write_skill(tmp_path, "jukebox", "prerequisites:\n  tools: [spotify_search]\n")
        build_skills_system_prompt(available_tools={"terminal"}, available_toolsets=set())
        path = tmp_path / ".skills_prompt_snapshot.json"
        snapshot = json.loads(path.read_text())
        snapshot["version"] = 3
        for entry in snapshot["skills"]:
            entry["conditions"]["requires_tools"] = []
        path.write_text(json.dumps(snapshot))
        clear_skills_system_prompt_cache()
        assert "jukebox" not in build_skills_system_prompt(available_tools={"terminal"}, available_toolsets=set())


class TestDeferredToolsCountAsHeld:
    @pytest.fixture
    def scoped(self, monkeypatch):
        import agent.tool_executor as tool_executor
        monkeypatch.setattr(tool_executor, "_tool_search_scoped_names", lambda agent: frozenset({"cronjob_manage"}))

    def test_deferred_tool_reachable_through_tool_call(self, scoped):
        agent = SimpleNamespace(valid_tool_names={"terminal", "tool_search", "tool_describe", "tool_call"})
        assert "cronjob_manage" in system_prompt._skill_gate_tool_names(agent)

    def test_deferred_tool_not_counted_without_tool_call(self, scoped):
        agent = SimpleNamespace(valid_tool_names={"terminal"})
        assert system_prompt._skill_gate_tool_names(agent) == {"terminal"}

    def test_skills_prompt_shows_skill_requiring_a_deferred_tool(self, scoped, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _write_skill(tmp_path, "watcher", "metadata:\n  hermes:\n    requires_tools: [cronjob_manage]\n")
        held = {"skill_view", "terminal", "tool_search", "tool_describe", "tool_call"}
        agent = SimpleNamespace(valid_tool_names=held, platform="cli", _hermes_home=None)
        monkeypatch.setattr(system_prompt, "_agent_skills_dir", lambda agent: None)
        assert "watcher" in system_prompt._skills_prompt(agent)
        clear_skills_system_prompt_cache(clear_snapshot=True)
        agent.valid_tool_names = held - {"tool_call"}
        assert "watcher" not in system_prompt._skills_prompt(agent)

    def test_skills_prompt_derives_toolsets_from_deferred_tools(self, scoped, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _write_skill(tmp_path, "watcher", "metadata:\n  hermes:\n    requires_toolsets: [cronjob]\n")
        held = {"skill_view", "terminal", "tool_search", "tool_describe", "tool_call"}
        agent = SimpleNamespace(valid_tool_names=held, platform="cli", _hermes_home=None)
        monkeypatch.setattr(system_prompt, "_agent_skills_dir", lambda agent: None)
        assert "watcher" in system_prompt._skills_prompt(agent)


@pytest.mark.parametrize(("enabled", "disabled", "reachable"), [
    (["hermes-cli"], None, True),
    (["hermes-cli"], ["cronjob"], False),
    (["terminal", "file", "web"], None, False),
])
def test_deferred_tools_follow_the_session_toolsets(enabled, disabled, reachable, monkeypatch):
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    agent = SimpleNamespace(valid_tool_names={"terminal", "tool_search", "tool_describe", "tool_call"},
                            enabled_toolsets=enabled, disabled_toolsets=disabled)
    assert ("cronjob_manage" in system_prompt._skill_gate_tool_names(agent)) is reachable


class TestSkillsListFollowsTheIndexGate:
    """skills_list offers only what the index offers, so a session is never pointed at a skill
    whose instructions call tools it does not hold."""

    @pytest.fixture
    def home(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import tools.skills_tool as skills_tool
        monkeypatch.setattr(skills_tool, "_skills_dir", lambda: tmp_path / "skills")
        skills_tool._SKILLS_CACHE.clear()
        _write_skill(tmp_path, "jukebox", "prerequisites:\n  tools: [spotify_search]\n")
        _write_skill(tmp_path, "plain", "")
        yield tmp_path
        skills_tool._SKILLS_CACHE.clear()

    @staticmethod
    def _listed(**kw):
        from tools.skills_tool import skills_list
        return {s["name"] for s in json.loads(skills_list(**kw))["skills"]}

    def test_hides_skill_without_its_tools(self, home):
        assert self._listed(held_tools={"terminal", "skills_list"}) == {"plain"}

    def test_shows_skill_with_its_tools(self, home):
        assert self._listed(held_tools={"spotify_search"}) == {"jukebox", "plain"}

    def test_lists_everything_without_session_info(self, home):
        assert self._listed() == {"jukebox", "plain"}

    def test_dispatch_passes_the_skill_gate_tools(self, home):
        import model_tools
        result = model_tools.handle_function_call("skills_list", {}, skill_gate_tools={"terminal", "skills_list"},
                                                  skip_pre_tool_call_hook=True)
        assert {s["name"] for s in json.loads(result)["skills"]} == {"plain"}

    def test_platform_gate_applies_without_session_tools(self, home, monkeypatch):
        _write_skill(home, "telegram-only", "metadata:\n  hermes:\n    session_platforms: [telegram]\n")
        monkeypatch.delenv("HERMES_PLATFORM", raising=False)
        monkeypatch.setenv("HERMES_SESSION_PLATFORM", "discord")
        assert self._listed() == {"jukebox", "plain"}
        monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
        assert self._listed() == {"jukebox", "plain", "telegram-only"}


class TestAgentDispatchHandsSkillsListTheHeldSet:
    """Both agent dispatch paths (sequential and concurrent) pass the index's held set to
    skills_list, and nothing to tools that never read it."""

    @pytest.fixture
    def captured(self, monkeypatch):
        import model_tools
        calls = []
        monkeypatch.setattr(model_tools, "handle_function_call",
                            lambda name, args, task_id=None, **kw: calls.append((name, kw)) or "{}")
        return calls

    @staticmethod
    def _agent():
        return SimpleNamespace(valid_tool_names=["terminal", "skills_list"], session_id="s", quiet_mode=False,
                               _context_engine_tool_names=None, _memory_manager=None)

    @pytest.mark.parametrize("name, expected", [("skills_list", {"terminal", "skills_list"}), ("terminal", None)])
    def test_sequential_dispatch(self, captured, name, expected):
        from agent.tool_executor import _ToolCallRef, _resolve_sequential_dispatch
        dispatch = _resolve_sequential_dispatch(self._agent(), _ToolCallRef(name, {}, "t", "c", []), [])
        dispatch.execute({})
        assert captured == [(name, captured[0][1])] and captured[0][1]["skill_gate_tools"] == expected

    @pytest.mark.parametrize("name, expected", [("skills_list", {"terminal", "skills_list"}), ("terminal", None)])
    def test_concurrent_invoke_tool(self, captured, name, expected):
        from agent.agent_runtime_helpers import invoke_tool
        invoke_tool(self._agent(), name, {}, "t", tool_call_id="c", messages=[], pre_tool_block_checked=True,
                    skip_tool_request_middleware=True, skip_tool_execution_middleware=True)
        assert captured == [(name, captured[0][1])] and captured[0][1]["skill_gate_tools"] == expected
