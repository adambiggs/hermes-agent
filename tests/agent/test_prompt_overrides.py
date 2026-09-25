"""Named prompt overrides preserve profile isolation, fragment gates and cache lifetime."""

from types import SimpleNamespace

import pytest

from agent.prompt_overrides import apply_fragment_override, normalize_overrides


@pytest.mark.parametrize("spec, expected", [
    ("custom", "custom"),
    ({"mode": "replace", "text": "custom"}, "custom"),
    ({"mode": "append", "text": "custom"}, "built-in\n\ncustom"),
    ({"mode": "prepend", "text": "custom"}, "custom\n\nbuilt-in"),
    ({"mode": "remove"}, None),
    ({"mode": "invalid", "text": "custom"}, "built-in"),
    ({"mode": "append", "text": []}, "built-in"),
    ({"mode": "prepend", "text": " "}, "built-in"),
    ([], "built-in"),
])
def test_normalized_overrides_transform_only_the_named_fragment(spec, expected):
    overrides = normalize_overrides({"task_completion": spec, "unknown": "ignored"})
    assert apply_fragment_override(overrides, "task_completion", "built-in") == expected
    assert apply_fragment_override(overrides, "identity", "identity") == "identity"
    assert "unknown" not in overrides
    for malformed in (None, [], True, 42, "invalid"):
        assert normalize_overrides(malformed) == {}


def test_profile_config_overrides_are_frozen_and_preserve_fragment_gates(tmp_path, monkeypatch):
    from agent.agent_init import _apply_agent_section
    from agent.prompt_builder import (
        TASK_COMPLETION_GUIDANCE, TOOL_USE_ENFORCEMENT_GUIDANCE, execution_guidance_text,
    )
    from agent.secret_scope import (
        is_multiplex_active, reset_secret_scope, set_multiplex_active, set_secret_scope,
    )
    from agent.system_prompt import build_system_prompt_parts
    from hermes_cli.config import atomic_config_write, load_config_readonly
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    launch_home = tmp_path / ".hermes"
    launch_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    homes = [launch_home / "profiles" / name for name in ("a", "b")]
    for home in homes:
        home.mkdir(parents=True)
        atomic_config_write(home / "config.yaml", {"agent": {
            "environment_probe": False, "bot_mode_protocol": False,
            "tool_use_enforcement": True, "execution_guidance": True,
            "prompt_overrides": {
                "task_completion": {"mode": "append", "text": f"completion-{home.name}"},
                "tool_use_enforcement": f"enforcement-{home.name}",
                "execution_discipline": {"mode": "prepend", "text": f"execution-{home.name}"},
            },
        }})

    previous_multiplex = is_multiplex_active()
    set_multiplex_active(True)
    rendered = []
    try:
        for home in (homes[0], homes[1], homes[0]):
            home_token = set_hermes_home_override(str(home))
            secret_token = set_secret_scope({}, profile_home=str(home))
            try:
                agent = SimpleNamespace(
                    run_budget_seconds=None, load_soul_identity=False, skip_context_files=True,
                    valid_tool_names={"memory"}, model="gpt-test", provider="", platform="",
                    _memory_store=None, _memory_manager=None, _kanban_worker_guidance="",
                    pass_session_id=False, session_id="",
                )
                _apply_agent_section(agent, load_config_readonly())
                parts = build_system_prompt_parts(agent)
                stable = parts["stable"]
                assert f"{TASK_COMPLETION_GUIDANCE.strip()}\n\ncompletion-{home.name}" in stable
                assert f"enforcement-{home.name}" in stable
                assert TOOL_USE_ENFORCEMENT_GUIDANCE.strip() not in stable
                assert f"execution-{home.name}\n\n{execution_guidance_text().strip()}" in stable
                assert f"completion-{'b' if home.name == 'a' else 'a'}" not in stable
                rendered.append(parts)

                config_path = home / "config.yaml"
                old_config = load_config_readonly()
                atomic_config_write(config_path, {"agent": {"prompt_overrides": {
                    "task_completion": "edited-during-conversation",
                }}})
                assert load_config_readonly()["agent"]["prompt_overrides"]["task_completion"] == "edited-during-conversation"
                assert build_system_prompt_parts(agent) == parts
                atomic_config_write(config_path, old_config)

                agent.valid_tool_names = set()
                gated = build_system_prompt_parts(agent)["stable"]
                assert f"completion-{home.name}" not in gated
                assert f"enforcement-{home.name}" not in gated
                assert f"execution-{home.name}" not in gated
                agent._prompt_overrides = {}
                no_overrides = build_system_prompt_parts(agent)
                del agent._prompt_overrides
                assert build_system_prompt_parts(agent) == no_overrides
            finally:
                reset_secret_scope(secret_token)
                reset_hermes_home_override(home_token)
    finally:
        set_multiplex_active(previous_multiplex)
    assert rendered[0] == rendered[2]
    assert rendered[0] != rendered[1]
