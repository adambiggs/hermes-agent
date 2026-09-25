"""Delegated workdir instructions and context discovery use their own namespaces."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools.delegate_tool import _build_child_agent
from tools.terminal_scope import install_and_reset_profile_terminal_scope
from tools.terminal_tool import _get_env_config


@pytest.mark.parametrize("backend, mount", [("docker", True), ("local", False)])
def test_child_workspace_matches_terminal_and_retains_project_context(tmp_path, backend, mount):
    """Build a real child prompt from profile config; replace only the LLM constructor."""
    home = tmp_path / "profile"
    home.mkdir()
    workdir = tmp_path / "guest-state" / "workspace"
    workdir.mkdir(parents=True)
    project_instruction = "Preserve the widget transaction ordering."
    (workdir / "AGENTS.md").write_text(project_instruction, encoding="utf-8")
    (home / "config.yaml").write_text(yaml.safe_dump({"terminal": {
        "backend": backend, "cwd": str(workdir), "docker_mount_cwd_to_workspace": mount,
    }}), encoding="utf-8")
    parent = SimpleNamespace(
        cwd=str(workdir), model="test-model", provider="openrouter", api_key="test-key",
        base_url="https://openrouter.ai/api/v1", api_mode="chat_completions",
        enabled_toolsets=["terminal"], disabled_toolsets=[],
    )
    token = set_hermes_home_override(home)
    try:
        with install_and_reset_profile_terminal_scope(home), patch("run_agent.AIAgent") as child_constructor:
            config = _get_env_config()
            _build_child_agent(
                task_index=0, goal="Read the repository", context=None, toolsets=None,
                model=None, max_iterations=5, task_count=1, parent_agent=parent,
            )
            prompt = child_constructor.call_args.kwargs["ephemeral_system_prompt"]
            assert f"WORKSPACE PATH:\n{config['cwd']}\n" in prompt
            assert project_instruction in prompt
            if mount:
                assert config["host_cwd"] == str(workdir)
                assert config["cwd"] != str(workdir)
                assert f"WORKSPACE PATH:\n{workdir}\n" not in prompt
    finally:
        reset_hermes_home_override(token)
