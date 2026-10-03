"""A cron agent's system prompt carries its delivery channel's hint.

Cron agents run as platform ``cron`` but the final response lands on the job's ``deliver``
channel. Without the destination hint the model never learns that MEDIA: tags become native
attachments there, and ``platform_hints.<channel>.append`` never reached scheduled jobs
(community report: Slack-delivered cron jobs stopped sending attachments).
"""

import types

import pytest

from agent.prompt_builder import PLATFORM_HINTS
from agent.system_prompt import platform_hint
from gateway.session_context import _VAR_MAP

EXTRA = "Always attach the report as a .pdf."


def _agent(platform, overrides=None):
    return types.SimpleNamespace(platform=platform, _platform_hint_overrides=overrides or {})


@pytest.fixture
def deliver_to(monkeypatch):
    var = _VAR_MAP["HERMES_CRON_AUTO_DELIVER_PLATFORM"]
    token = var.set("slack")
    monkeypatch.delenv("HERMES_DESKTOP_TERMINAL", raising=False)
    yield
    var.reset(token)


def test_cron_agent_gets_delivery_channel_hint_with_its_override(deliver_to):
    hint = platform_hint(_agent("cron", {"slack": {"append": EXTRA}}))
    assert hint.startswith(PLATFORM_HINTS["cron"])
    assert PLATFORM_HINTS["slack"] in hint
    assert EXTRA in hint
    # The cron hint's own override channel is untouched by the destination's.
    assert platform_hint(_agent("cron", {"cron": {"replace": "X"}})).startswith("X\n\n")


def test_delivery_target_only_applies_to_cron_agents(deliver_to):
    assert platform_hint(_agent("telegram")) == PLATFORM_HINTS["telegram"]
    _VAR_MAP["HERMES_CRON_AUTO_DELIVER_PLATFORM"].set("")
    assert platform_hint(_agent("cron")) == PLATFORM_HINTS["cron"]



@pytest.mark.parametrize("platform", ["cli", "tui"])
def test_local_cron_note_only_where_cron_scheduling_is_available(platform, monkeypatch):
    from agent.prompt_builder import _LOCAL_CRON_DELIVERY_NOTE

    monkeypatch.delenv("HERMES_DESKTOP_TERMINAL", raising=False)
    for var in ("HERMES_INTERACTIVE", "HERMES_GATEWAY_SESSION", "HERMES_EXEC_ASK"):
        monkeypatch.delenv(var, raising=False)
    hint = platform_hint(_agent(platform))
    assert "cronjob_manage" not in hint
    assert hint.endswith(".")
    appended = platform_hint(_agent(platform, {platform: {"append": EXTRA}}))
    assert "cronjob_manage" not in appended and appended.endswith(f".\n\n{EXTRA}")

    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    assert _LOCAL_CRON_DELIVERY_NOTE in platform_hint(_agent(platform))
