"""Tests for per-task provider/model routing on delegate_task.

Covers the three new units:
  * `_apply_toolset_cap`        — the model-unwidenable provider ceiling
  * `_get_provider_toolset_cap` — reads providers.<name>.delegation_toolsets
  * `_resolve_task_credentials` — per-task provider/model override + memoisation

and the schema surface (top-level + per-task `provider`/`model`).

The cap is the load-bearing guardrail: it lets an operator route narrow grunt
work to a weaker/cheaper local model while guaranteeing that model can never
receive a toolset (e.g. `terminal`) the operator didn't sanction — regardless
of what the orchestrating model requests.
"""

import json
import threading
from unittest.mock import MagicMock, patch

from tools.delegate_tool import (
    DELEGATE_TASK_SCHEMA,
    _apply_toolset_cap,
    _build_child_agent,
    _get_provider_toolset_cap,
    _normalize_cap,
    _resolve_task_credentials,
    delegate_task,
)


def _make_capped_parent(enabled):
    """Mock parent with an explicit enabled_toolsets set for clean intersection."""
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "***"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.enabled_toolsets = list(enabled)
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent.openrouter_min_coding_score = None
    parent._session_db = None
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    return parent


class TestApplyToolsetCap:
    """The final intersection the orchestrating model cannot widen."""

    def test_none_cap_is_passthrough(self):
        """No provider cap configured → toolsets unchanged."""
        ts = ["terminal", "file", "web"]
        assert _apply_toolset_cap(ts, None) == ts

    def test_cap_drops_disallowed_toolsets(self):
        """Orchestrator-requested toolsets outside the cap are stripped."""
        capped = _apply_toolset_cap(["terminal", "file", "web"], ["file", "search"])
        assert capped == ["file"]
        assert "terminal" not in capped
        assert "web" not in capped

    def test_cap_is_a_ceiling_not_a_fixed_set(self):
        """A child requesting a subset of the cap keeps exactly that subset."""
        assert _apply_toolset_cap(["file"], ["file", "search"]) == ["file"]

    def test_cap_preserves_order(self):
        assert _apply_toolset_cap(
            ["web", "file", "search"], ["search", "file", "web"]
        ) == [
            "web",
            "file",
            "search",
        ]

    def test_empty_cap_denies_everything(self):
        """An explicit empty allowlist means the child gets no toolsets."""
        assert _apply_toolset_cap(["file", "terminal"], []) == []


class TestGetProviderToolsetCap:
    """Reading providers.<name>.delegation_toolsets from config."""

    def _cfg(self, providers):
        return {"providers": providers}

    def test_returns_cap_for_named_provider(self):
        cfg = self._cfg({
            "gemma-local": {
                "base_url": "http://x:1234/v1",
                "delegation_toolsets": ["file", "search"],
            }
        })
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert _get_provider_toolset_cap("gemma-local") == ["file", "search"]

    def test_no_cap_key_returns_none(self):
        cfg = self._cfg({"gemma-local": {"base_url": "http://x:1234/v1"}})
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert _get_provider_toolset_cap("gemma-local") is None

    def test_unknown_provider_returns_none(self):
        cfg = self._cfg({"gemma-local": {"delegation_toolsets": ["file"]}})
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert _get_provider_toolset_cap("something-else") is None

    def test_none_provider_returns_none(self):
        assert _get_provider_toolset_cap(None) is None

    def test_cap_strips_always_blocked_toolsets(self):
        """A misconfigured allowlist cannot re-introduce a blocked toolset."""
        cfg = self._cfg({
            "weak": {"delegation_toolsets": ["file", "memory", "delegation"]}
        })
        with patch("hermes_cli.config.load_config", return_value=cfg):
            cap = _get_provider_toolset_cap("weak")
        assert "memory" not in cap
        assert "delegation" not in cap
        assert "file" in cap

    def test_scalar_cap_is_coerced_to_list(self):
        cfg = self._cfg({"weak": {"delegation_toolsets": "file"}})
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert _get_provider_toolset_cap("weak") == ["file"]

    def test_matches_by_display_name(self):
        cfg = self._cfg({
            "ep1": {"name": "Gemma Local", "delegation_toolsets": ["file"]}
        })
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert _get_provider_toolset_cap("Gemma Local") == ["file"]

    def test_cap_found_on_legacy_custom_providers_list(self):
        """The cap is honoured on legacy custom_providers entries too (the
        format the VM actually uses), not just the new providers: dict."""
        cfg = {
            "providers": {},
            "custom_providers": [
                {
                    "name": "lmstudio-mac",
                    "base_url": "http://192.168.1.145:1234/v1",
                    "delegation_toolsets": ["file", "search"],
                }
            ],
        }
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert _get_provider_toolset_cap("lmstudio-mac") == ["file", "search"]

    def test_custom_providers_entry_without_cap_returns_none(self):
        cfg = {
            "custom_providers": [
                {"name": "lmstudio-mac", "base_url": "http://x:1234/v1"}
            ]
        }
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert _get_provider_toolset_cap("lmstudio-mac") is None


class TestNormalizeCap:
    """Coercion + always-blocked stripping shared by named and global caps."""

    def test_none_is_none(self):
        assert _normalize_cap(None, source="x") is None

    def test_scalar_wrapped(self):
        assert _normalize_cap("file", source="x") == ["file"]

    def test_non_list_rejected(self):
        assert _normalize_cap(123, source="x") is None

    def test_all_blocked_resolves_to_empty(self):
        """A cap of only always-blocked toolsets denies everything (not None)."""
        assert _normalize_cap(["memory", "delegation"], source="x") == []

    def test_blanks_dropped(self):
        assert _normalize_cap(["file", "  ", ""], source="x") == ["file"]


class TestResolveTaskCredentials:
    """Per-task provider/model override resolution + memoisation."""

    BASE = {
        "model": "gpt-5.5",
        "provider": None,
        "base_url": None,
        "api_key": None,
        "api_mode": None,
    }

    def test_no_override_returns_base_creds(self):
        cache: dict = {}
        out = _resolve_task_credentials(
            {},
            object(),
            provider_override=None,
            model_override=None,
            base_creds=self.BASE,
            cache=cache,
        )
        assert out is self.BASE
        assert cache == {}

    def test_provider_override_overlays_and_clears_direct_endpoint(self):
        """A per-task provider forces the provider-resolution path."""
        cfg = {
            "provider": "",
            "base_url": "http://global:1/v1",
            "api_key": "g",
            "model": "gpt-5.5",
        }
        cache: dict = {}
        with patch(
            "tools.delegate_tool._resolve_delegation_credentials",
            return_value={"model": "gemma", "provider": "custom"},
        ) as m:
            _resolve_task_credentials(
                cfg,
                object(),
                provider_override="gemma-local",
                model_override=None,
                base_creds=self.BASE,
                cache=cache,
            )
        overlay = m.call_args[0][0]
        assert overlay["provider"] == "gemma-local"
        # global direct-endpoint fields cleared so the provider name wins
        assert overlay["base_url"] == ""
        assert overlay["api_key"] == ""

    def test_model_only_override_keeps_global_provider(self):
        cfg = {"provider": "openrouter", "model": "gpt-5.5"}
        cache: dict = {}
        with patch(
            "tools.delegate_tool._resolve_delegation_credentials",
            return_value={"model": "cheap"},
        ) as m:
            _resolve_task_credentials(
                cfg,
                object(),
                provider_override=None,
                model_override="cheap-model",
                base_creds=self.BASE,
                cache=cache,
            )
        overlay = m.call_args[0][0]
        assert overlay["model"] == "cheap-model"
        assert overlay["provider"] == "openrouter"  # untouched
        assert "base_url" not in overlay or overlay["base_url"] != ""

    def test_result_is_memoised_by_provider_model(self):
        cache: dict = {}
        with patch(
            "tools.delegate_tool._resolve_delegation_credentials",
            return_value={"model": "gemma"},
        ) as m:
            for _ in range(3):
                _resolve_task_credentials(
                    {},
                    object(),
                    provider_override="gemma-local",
                    model_override=None,
                    base_creds=self.BASE,
                    cache=cache,
                )
        assert m.call_count == 1  # resolved once, served from cache after


class TestSchemaSurface:
    """The model must see provider/model at both levels."""

    def test_top_level_provider_and_model_present(self):
        props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
        assert "provider" in props
        assert "model" in props

    def test_per_task_provider_and_model_present(self):
        task_props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"][
            "properties"
        ]
        assert "provider" in task_props
        assert "model" in task_props


class TestCapWiringThroughBuildChildAgent:
    """The cap must actually reach the constructed child, not just the helper."""

    def test_cap_strips_terminal_even_when_orchestrator_requests_it(self):
        parent = _make_capped_parent(["terminal", "file", "web"])
        with patch("run_agent.AIAgent") as MockAgent:
            _build_child_agent(
                task_index=0,
                goal="grunt work",
                context=None,
                toolsets=[
                    "terminal",
                    "file",
                    "web",
                ],  # orchestrator asks for everything
                model=None,
                max_iterations=10,
                task_count=1,
                parent_agent=parent,
                toolset_cap=["file"],  # operator pinned to file only
            )
            enabled = MockAgent.call_args.kwargs["enabled_toolsets"]
        assert enabled == ["file"]
        assert "terminal" not in enabled
        assert "web" not in enabled

    def test_no_cap_preserves_requested_toolsets(self):
        parent = _make_capped_parent(["terminal", "file", "web"])
        with patch("run_agent.AIAgent") as MockAgent:
            _build_child_agent(
                task_index=0,
                goal="full access",
                context=None,
                toolsets=["terminal", "file"],
                model=None,
                max_iterations=10,
                task_count=1,
                parent_agent=parent,
                toolset_cap=None,
            )
            enabled = MockAgent.call_args.kwargs["enabled_toolsets"]
        assert sorted(enabled) == ["file", "terminal"]


# Gemma-local provider entry the orchestrator routes grunt work to: a weak local
# endpoint hard-capped to file-only, no terminal.
_GEMMA_CREDS = {
    "model": "gemma-4-e4b",
    "provider": "custom",
    "base_url": "http://mac.local:1234/v1",
    "api_key": "lm-studio",
    "api_mode": "chat_completions",
}
_INHERIT_CREDS = {
    "model": None,
    "provider": None,
    "base_url": None,
    "api_key": None,
    "api_mode": None,
}


def _fake_resolve(cfg, parent_agent):
    """Stand in for the runtime provider system: gemma-local → local creds,
    everything else → inherit-from-parent."""
    if cfg.get("provider") == "gemma-local":
        return dict(_GEMMA_CREDS)
    return dict(_INHERIT_CREDS)


class TestDelegateTaskEndToEndRouting:
    """Behavioral test of the full delegate_task() seam: a per-task provider
    routes the child's credentials AND the operator cap fences its toolsets —
    all the way through to the constructed AIAgent."""

    _DEFAULT_CFG = {
        "providers": {
            "gemma-local": {
                "base_url": "http://mac.local:1234/v1",
                "default_model": "gemma-4-e4b",
                "delegation_toolsets": ["file"],
            }
        }
    }

    def _run(
        self, *, cfg=None, resolve=_fake_resolve, delegation_cfg=None, **delegate_kwargs
    ):
        parent = _make_capped_parent(["terminal", "file", "web"])
        cfg = self._DEFAULT_CFG if cfg is None else cfg
        with (
            patch("run_agent.AIAgent") as MockAgent,
            patch(
                "tools.delegate_tool._resolve_delegation_credentials",
                side_effect=resolve,
            ) as resolve_mock,
            patch("hermes_cli.config.load_config", return_value=cfg),
            # delegate_task reads its own delegation block via _load_config();
            # pin it (default empty) so the suite is hermetic and the global
            # delegation.delegation_toolsets cap is controllable.
            patch(
                "tools.delegate_tool._load_config", return_value=delegation_cfg or {}
            ),
        ):
            child = MagicMock()
            child.run_conversation.return_value = {
                "final_response": "done",
                "completed": True,
                "api_calls": 1,
            }
            MockAgent.return_value = child
            result = json.loads(delegate_task(parent_agent=parent, **delegate_kwargs))
            return result, MockAgent, resolve_mock

    def test_provider_routes_credentials_and_cap_fences_toolsets(self):
        """gpt-5.5 orchestrator hands a file-only task to the local Gemma."""
        result, MockAgent, _ = self._run(
            goal="rename a variable across the file",
            toolsets=["terminal", "file", "web"],  # orchestrator over-asks
            provider="gemma-local",
        )
        kwargs = MockAgent.call_args.kwargs
        # Credentials routed to the local endpoint…
        assert kwargs["base_url"] == "http://mac.local:1234/v1"
        assert kwargs["model"] == "gemma-4-e4b"
        # …and the operator cap fenced the toolset despite the over-ask.
        assert kwargs["enabled_toolsets"] == ["file"]
        assert "terminal" not in kwargs["enabled_toolsets"]
        assert result["results"][0]["status"] != "error"

    def test_no_provider_inherits_parent_and_keeps_toolsets(self):
        """Without a provider override the child inherits the parent (no cap)."""
        _result, MockAgent, _ = self._run(
            goal="do it on my own model",
            toolsets=["terminal", "file"],
        )
        kwargs = MockAgent.call_args.kwargs
        # Inherited parent base_url (not the gemma endpoint)
        assert kwargs["base_url"] == "https://openrouter.ai/api/v1"
        assert sorted(kwargs["enabled_toolsets"]) == ["file", "terminal"]

    def test_per_task_provider_in_batch_overrides_top_level(self):
        """In a batch, a per-task provider beats the top-level default."""
        _result, MockAgent, _ = self._run(
            tasks=[
                {
                    "goal": "cheap grunt work",
                    "provider": "gemma-local",
                    "toolsets": ["terminal", "file"],
                },
            ],
        )
        kwargs = MockAgent.call_args.kwargs
        assert kwargs["base_url"] == "http://mac.local:1234/v1"
        assert kwargs["enabled_toolsets"] == ["file"]

    def test_global_delegation_cap_fences_base_url_target(self):
        """A delegation target wired via delegation.base_url (no provider name)
        is still capped by a global delegation.delegation_toolsets allowlist."""
        # cfg has NO providers entry — only the global cap. Credentials come
        # from the global delegation.base_url path (simulated via base creds).
        cfg = {"delegation_toolsets": ["file"]}

        def resolve(c, parent):
            return {
                "model": "local-model",
                "provider": "custom",
                "base_url": "http://lm.local:1234/v1",
                "api_key": "x",
                "api_mode": "chat_completions",
            }

        _result, MockAgent, _ = self._run(
            cfg=cfg,
            resolve=resolve,
            delegation_cfg={"delegation_toolsets": ["file"]},
            goal="grunt",
            toolsets=["terminal", "file", "web"],  # over-ask
        )
        # Global cap applied even though there is no providers.<name> to key on.
        assert MockAgent.call_args.kwargs["enabled_toolsets"] == ["file"]

    def test_bad_provider_raises_returns_task_error(self):
        """A per-task provider that fails to resolve surfaces as a tool error,
        and no child is constructed."""

        def resolve(c, parent):
            if c.get("provider") == "broken":
                raise ValueError("no API key for 'broken'")
            return dict(_INHERIT_CREDS)

        result, MockAgent, _ = self._run(
            resolve=resolve,
            goal="x",
            provider="broken",
        )
        assert "error" in result
        assert "Task 0" in result["error"]
        MockAgent.assert_not_called()

    def test_batch_same_provider_resolves_credentials_once(self):
        """Two tasks routed to the same provider hit the memoised cache: the
        credential resolver runs once for the base creds + once per distinct
        (provider, model) — not once per task."""
        _result, _MockAgent, resolve_mock = self._run(
            tasks=[
                {"goal": "a", "provider": "gemma-local", "toolsets": ["file"]},
                {"goal": "b", "provider": "gemma-local", "toolsets": ["file"]},
            ],
        )
        # 1 base-creds resolve + 1 for the shared gemma-local key (cached for #2).
        assert resolve_mock.call_count == 2
