"""Per-task provider routing and operator-owned delegation tool ceilings."""

from __future__ import annotations

from hermes_cli.config import get_compatible_custom_providers, is_provider_enabled, load_config_readonly
from hermes_cli.providers import custom_provider_aliases, custom_provider_slug
from hermes_cli.route_identity import normalize_route_base_url
from hermes_cli.runtime_provider_custom import _entry_url
from tools.delegate_tool_toolsets import _strip_blocked_tools
from toolsets import resolve_toolset


def _provider_entries(config, *, include_disabled=False):
    """Match runtime-provider precedence, including the legacy compatibility view."""
    providers = config.get("providers") or {}
    if isinstance(providers, dict):
        for key, entry in providers.items():
            if isinstance(entry, dict) and (include_disabled or is_provider_enabled(entry)) and _entry_url(entry):
                yield str(key), str(entry.get("name") or key), entry
    for entry in get_compatible_custom_providers(config):
        yield str(entry.get("provider_key") or ""), str(entry["name"]), entry


def _normalize_cap(raw, *, source):
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list) or any(not isinstance(t, str) or not t.strip() for t in raw):
        raise ValueError(f"{source} must be a list of toolset names; refusing uncapped delegation.")
    return _strip_blocked_tools([t.strip() for t in raw])


def _provider_caps(provider, base_url, config):
    requested = str(provider or "").strip().lower().replace(" ", "-")
    # Provider resolution strips all trailing slashes before constructing the transport URL.
    endpoint = normalize_route_base_url(str(base_url or "").strip().rstrip("/"))
    # Disabling a named route does not remove its endpoint ceiling: a generic alias can still reach it.
    for key, name, entry in _provider_entries(config, include_disabled=True):
        same_endpoint = endpoint and endpoint == normalize_route_base_url(str(_entry_url(entry)).strip().rstrip("/"))
        if requested in custom_provider_aliases(name, key) or same_endpoint:
            yield _normalize_cap(entry.get("delegation_toolsets"), source=f"provider {key or name}.delegation_toolsets")


def _fallback_cap_route(route, config):
    """Resolve implicit endpoints before exposing tools a fallback would retain."""
    if route.get("base_url") or not any(
        entry.get("delegation_toolsets") is not None for _, _, entry in _provider_entries(config, include_disabled=True)
    ):
        return route
    from hermes_cli.fallback_config import resolve_entry_api_key
    from hermes_cli.runtime_provider import resolve_runtime_provider
    try:
        resolved = resolve_runtime_provider(
            requested=route.get("provider"), target_model=route.get("model"),
            explicit_api_key=resolve_entry_api_key(route),
        )
    except Exception as exc:
        raise ValueError("Cannot verify delegation fallback endpoint against provider tool ceilings; refusing delegation.") from exc
    if resolved.get("provider") == "custom" and not resolved.get("base_url"):
        raise ValueError("Custom delegation fallback has no verifiable endpoint; refusing uncapped delegation.")
    return {**route, "base_url": resolved.get("base_url")}


def _apply_toolset_cap(toolsets, cap):
    """Intersect resolved surfaces so a parent's composite bundle can narrow to file/web."""
    if cap is None:
        return toolsets
    inherited = {tool for name in toolsets for tool in resolve_toolset(name)}
    allowed = {tool for name in cap for tool in resolve_toolset(name)}
    ceiling = inherited & allowed
    return [
        name for name in dict.fromkeys([*toolsets, *cap])
        if (resolved := set(resolve_toolset(name))) and resolved <= ceiling
    ]


def _cap_child_toolsets(toolsets, runtime, delegation_cfg, routing_cfg=None):
    """Run after role grants; neither model arguments nor nested delegates can widen this ceiling."""
    config = load_config_readonly()
    toolsets = _apply_toolset_cap(toolsets, _normalize_cap(
        delegation_cfg.get("delegation_toolsets"), source="delegation.delegation_toolsets"))
    # A direct base_url selects a generic wire provider; its configured provider still owns the ceiling.
    declared_route = delegation_cfg if routing_cfg is None else routing_cfg
    # Fallbacks keep this child's tools; honor their caps before any request can switch routes.
    routes = [declared_route, runtime, *(
        _fallback_cap_route(route, config) for route in runtime.get("fallback_model") or []
    )]
    for route in routes:
        provider = route.get("requested_provider") or route.get("provider")
        for cap in _provider_caps(provider, route.get("base_url"), config):
            toolsets = _apply_toolset_cap(toolsets, cap)
    return toolsets


def _resolve_task_routes(task_list, cfg, parent_agent, *, provider=None, model=None):
    """Resolve the entire batch before spawning; reuse bundles only within this call/profile."""
    from tools.delegate_tool import _resolve_delegation_credentials

    cache = {}
    routes = []
    for task in task_list:
        selected = []
        for field, default in (("provider", provider), ("model", model)):
            value = task.get(field) or default
            if value is not None and not isinstance(value, str):
                raise ValueError(f"Task {field} must be a string.")
            selected.append((value or "").strip())
        requested_provider, requested_model = selected
        key = tuple(selected)
        if key not in cache:
            routing_cfg = dict(cfg)
            if requested_provider:
                # A model-selected provider owns its endpoint and transport; never inherit
                # direct-endpoint credentials or an operator's ACP command from another route.
                for field in ("base_url", "api_key", "api_mode", "command", "args"):
                    routing_cfg.pop(field, None)
                routing_cfg["provider"] = requested_provider
            if requested_model:
                routing_cfg["model"] = requested_model
            cache[key] = (_resolve_delegation_credentials(routing_cfg, parent_agent), routing_cfg)
        routes.append(cache[key])
    return routes


def _build_provider_param_description():
    config = load_config_readonly()
    providers = []
    seen = set()
    for key, name, entry in _provider_entries(config):
        identity = custom_provider_slug(name, key)
        if identity in seen:
            continue
        seen.add(identity)
        label = repr(identity)
        if entry.get("delegation_toolsets") is not None:
            cap = _normalize_cap(entry["delegation_toolsets"], source=f"provider {identity}.delegation_toolsets")
            label += " (restricted to toolsets: " + (", ".join(cap) or "none") + ")"
        providers.append(label)
    description = "Optional provider for this child. Omit to inherit the delegation default. Operator toolset ceilings always apply."
    if providers:
        description += " Configured providers: " + "; ".join(providers) + "."
    return description
