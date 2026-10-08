"""The native vault result must distinguish invalid evaluation output from zero fills."""
import json

import pytest

from agent.vault_store import VaultStore
from tools import browser_vault_tool
from tools.registry import registry


@pytest.mark.parametrize("tool", ["browser_vault_fill", "browser_vault_enter_code"])
@pytest.mark.parametrize("payload,expected", [
    (None, "invalid_fill_result"),
    ("synthetic-sensitive-result", "invalid_fill_result"),
    ({}, "invalid_fill_result"),
    ({"filled": "synthetic-sensitive-result"}, "invalid_fill_result"),
    ({"filled": False}, "invalid_fill_result"),
    ({"filled": -1}, "invalid_fill_result"),
    ({"filled": 0.5}, "invalid_fill_result"),
    ({"filled": 0}, "zero_fill_count"),
    ({"filled": 0, "unfilled_reasons": ["target_missing", "synthetic-sensitive-result"]}, "zero_fill_count"),
    ({"filled": 0, "unfilled_reasons": "synthetic-sensitive-result"}, "zero_fill_count"),
    ({"filled": 1}, None),
    ({"refused": "origin_changed", "found": "https://other.example"}, "origin_changed"),
])
def test_fill_dispatch_preserves_result_validity_without_echoing_values(tmp_path, monkeypatch, tool, payload, expected):
    store = VaultStore(base_dir=tmp_path / "vault")
    meta = store.add_item("login", "Synthetic", {
        "identifier_type": "username", "identifier": "synthetic-user",
        "password": "synthetic-sensitive-password", "otp_secret": "JBSWY3DPEHPK3PXP",
    }, origin="https://example.com")
    monkeypatch.setattr("agent.vault_store.get_vault_store", lambda: store)
    monkeypatch.setattr(browser_vault_tool, "_focus_bound_origin", lambda *args, **kwargs: "https://example.com")
    controls = [{"index": 0, "type": "password", "autocomplete": "current-password"},
                {"index": 1, "type": "text", "autocomplete": "one-time-code"}]
    monkeypatch.setattr(browser_vault_tool, "_eval_js", lambda task_id, expression: {
        "success": True, "result": json.dumps(controls) if "querySelectorAll" in expression else "https://example.com/login"})
    monkeypatch.setattr(browser_vault_tool, "_eval_js_secret", lambda task_id, expression: {
        "success": True, "result": json.dumps(payload) if isinstance(payload, dict) else payload})
    raw = registry.dispatch(tool, {"handle": meta.id}, task_id="synthetic-result-test")
    out = json.loads(raw)
    assert out.get("error_type") == expected
    assert out["success"] is (expected is None)
    assert "synthetic-sensitive" not in raw
    if expected == "invalid_fill_result":
        assert "filled_fields" not in out
    elif expected == "zero_fill_count":
        assert out["filled_fields"] == 0
        assert out["unfilled_reasons"] == (["target_missing"] if isinstance(payload.get("unfilled_reasons"), list) else [])
    elif expected is None:
        assert out["filled_fields"] == payload["filled"]
