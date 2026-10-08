"""Execute the existing fill script against synthetic controls; never open a browser."""
import json
import subprocess

import pytest

from agent.vault_login_classifier import build_fill_js


@pytest.mark.parametrize("scenario,reason", [
    ("filled", None), ("target_missing", "target_missing"),
    ("password_type_changed", "password_type_changed"),
    ("write_exception", "write_exception"), ("value_empty", "value_empty"),
    ("option_not_found", "option_not_found"), ("origin_changed", "origin_changed"),
])
def test_fill_script_reports_fixed_reasons_without_values_and_keeps_origin_guard(scenario, reason):
    expression = build_fill_js([{"index": 0, "token": "country" if scenario == "option_not_found" else "current-password",
                                 "value": "synthetic-sensitive-value"}], "https://example.com", nonce="synthetic")
    setup = r'''const scenario = SCENARIO;
let writes = 0;
class HTMLInputElement {
  constructor() { this._value = ""; this.type = scenario === "password_type_changed" ? "text" : "password";
    this.tagName = scenario === "option_not_found" ? "SELECT" : "INPUT"; this.options = [];
    this.marker = scenario === "target_missing" ? null : "synthetic:0"; }
  focus() {}
  dispatchEvent() {}
  removeAttribute() { this.marker = null; }
}
Object.defineProperty(HTMLInputElement.prototype, "value", {
  get() { return this._value; },
  set(value) { writes += 1;
    if (scenario === "write_exception") throw Error("synthetic-sensitive-exception");
    this._value = scenario === "value_empty" ? "" : value; }
});
const element = new HTMLInputElement();
const document = {
  querySelector(selector) { return element.marker && selector === '[data-hermes-vault-slot="synthetic:0"]' ? element : null; },
  querySelectorAll() { return [element]; }
};
const window = { location: { origin: scenario === "origin_changed" ? "https://other.example" : "https://example.com" } };
class InputEvent {}
class Event {}
const result = JSON.parse(EXPRESSION);
console.log(JSON.stringify({ result, writes, marker_present: element.marker !== null }));
'''
    script = setup.replace("SCENARIO", json.dumps(scenario)).replace("EXPRESSION", expression)
    proc = subprocess.run(["node", "-"], input=script, text=True, capture_output=True, check=True, timeout=10)
    assert "synthetic-sensitive" not in proc.stdout
    out = json.loads(proc.stdout)
    if reason == "origin_changed":
        assert out["result"]["refused"] == reason
        assert out["writes"] == 0
        assert out["marker_present"] is True
    else:
        assert out["result"]["filled"] == (1 if reason is None else 0)
        assert out["result"]["unfilled_reasons"] == ([] if reason is None else [reason])
        assert out["marker_present"] is False
