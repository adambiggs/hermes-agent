"""Verify asynchronous acknowledgement without a browser or credential values."""
import json
import subprocess

import pytest

from agent.vault_login_classifier import build_fill_js


@pytest.mark.parametrize("scenario,filled,reason", [
    ("deferred_state", 1, None),
    ("two_controls", 2, None),
    ("origin_changed", None, "origin_changed"),
    ("target_replaced", 0, "target_missing"),
    ("target_detached", 0, "target_missing"),
    ("target_unstamped", 0, "target_missing"),
    ("password_type_changed", 0, "password_type_changed"),
    ("remains_empty", 0, "value_empty"),
    ("getter_throws", 0, "write_exception"),
    ("select_empty_value", 1, None),
])
def test_acknowledgement_yields_once_and_rechecks_origin_target_and_type(scenario, filled, reason):
    fills = [{"index": 0, "token": "country" if scenario == "select_empty_value" else "current-password",
              "value": "Synthetic country" if scenario == "select_empty_value" else "synthetic-sensitive-value"}]
    if scenario == "two_controls":
        fills.append({"index": 1, "token": "one-time-code", "value": "synthetic-sensitive-value"})
    expression = build_fill_js(fills, "https://example.com", nonce="fixture")
    setup = r'''const scenario = SCENARIO;
const timers = [];
let writes = 0, events = 0;
function setTimeout(callback, delay) { if (delay !== 0) throw Error("Unexpected delay"); timers.push(callback); }
class HTMLInputElement {
  constructor(index) { this._value = ""; this.type = "password"; this.tagName = "INPUT";
    this.marker = "fixture:" + index; this.isConnected = true; }
  focus() {}
  dispatchEvent(event) { events += 1;
    if (event.type === "input") { const proposed = this._value; this._value = "";
      Promise.resolve().then(() => { if (scenario !== "remains_empty") this._value = proposed; }); }
  }
  getAttribute() { return this.marker; }
  removeAttribute() { this.marker = null; }
}
Object.defineProperty(HTMLInputElement.prototype, "value", {
  configurable: true, get() { return this._value; }, set(value) { writes += 1; this._value = value; }
});
class Event { constructor(type) { this.type = type; } }
class InputEvent extends Event {}
const elements = [new HTMLInputElement(0)];
if (scenario === "two_controls") elements.push(new HTMLInputElement(1));
if (scenario === "select_empty_value") {
  elements[0].tagName = "SELECT";
  elements[0].options = [{ value: "", textContent: "Synthetic country" }];
}
const unrelated = new HTMLInputElement(9); unrelated.marker = "another-request:9";
const originalElements = [...elements];
const window = { location: { origin: "https://example.com" } };
const document = {
  querySelector(selector) { return elements.find(e => e.isConnected && e.marker && selector === '[data-hermes-vault-slot="' + e.marker + '"]') || null; },
  querySelectorAll(selector) { return [...elements, unrelated].filter(e => e.isConnected && e.marker && (selector === "[data-hermes-vault-slot]" || e.marker.startsWith("fixture:"))); }
};
(async () => {
  const pending = EXPRESSION;
  const scheduled = timers.length;
  // Explicit barrier: the fixture's deferred update runs before resolving the requested yield.
  await Promise.resolve();
  if (scenario === "origin_changed") window.location.origin = "https://other.example";
  if (scenario === "target_replaced") elements[0] = new HTMLInputElement(0);
  if (scenario === "target_detached") elements[0].isConnected = false;
  if (scenario === "target_unstamped") elements[0].marker = null;
  if (scenario === "password_type_changed") elements[0].type = "text";
  if (scenario === "getter_throws") Object.defineProperty(elements[0], "value", { get() { throw Error("synthetic-sensitive-exception"); } });
  for (const callback of timers) callback();
  const result = JSON.parse(await pending);
  console.log(JSON.stringify({ result, scheduled, writes, events,
    owned_markers_removed: [...elements, ...originalElements].every(e => e.marker === null), unrelated_marker_preserved: unrelated.marker === "another-request:9" }));
})().catch(error => { console.error(error); process.exit(1); });
'''
    script = setup.replace("SCENARIO", json.dumps(scenario)).replace("EXPRESSION", expression)
    proc = subprocess.run(["node", "-"], input=script, text=True, capture_output=True, check=True, timeout=10)
    assert "synthetic-sensitive" not in proc.stdout
    out = json.loads(proc.stdout)
    assert out["scheduled"] == 1
    assert out["writes"] == len(fills)
    assert out["events"] == (1 if scenario == "select_empty_value" else 2 * len(fills))
    assert out["owned_markers_removed"] is True
    assert out["unrelated_marker_preserved"] is True
    if filled is None:
        assert out["result"]["refused"] == reason
        assert "filled" not in out["result"]
    else:
        assert out["result"] == {"filled": filled, "unfilled_reasons": [] if reason is None else [reason]}
