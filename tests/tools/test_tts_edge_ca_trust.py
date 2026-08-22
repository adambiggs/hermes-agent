"""Tests for Edge TTS CA trust configuration.

``edge-tts`` builds its SSL context from certifi's bundled roots, so it does
not honour ``SSL_CERT_FILE`` / ``REQUESTS_CA_BUNDLE`` the way the rest of
Hermes does. On a host behind a TLS-inspecting egress proxy that leaves Edge
TTS as the only outbound path failing with CERTIFICATE_VERIFY_FAILED. These
tests pin the additive trust fix: the configured bundle's roots are loaded
into the contexts edge-tts already built, and verification is never relaxed.
"""

import ssl
import sys
import types

import certifi
import pytest

from tools.tts_tool import (
    _apply_edge_tts_ca_trust,
    _configured_ca_bundle,
    _import_edge_tts,
)


@pytest.fixture(autouse=True)
def _clear_ca_env(monkeypatch):
    for var in ("HERMES_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE",
                "CURL_CA_BUNDLE"):
        monkeypatch.delenv(var, raising=False)


def _fake_edge_tts(*context_holders: str):
    """Build a stand-in edge_tts package whose submodules hold SSL contexts.

    Uses real ``ssl.SSLContext`` objects with an empty trust store, so a test
    can assert on the certificates actually loaded rather than on a call.
    """
    pkg = types.ModuleType("edge_tts")
    for name in context_holders:
        sub = types.ModuleType(f"edge_tts.{name}")
        sub._SSL_CTX = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        setattr(pkg, name, sub)
    return pkg


# ---------------------------------------------------------------------------
# Bundle resolution
# ---------------------------------------------------------------------------

class TestConfiguredCABundle:
    def test_returns_none_when_nothing_configured(self):
        assert _configured_ca_bundle() is None

    def test_prefers_hermes_ca_bundle(self, monkeypatch, tmp_path):
        preferred = tmp_path / "hermes.pem"
        preferred.write_text("")
        other = tmp_path / "other.pem"
        other.write_text("")
        monkeypatch.setenv("HERMES_CA_BUNDLE", str(preferred))
        monkeypatch.setenv("SSL_CERT_FILE", str(other))

        assert _configured_ca_bundle() == str(preferred)

    def test_falls_through_to_ssl_cert_file(self, monkeypatch, tmp_path):
        bundle = tmp_path / "system.pem"
        bundle.write_text("")
        monkeypatch.setenv("SSL_CERT_FILE", str(bundle))

        assert _configured_ca_bundle() == str(bundle)

    def test_skips_env_var_pointing_at_a_missing_file(self, monkeypatch, tmp_path):
        bundle = tmp_path / "system.pem"
        bundle.write_text("")
        monkeypatch.setenv("HERMES_CA_BUNDLE", str(tmp_path / "gone.pem"))
        monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(bundle))

        assert _configured_ca_bundle() == str(bundle)


# ---------------------------------------------------------------------------
# Applying the trust material
# ---------------------------------------------------------------------------

class TestApplyEdgeTTSCATrust:
    def test_loads_configured_roots_into_the_edge_tts_context(self, monkeypatch):
        monkeypatch.setenv("SSL_CERT_FILE", certifi.where())
        edge_tts = _fake_edge_tts("communicate", "voices")
        assert edge_tts.communicate._SSL_CTX.get_ca_certs() == []

        assert _apply_edge_tts_ca_trust(edge_tts) is True

        assert len(edge_tts.communicate._SSL_CTX.get_ca_certs()) > 0
        assert len(edge_tts.voices._SSL_CTX.get_ca_certs()) > 0

    def test_keeps_certificate_verification_on(self, monkeypatch):
        monkeypatch.setenv("SSL_CERT_FILE", certifi.where())
        edge_tts = _fake_edge_tts("communicate")
        ctx = edge_tts.communicate._SSL_CTX
        ctx.verify_mode = ssl.CERT_REQUIRED
        ctx.check_hostname = True

        _apply_edge_tts_ca_trust(edge_tts)

        assert ctx.verify_mode is ssl.CERT_REQUIRED
        assert ctx.check_hostname is True

    def test_no_configured_bundle_is_a_noop(self):
        edge_tts = _fake_edge_tts("communicate")

        assert _apply_edge_tts_ca_trust(edge_tts) is False
        assert edge_tts.communicate._SSL_CTX.get_ca_certs() == []

    def test_survives_a_version_without_a_module_level_context(self, monkeypatch):
        monkeypatch.setenv("SSL_CERT_FILE", certifi.where())
        edge_tts = types.ModuleType("edge_tts")
        edge_tts.communicate = types.ModuleType("edge_tts.communicate")

        assert _apply_edge_tts_ca_trust(edge_tts) is False

    def test_unreadable_bundle_does_not_raise(self, monkeypatch, tmp_path):
        junk = tmp_path / "not-a-bundle.pem"
        junk.write_text("this is not PEM\n")
        monkeypatch.setenv("SSL_CERT_FILE", str(junk))
        edge_tts = _fake_edge_tts("communicate")

        assert _apply_edge_tts_ca_trust(edge_tts) is False


# ---------------------------------------------------------------------------
# Wiring: the lazy import applies the trust material
# ---------------------------------------------------------------------------

class TestImportAppliesTrust:
    def test_import_edge_tts_applies_configured_roots(self, monkeypatch):
        monkeypatch.setenv("SSL_CERT_FILE", certifi.where())
        edge_tts = _fake_edge_tts("communicate")
        monkeypatch.setitem(sys.modules, "edge_tts", edge_tts)

        returned = _import_edge_tts()

        assert returned is edge_tts
        assert len(edge_tts.communicate._SSL_CTX.get_ca_certs()) > 0
