"""Tests for security.allowed_private_networks CIDR allowlist in url_safety.

Some environments run a local DNS resolver that maps public domains into the
RFC 2544 benchmark range 198.18.0.0/15. The allowlist lets a user exempt only
that range while keeping SSRF blocking in force everywhere else, and crucially
without ever exposing the always-blocked cloud-metadata floor.
"""

import socket

import pytest

import tools.url_safety as url_safety


# IP each test hostname resolves to (no real DNS).
_FAKE_DNS = {
    "example.com": "198.18.15.159",         # mapped into benchmark range
    "router.lan": "192.168.1.1",            # real LAN
    "public.example": "93.184.216.34",      # ordinary public IP
    "metadata.evil": "169.254.169.254",     # cloud metadata (always-blocked floor)
    "mapped.test": "::ffff:198.18.15.159",  # IPv4-mapped IPv6 of benchmark IP
    "sinkholed.example": "10.231.1.1",      # transparent egress proxy / DNS sinkhole
    "mapped.proxy": "::ffff:10.231.1.1",    # IPv4-mapped form of the same proxy
    "v6host.example": "fd00::80",           # bare IPv6 ending in digits after ':'
}


@pytest.fixture
def patched(monkeypatch):
    """Patch DNS + config and clear the module's cache around each test."""
    def fake_getaddrinfo(host, *args, **kwargs):
        ip = _FAKE_DNS[host]
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (ip, 0))]

    monkeypatch.setattr(url_safety.socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.delenv("HERMES_ALLOW_PRIVATE_URLS", raising=False)

    def set_config(cfg):
        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config", lambda: cfg, raising=False
        )
        url_safety._reset_allow_private_cache()

    set_config({})
    yield set_config
    url_safety._reset_allow_private_cache()


def test_benchmark_ip_blocked_without_allowlist(patched):
    assert url_safety.is_safe_url("https://example.com/some/path") is False


def test_public_always_allowed(patched):
    assert url_safety.is_safe_url("https://public.example") is True


def test_benchmark_ip_allowed_with_allowlist(patched):
    patched({"security": {"allowed_private_networks": ["198.18.0.0/15"]}})
    assert url_safety.is_safe_url("https://example.com/some/path") is True


def test_real_lan_still_blocked_with_allowlist(patched):
    patched({"security": {"allowed_private_networks": ["198.18.0.0/15"]}})
    assert url_safety.is_safe_url("http://router.lan") is False


def test_metadata_floor_cannot_be_allowlisted(patched):
    # Even explicitly listing the link-local range must not expose metadata.
    patched({"security": {"allowed_private_networks": ["169.254.0.0/16", "198.18.0.0/15"]}})
    assert url_safety.is_safe_url("http://metadata.evil") is False


def test_single_string_cidr_accepted(patched):
    patched({"security": {"allowed_private_networks": "198.18.0.0/15"}})
    assert url_safety.is_safe_url("https://example.com/q") is True


def test_invalid_entry_skipped_valid_applies(patched):
    patched({"security": {"allowed_private_networks": ["not-a-cidr", "198.18.0.0/15"]}})
    assert url_safety.is_safe_url("https://example.com/q") is True
    assert url_safety.is_safe_url("http://router.lan") is False


def test_ipv4_mapped_ipv6_covered_by_v4_cidr(patched):
    patched({"security": {"allowed_private_networks": ["198.18.0.0/15"]}})
    assert url_safety.is_safe_url("https://mapped.test") is True


def test_non_matching_allowlist_leaves_benchmark_ip_blocked(patched):
    patched({"security": {"allowed_private_networks": ["10.0.0.0/8"]}})
    assert url_safety.is_safe_url("https://example.com/q") is False


# ---------------------------------------------------------------------------
# Optional per-entry port scoping: "<cidr>:<port>[,<port>...]"
# ---------------------------------------------------------------------------
# A transparent egress proxy usually shares its host with unrelated internal
# services. Exempting the address wholesale would reach those too, so an entry
# may narrow itself to the proxy's own ports. Bare CIDRs keep their existing
# all-ports meaning, so configs written against the plain allowlist are
# unaffected.


def test_port_scoped_entry_allows_listed_https_port(patched):
    patched({"security": {"allowed_private_networks": ["10.231.1.1/32:80,443"]}})
    assert url_safety.is_safe_url("https://sinkholed.example/pricing") is True


def test_port_scoped_entry_allows_listed_http_port(patched):
    patched({"security": {"allowed_private_networks": ["10.231.1.1/32:80,443"]}})
    assert url_safety.is_safe_url("http://sinkholed.example/pricing") is True


def test_port_scoped_entry_blocks_unlisted_port(patched):
    """Co-located services on the proxy host must stay unreachable."""
    patched({"security": {"allowed_private_networks": ["10.231.1.1/32:80,443"]}})
    assert url_safety.is_safe_url("http://sinkholed.example:7444/mail") is False


def test_explicit_listed_port_in_url_allowed(patched):
    patched({"security": {"allowed_private_networks": ["10.231.1.1/32:8080"]}})
    assert url_safety.is_safe_url("http://sinkholed.example:8080/") is True
    assert url_safety.is_safe_url("http://sinkholed.example/") is False


def test_bare_cidr_still_means_all_ports(patched):
    """Backward compatibility: an entry with no port scope is unrestricted."""
    patched({"security": {"allowed_private_networks": ["10.231.1.1/32"]}})
    assert url_safety.is_safe_url("http://sinkholed.example:7444/mail") is True


def test_port_scope_applies_to_ipv4_mapped_ipv6(patched):
    patched({"security": {"allowed_private_networks": ["10.231.1.1/32:443"]}})
    assert url_safety.is_safe_url("https://mapped.proxy/") is True
    assert url_safety.is_safe_url("http://mapped.proxy:7444/") is False


def test_port_scope_cannot_unblock_metadata_floor(patched):
    """The always-blocked floor is checked first and is not port-scoped."""
    patched({"security": {"allowed_private_networks": ["169.254.0.0/16:80,443"]}})
    assert url_safety.is_safe_url("http://metadata.evil/latest/meta-data/") is False


def test_bare_ipv6_address_not_mistaken_for_port_suffix(patched):
    """fd00::80 is an address, not fd00: scoped to port 80."""
    patched({"security": {"allowed_private_networks": ["fd00::80"]}})
    assert url_safety.is_safe_url("http://v6host.example/") is True


def test_invalid_port_scope_entry_skipped(patched):
    patched({"security": {"allowed_private_networks": ["10.231.1.1/32:notaport", "198.18.0.0/15"]}})
    assert url_safety.is_safe_url("https://sinkholed.example/") is False
    assert url_safety.is_safe_url("https://example.com/q") is True


def test_out_of_range_port_scope_entry_skipped(patched):
    patched({"security": {"allowed_private_networks": ["10.231.1.1/32:70000"]}})
    assert url_safety.is_safe_url("https://sinkholed.example/") is False


# ---------------------------------------------------------------------------
# Real config.yaml resolution
# ---------------------------------------------------------------------------
# The tests above inject config by replacing read_raw_config, which does not
# prove the setting survives the actual config loader. This one writes a real
# config.yaml under a temp HERMES_HOME and lets Hermes resolve it, with DNS
# still mocked.


def test_allowlist_resolves_from_real_config_file(tmp_path, monkeypatch):
    import importlib
    import yaml

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_ALLOW_PRIVATE_URLS", raising=False)
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(
            {"security": {"allowed_private_networks": ["10.231.1.1/32:80,443"]}}
        )
    )

    import hermes_cli.config as hermes_config
    importlib.reload(hermes_config)

    def fake_getaddrinfo(host, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.231.1.1", 0))]

    monkeypatch.setattr(url_safety.socket, "getaddrinfo", fake_getaddrinfo)
    url_safety._reset_allow_private_cache()
    try:
        assert url_safety.is_safe_url("https://sinkholed.example/") is True
        assert url_safety.is_safe_url("http://sinkholed.example:7444/") is False
    finally:
        url_safety._reset_allow_private_cache()
