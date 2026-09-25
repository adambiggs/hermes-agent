"""A private-network exception grants only the configured address/port pairs."""

import asyncio
import socket
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import yaml

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import url_safety


@contextmanager
def profile(home):
    token = set_hermes_home_override(home)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def write_policy(home, entries):
    home.mkdir(exist_ok=True)
    (home / "config.yaml").write_text(yaml.safe_dump({"security": {
        "allow_private_urls": False, "allowed_private_networks": entries,
    }}), encoding="utf-8")


@pytest.mark.parametrize("entries, addresses, url, allowed", [
    (["10.231.1.1/32:80,443"], ["10.231.1.1"], "http://service.test/", True),
    (["10.231.1.1/32:80,443"], ["10.231.1.1"], "https://service.test/", True),
    (["10.231.1.1/32:80,443"], ["10.231.1.1"], "https://service.test:443/", True),
    (["10.231.1.1/32:80,443"], ["10.231.1.1"], "https://service.test:8080/", False),
    (["10.231.1.1/32:80,443"], ["10.231.1.2"], "https://service.test/", False),
    (["10.231.1.1/32:80,443"], ["10.231.1.1", "10.0.0.2"], "https://service.test/", False),
    (["10.231.1.1/32:443"], ["::ffff:10.231.1.1"], "https://service.test/", True),
    (["10.231.1.1/32:443"], ["::ffff:0:10.231.1.1"], "https://service.test/", True),
    (["10.231.1.1/32:443"], ["::ffff:10.231.1.1"], "http://service.test/", False),
    (["fd12::/64:443"], ["fd12::8"], "https://service.test/", True),
    (["fd12::80"], ["fd12::80"], "https://service.test:8080/", True),
    ("10.231.1.1/32:443", ["10.231.1.1"], "https://service.test/", True),
    (["10.231.1.1/32"], ["10.231.1.1"], "https://service.test:8080/", True),
    (["broken", "10.231.1.1/32:443"], ["10.231.1.1"], "https://service.test/", True),
    (["10.231.1.1/32:0", "10.231.1.1/32:65536", "10.231.1.1/32:443,", "10.231.1.1/32:no"], ["10.231.1.1"], "https://service.test/", False),
    ({"10.231.1.1/32": 443}, ["10.231.1.1"], "https://service.test/", False),
    ([], ["10.231.1.1"], "https://service.test/", False),
    ([], ["93.184.216.34"], "https://service.test/", True),
    (["0.0.0.0/0:80,443", "::/0:80,443"], ["169.254.169.254"], "https://service.test/", False),
    (["0.0.0.0/0:80,443", "::/0:80,443"], ["::ffff:169.254.170.2"], "http://service.test/", False),
    (["0.0.0.0/0:80,443", "::/0:80,443"], ["::ffff:0:100.100.100.200"], "http://service.test/", False),
    (["0.0.0.0/0:80,443", "::/0:80,443"], ["fd00:ec2::254"], "https://service.test/", False),
    (["0.0.0.0/0:80,443"], ["10.231.1.1"], "https://metadata.google.internal/", False),
])
def test_preflight_and_connect_share_address_port_policy(tmp_path, monkeypatch, entries, addresses, url, allowed):
    from urllib.parse import urlparse

    write_policy(tmp_path, entries)
    monkeypatch.delenv("HERMES_ALLOW_PRIVATE_URLS", raising=False)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *_a, **_kw: [
        (socket.AF_INET6 if ":" in ip else socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))
        for ip in addresses
    ])
    parsed = urlparse(url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    with profile(tmp_path):
        assert url_safety.is_safe_url(url) is allowed
        if allowed:
            assert url_safety._resolved_http_connect_ips(parsed.hostname, port, parsed.scheme) == addresses
        else:
            with pytest.raises(url_safety.SSRFConnectionBlocked):
                url_safety._resolved_http_connect_ips(parsed.hostname, port, parsed.scheme)


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_real_connections_redirects_and_profile_isolation(tmp_path, monkeypatch, asynchronous):
    """Real config → validator → httpx → TCP, with two profiles in one process."""
    from agent.secret_scope import is_multiplex_active, set_multiplex_active

    monkeypatch.delenv("HERMES_ALLOW_PRIVATE_URLS", raising=False)
    hits = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append((self.server.server_port, self.path))
            targets = {
                "/same": "/ok",
                "/escape": f"http://127.0.0.1:{blocked.server_port}/secret",
                "/metadata": "http://169.254.169.254/latest/meta-data/",
            }
            self.send_response(302 if self.path in targets else 200)
            if self.path in targets:
                self.send_header("Location", targets[self.path])
            self.end_headers()
            self.wfile.write(b"allowed")

        def log_message(self, *_args):
            pass

    allowed = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    blocked = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    servers = [allowed, blocked]
    threads = [threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}) for server in servers]
    for thread in threads:
        thread.start()
    home_a, home_b = tmp_path / "a", tmp_path / "b"
    write_policy(home_a, [f"127.0.0.1/32:{allowed.server_port}"])
    write_policy(home_b, [])
    base = f"http://127.0.0.1:{allowed.server_port}"

    async def async_get(url):
        async with url_safety.create_ssrf_safe_async_client(trust_env=False, follow_redirects=True, timeout=2) as client:
            return await client.get(url)

    def get(url):
        if asynchronous:
            return asyncio.run(async_get(url))
        with url_safety.create_ssrf_safe_client(trust_env=False, follow_redirects=True, timeout=2) as client:
            return client.get(url)

    was_multiplex = is_multiplex_active()
    set_multiplex_active(True)
    try:
        for home, permits in [(home_a, True), (home_b, False), (home_a, True)]:
            with profile(home):
                assert url_safety.is_safe_url(base) is permits
                if permits:
                    assert get(base + "/same").text == "allowed"
                    for path in ("/escape", "/metadata"):
                        with pytest.raises(url_safety.SSRFConnectionBlocked):
                            get(base + path)
                else:
                    with pytest.raises(url_safety.SSRFConnectionBlocked):
                        get(base)
        assert not any(port == blocked.server_port for port, _ in hits)
    finally:
        set_multiplex_active(was_multiplex)
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=2)
