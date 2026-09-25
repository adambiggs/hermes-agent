"""Edge's request context trusts configured roots without weakening TLS checks."""

import asyncio
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
from pathlib import Path
import ssl
from threading import Thread
from types import ModuleType
from urllib.error import URLError
from urllib.request import urlopen

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import pytest

from tools import tts_tool
from tools.tts_tool_edge import apply_edge_tts_ca_trust
from tools.tts_tool_providers import _generate_edge_tts


@pytest.fixture
def tls_endpoint(tmp_path, monkeypatch):
    for name in ("HERMES_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        monkeypatch.delenv(name, raising=False)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(x509.NameOID.COMMON_NAME, "Test CA")])
    now = datetime.now(timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(subject).issuer_name(subject)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(days=1)).not_valid_after(now + timedelta(days=1))
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), critical=False)
            .sign(key, hashes.SHA256()))
    ca = tmp_path / "ca.pem"
    ca.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    private_key = tmp_path / "key.pem"
    private_key.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    private_key.chmod(0o600)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"verified speech")

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(ca, private_key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield ca, server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_edge_generation_uses_configured_ca_and_keeps_peer_verification(tls_endpoint, tmp_path, monkeypatch):
    ca, port = tls_endpoint
    sdk = ModuleType("edge_tts")
    sdk.communicate = ModuleType("edge_tts.communicate")
    sdk.communicate.context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

    class Communicate:
        def __init__(self, text, **kwargs):
            self.text = text

        async def save(self, output_path):
            with urlopen(f"https://127.0.0.1:{port}/", context=sdk.communicate.context, timeout=5) as response:
                Path(output_path).write_bytes(response.read())

    sdk.Communicate = Communicate
    monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: sdk)
    output = tmp_path / "speech.mp3"
    with pytest.raises(URLError):
        asyncio.run(_generate_edge_tts("test", str(output), {}))
    monkeypatch.setenv("SSL_CERT_FILE", str(ca))
    assert asyncio.run(_generate_edge_tts("test", str(output), {})) == str(output)
    assert output.read_bytes() == b"verified speech"
    assert sdk.communicate.context.verify_mode == ssl.CERT_REQUIRED
    assert sdk.communicate.context.check_hostname
    with pytest.raises(URLError):
        urlopen(f"https://localhost:{port}/", context=sdk.communicate.context, timeout=5)


def test_unusable_ca_never_disables_verification(tmp_path, monkeypatch):
    for name in ("HERMES_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        monkeypatch.delenv(name, raising=False)
    junk = tmp_path / "invalid.pem"
    junk.write_text("invalid certificate")
    monkeypatch.setenv("HERMES_CA_BUNDLE", str(junk))
    sdk = ModuleType("edge_tts")
    assert not apply_edge_tts_ca_trust(sdk)
    sdk.communicate = ModuleType("edge_tts.communicate")
    sdk.communicate.context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    assert not apply_edge_tts_ca_trust(sdk)
    assert sdk.communicate.context.verify_mode == ssl.CERT_REQUIRED
    assert sdk.communicate.context.check_hostname
    assert sdk.communicate.context.get_ca_certs() == []
