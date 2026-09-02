"""Descriptor-confined loopback previews for ``file:///output/...`` URLs."""

from __future__ import annotations

import atexit
import email.utils
import http.server
import json
import mimetypes
import os
import shutil
import stat
import threading
import unicodedata
import urllib.parse
from pathlib import Path
from typing import Optional


OUTPUT_PREVIEW_ROOT = Path("/output")
OUTPUT_PREVIEW_QUERY = "__hermes_file_preview=1"
_server_lock = threading.Lock()
_server: Optional["_OutputPreviewServer"] = None
_server_thread: Optional[threading.Thread] = None


def decode_output_preview_segments(url: str) -> Optional[list[str]]:
    """Return safe path segments for file:///output/... or None if non-file.

    The returned segments are relative to /output. Ambiguous URL forms are
    rejected before any filesystem access. Symlinks are rejected separately by
    descriptor-relative O_NOFOLLOW opens in ``_OutputPreviewServer.open_file``.
    """
    # Classify a possible file URL from the caller's raw text before the
    # browser's generic HTTP URL normalizer can trim or repair it. Avoid
    # parsing unrelated malformed HTTP URLs here: urlsplit itself can raise
    # for those, but they belong to the normal browser safety path.
    if not url.lstrip().lower().startswith("file:"):
        return None
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme.lower() != "file":
        raise ValueError("file preview URL has an ambiguous scheme")
    if not url.startswith("file:///output/"):
        raise ValueError("file preview URL is not canonical file:///output/")
    if any(
        character.isspace() or unicodedata.category(character) == "Cc"
        for character in url
    ):
        raise ValueError("file preview URL contains raw whitespace or controls")
    if parsed.netloc or "?" in url or "#" in url:
        raise ValueError("file preview URL has authority, query, or fragment")
    try:
        decoded_path = urllib.parse.unquote_to_bytes(parsed.path).decode("utf-8")
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("file preview URL has invalid path encoding") from exc
    if "\\" in decoded_path or any(
        unicodedata.category(character) == "Cc" for character in decoded_path
    ):
        raise ValueError("file preview URL has an ambiguous path")
    parts = decoded_path.split("/")
    if (
        len(parts) < 3
        or parts[0] != ""
        or parts[1] != "output"
        or any(part in {"", ".", ".."} for part in parts[2:])
    ):
        raise ValueError("file preview URL is outside /output")
    return parts[2:]


class _OutputPreviewServer(http.server.ThreadingHTTPServer):
    """Loopback HTTP server whose file opens cannot escape one root fd."""

    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, root: Path):
        if (
            not hasattr(os, "O_DIRECTORY")
            or not hasattr(os, "O_NOFOLLOW")
            or os.open not in os.supports_dir_fd
        ):
            raise OSError("safe descriptor-relative file opens are unavailable")
        root_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
        self.root_path = root
        self.root_fd = os.open(root, root_flags)
        try:
            super().__init__(("127.0.0.1", 0), _OutputPreviewHandler)
        except Exception:
            os.close(self.root_fd)
            self.root_fd = -1
            raise

    def open_file(self, segments: list[str]) -> int:
        """Open a regular file beneath root without following any symlink."""
        if not segments:
            raise OSError("empty output preview path")
        current_fd = os.dup(self.root_fd)
        nofollow = os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        try:
            for index, segment in enumerate(segments):
                flags = os.O_RDONLY | nofollow
                if index < len(segments) - 1:
                    flags |= os.O_DIRECTORY
                next_fd = os.open(segment, flags, dir_fd=current_fd)
                os.close(current_fd)
                current_fd = next_fd
            if not stat.S_ISREG(os.fstat(current_fd).st_mode):
                raise OSError("output preview target is not a regular file")
            return current_fd
        except Exception:
            os.close(current_fd)
            raise

    def server_close(self) -> None:
        try:
            super().server_close()
        finally:
            if self.root_fd >= 0:
                os.close(self.root_fd)
                self.root_fd = -1


class _OutputPreviewHandler(http.server.BaseHTTPRequestHandler):
    """Serve regular output files only; never list directories or follow links."""

    server_version = "HermesOutputPreview"
    sys_version = ""

    def log_message(self, _format: str, *_args: object) -> None:
        # Output paths may describe private work. Do not copy them to gateway
        # logs merely because Chromium fetched a preview asset.
        return

    def _segments(self) -> list[str]:
        parsed = urllib.parse.urlsplit(self.path)
        try:
            decoded_path = urllib.parse.unquote_to_bytes(parsed.path).decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            raise OSError("invalid preview request path") from exc
        if "\x00" in decoded_path or "\\" in decoded_path:
            raise OSError("ambiguous preview request path")
        parts = decoded_path.split("/")
        if parts[0] != "" or any(part in {"", ".", ".."} for part in parts[1:]):
            raise OSError("unsafe preview request path")
        return parts[1:]

    @staticmethod
    def _is_state_api(segments: list[str]) -> bool:
        return (
            len(segments) >= 6
            and segments[0] == "apps"
            and segments[2:5] == ["api", "v1", "state"]
        )

    def _send_empty(self, status: int) -> None:
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _send_preview_only(self, *, head_only: bool) -> None:
        body = json.dumps({
            "ok": False,
            "error": {
                "code": "preview_only",
                "message": "Server state is unavailable in local file preview.",
            },
        }).encode("utf-8")
        self.send_response(409)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def _serve(self, *, head_only: bool) -> None:
        try:
            segments = self._segments()
            if self._is_state_api(segments):
                self._send_preview_only(head_only=head_only)
                return
            file_fd = self.server.open_file(segments)  # type: ignore[attr-defined]
        except OSError:
            self._send_empty(404)
            return

        with os.fdopen(file_fd, "rb") as source:
            file_stat = os.fstat(source.fileno())
            content_type = mimetypes.guess_type(segments[-1])[0]
            self.send_response(200)
            self.send_header("Content-Type", content_type or "application/octet-stream")
            self.send_header("Content-Length", str(file_stat.st_size))
            self.send_header(
                "Last-Modified",
                email.utils.formatdate(file_stat.st_mtime, usegmt=True),
            )
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            if not head_only:
                shutil.copyfileobj(source, self.wfile)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._serve(head_only=False)

    def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._serve(head_only=True)

    def _state_write_or_reject(self) -> None:
        try:
            segments = self._segments()
        except OSError:
            self._send_empty(404)
            return
        if self._is_state_api(segments):
            self._send_preview_only(head_only=False)
        else:
            self._send_empty(405)

    def do_PUT(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._state_write_or_reject()

    def do_DELETE(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._state_write_or_reject()


def stop_output_preview_server() -> None:
    """Stop the process-local output preview server, if it was started."""
    global _server, _server_thread
    with _server_lock:
        server = _server
        thread = _server_thread
        _server = None
        _server_thread = None
    if server is not None:
        server.shutdown()
        server.server_close()
    if thread is not None and thread.is_alive():
        thread.join(timeout=2)


def _ensure_output_preview_server() -> _OutputPreviewServer:
    """Return the process-local loopback preview server, starting it lazily."""
    global _server, _server_thread
    with _server_lock:
        if _server is not None:
            return _server
        server = _OutputPreviewServer(OUTPUT_PREVIEW_ROOT)
        thread = threading.Thread(
            target=server.serve_forever,
            name="hermes-output-preview",
            daemon=True,
        )
        try:
            thread.start()
        except Exception as exc:
            server.server_close()
            raise OSError("could not start output preview server") from exc
        _server = server
        _server_thread = thread
        return server


def output_preview_url(url: str) -> Optional[str]:
    """Translate safe file:///output/... input to the confined loopback server."""
    segments = decode_output_preview_segments(url)
    if segments is None:
        return None
    try:
        server = _ensure_output_preview_server()
        file_fd = server.open_file(segments)
    except OSError as exc:
        raise ValueError("file preview target is unavailable or unsafe") from exc
    else:
        os.close(file_fd)
    encoded_path = "/".join(urllib.parse.quote(part, safe="-._~") for part in segments)
    host, port = server.server_address
    return f"http://{host}:{port}/{encoded_path}?{OUTPUT_PREVIEW_QUERY}"


atexit.register(stop_output_preview_server)
