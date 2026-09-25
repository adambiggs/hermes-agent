"""Apply the process CA bundle to Edge TTS without disabling verification."""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_CA_BUNDLE_ENV_VARS = (
    "HERMES_CA_BUNDLE",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
)

# Submodules of edge_tts that construct an SSL context at import time.
_EDGE_TTS_SSL_MODULES = ("communicate", "voices")


def _configured_ca_bundle() -> Optional[str]:
    """Return the CA bundle Hermes is configured to trust, if any.

    Same env-var precedence as ``agent.ssl_guard`` and the ``requests``
    verify resolution in ``agent.model_metadata`` so one variable covers every
    call site in the process. Paths that do not exist are skipped rather than
    trusted blindly.
    """
    for env_var in _CA_BUNDLE_ENV_VARS:
        value = os.getenv(env_var)
        if value and os.path.isfile(value):
            return value
    return None


def apply_edge_tts_ca_trust(edge_tts_module) -> bool:
    """Add the configured CA roots to edge-tts's SSL contexts.

    Returns True when at least one context was extended. Scans module
    attributes for ``ssl.SSLContext`` instances rather than naming edge-tts's
    private constant, so an upstream rename does not silently reintroduce the
    verification failure.
    """
    bundle = _configured_ca_bundle()
    if not bundle:
        return False

    import ssl as _ssl

    extended = False
    for module_name in _EDGE_TTS_SSL_MODULES:
        submodule = getattr(edge_tts_module, module_name, None)
        if submodule is None:
            continue
        for context in list(vars(submodule).values()):
            if not isinstance(context, _ssl.SSLContext):
                continue
            try:
                context.load_verify_locations(cafile=bundle)
            except Exception as e:
                logger.warning(
                    "Could not add CA bundle %s to edge-tts trust store: %s",
                    bundle, e,
                )
            else:
                extended = True

    if not extended:
        logger.debug(
            "edge-tts exposes no module-level SSL context; CA bundle %s not applied",
            bundle,
        )
    return extended

