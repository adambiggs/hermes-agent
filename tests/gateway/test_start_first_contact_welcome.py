"""`/start` must answer the first person who ever messages an instance.

Telegram fires `/start` on bot launch and on every deep link, so the gateway
returns "" for it. For an established user that is correct — a help dump would
interrupt a running conversation. For a brand new user it means their first tap
does nothing at all, which is the whole first impression of the product.

These tests pin the narrow exception: a welcome only when the session store is
completely empty, so it can fire at most once per install.
"""

import asyncio
import types

from gateway.run import GatewayRunner


class _Store:
    def __init__(self, any_sessions: bool):
        self._any = any_sessions

    async def has_any_sessions(self) -> bool:
        return self._any


def _gateway(any_sessions: bool, config: dict | None = None) -> types.SimpleNamespace:
    """A bare object bound to the real method — no gateway boot, no network."""
    stub = types.SimpleNamespace(
        async_session_store=_Store(any_sessions),
        config=config if config is not None else {},
        DEFAULT_START_MESSAGE=GatewayRunner.DEFAULT_START_MESSAGE,
    )
    stub._first_contact_welcome = types.MethodType(
        GatewayRunner._first_contact_welcome, stub
    )
    return stub


def test_first_contact_gets_a_welcome():
    welcome = asyncio.run(_gateway(any_sessions=False)._first_contact_welcome())

    assert welcome
    assert welcome == GatewayRunner.DEFAULT_START_MESSAGE


def test_an_instance_in_use_stays_silent():
    """The gate is the whole store, so a welcome can never land mid-conversation."""
    welcome = asyncio.run(_gateway(any_sessions=True)._first_contact_welcome())

    assert welcome == ""


def test_operator_copy_replaces_the_default():
    gateway = _gateway(
        any_sessions=False,
        config={"onboarding": {"start_message": "Hi, this is Hermes.  "}},
    )

    assert asyncio.run(gateway._first_contact_welcome()) == "Hi, this is Hermes."


def test_empty_operator_copy_restores_the_old_silence():
    gateway = _gateway(
        any_sessions=False, config={"onboarding": {"start_message": ""}}
    )

    assert asyncio.run(gateway._first_contact_welcome()) == ""


def test_a_broken_session_store_does_not_break_command_dispatch():
    """A welcome is a nicety; /start must still resolve if the check explodes."""

    class _Broken:
        async def has_any_sessions(self):
            raise RuntimeError("store unavailable")

    stub = types.SimpleNamespace(async_session_store=_Broken(), config={})
    stub._first_contact_welcome = types.MethodType(
        GatewayRunner._first_contact_welcome, stub
    )

    assert asyncio.run(stub._first_contact_welcome()) == ""


def test_the_default_welcome_promises_no_capability():
    """What is connected differs per install; the default must not claim any."""
    text = GatewayRunner.DEFAULT_START_MESSAGE.lower()

    for claim in ("calendar", "email", "reminder", "budget", "bank", "payment"):
        assert claim not in text
