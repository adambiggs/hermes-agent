"""Regression for adambiggs/toby#29: only human input can answer clarify."""

import asyncio
from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from tools import clarify_gateway as cm


class _Adapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.TELEGRAM)

    async def connect(self):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="sent")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "private"}


def _event(*, internal=False, text="completion", chat_id="chat"):
    return MessageEvent(
        text=text,
        internal=internal,
        source=SessionSource(
            platform=Platform.TELEGRAM, chat_id=chat_id,
            chat_type="dm", user_id=None if internal else "human",
        ),
    )


@pytest.fixture
def gateway():
    adapter = _Adapter()
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.session_store = None
    runner._busy_input_mode = "interrupt"
    runner._draining = False
    runner._session_key_for_source = build_session_key
    runner._delivery_adapter_for = lambda source: adapter
    runner._authorization_home_for_source = lambda source: None
    runner._is_user_authorized = lambda source: True
    runner._admit_bot_message = lambda source: True
    runner._scale_to_zero_note_real_inbound = lambda: None

    async def admit_hook(event, source):
        return event

    runner._hm_pre_gateway_dispatch_hook = admit_hook
    key = build_session_key(_event().source)
    parent = MagicMock()
    parent._active_children = []
    runner._session_state(key).turn.agent = parent
    adapter._message_handler = runner._handle_message
    adapter._busy_session_handler = runner._handle_active_session_busy_message
    adapter._active_sessions[key] = asyncio.Event()
    yield runner, adapter, key, parent
    cm.clear_session(key)
    cm.clear_session(build_session_key(_event(chat_id="other").source))


@pytest.mark.asyncio
@pytest.mark.parametrize("ingress", ["adapter", "runner"])
async def test_internal_deliveries_leave_mixed_form_unanswered_and_queued(gateway, ingress):
    runner, adapter, key, parent = gateway
    adapter._session_tasks[key] = asyncio.current_task()
    entries = [
        cm.register("open-1", key, "First free response", None),
        cm.register("open-2", key, "Second free response", None),
        cm.register("single", key, "Pick one", ["A", "B"]),
        cm.register("multi", key, "Pick several", ["A", "B"], multi_select=True),
    ]
    events = [_event(internal=True), _event(internal=True)]
    dispatch = adapter.handle_message if ingress == "adapter" else runner._handle_message
    for event in events:
        await dispatch(event)

    assert all(entry.response is None and not entry.event.is_set() for entry in entries)
    assert adapter._pending_messages[key] is events[0]
    assert runner._overflow_queue(key) == [events[1]]
    assert all(event._gateway_accepted for event in events)
    parent.interrupt.assert_not_called()
    parent.steer.assert_not_called()
    for entry in entries:
        assert cm.wait_for_response(entry.clarify_id, timeout=0.01) is None
    assert not cm.has_pending(key)
    # Timing out the human form must not discard the separately admitted notifications.
    assert adapter._pending_messages[key] is events[0]
    assert runner._overflow_queue(key) == [events[1]]


@pytest.mark.asyncio
@pytest.mark.parametrize("choices,multi_select,answer,expected", [
    (None, False, "human answer", "human answer"),
    (["A", "B"], False, "2", "B"),
    (["A", "B"], True, "1,2", '["A", "B"]'),
])
async def test_only_human_input_in_the_prompt_session_resolves_it(
    gateway, choices, multi_select, answer, expected,
):
    runner, adapter, key, parent = gateway
    adapter._session_tasks[key] = asyncio.current_task()
    entry = cm.register("prompt", key, "Question", choices, multi_select=multi_select)
    other_key = build_session_key(_event(chat_id="other").source)
    other = cm.register("other-prompt", other_key, "Unrelated question", None)
    # An internal payload shaped exactly like a valid answer is still a notification.
    await adapter.handle_message(_event(internal=True, text=answer))
    assert entry.response is None and not entry.event.is_set()
    await adapter.handle_message(_event(text=answer))
    assert cm.wait_for_response(entry.clarify_id, timeout=0.01) == expected
    assert other.response is None and not other.event.is_set()
    parent.interrupt.assert_not_called()
    assert adapter._pending_messages[key].internal is True
