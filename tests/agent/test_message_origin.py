"""Native producer, host folds and provider projections of literal origin claims."""

import copy
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from agent.message_metadata import (
    append_message,
    is_compaction_scaffolding,
    message_identity,
    record_absorbed_message,
    scaffolding_replay_fields,
    stamp_message_timestamp,
)


def test_host_clock_observation_is_sampled_once_before_coercion(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "agent.message_metadata.wall_time", lambda: calls.append(0) or 0
    )
    message = {"role": "user", "content": "hello"}
    append_message([], message)
    stamp_message_timestamp(message)
    assert calls == [0]
    assert message["timestamp"] == 0
    contributor = message["origin_statements"]["contributors"][0]
    assert contributor["times"] == [
        {
            "literal": 0,
            "encoding": "unix_seconds",
            "meaning": "host_stamp",
            "basis": {"producer": "hermes", "field": "stamp_message_timestamp"},
        }
    ]
    assert contributor["refs"] == [
        {"kind": "host_row", "value": message["message_uid"]}
    ]
    assert contributor["primary_author"] is None
    assert contributor["content_class"] == "unknown"


@pytest.mark.parametrize(
    "literal", [0, 0.0, "2026-10-17T09:10:11.123456+07:00", "undated"]
)
def test_explicit_unmarked_timestamp_retains_its_literal_and_unknown_basis(literal):
    message = {"role": "user", "timestamp": literal}
    stamp_message_timestamp(message)
    time = message["origin_statements"]["contributors"][0]["times"][0]
    assert time["literal"] == literal and type(time["literal"]) is type(literal)
    assert time["meaning"] == "unknown"


def test_event_clock_default_and_explicit_zero_are_different_observations(monkeypatch):
    from gateway.platforms import event

    class Clock:
        @staticmethod
        def now():
            return datetime(1970, 1, 1, tzinfo=UTC)

    monkeypatch.setattr(event, "datetime", Clock)
    default = event.MessageEvent("host")
    explicit = event.MessageEvent("legacy", timestamp=0)
    assert (
        default.origin_statements["contributors"][0]["times"][0]["meaning"]
        == "host_stamp"
    )
    assert explicit.origin_statements["contributors"][0]["times"][0] == {
        "literal": 0,
        "encoding": "unix_seconds",
        "meaning": "unknown",
        "basis": {"producer": "hermes", "field": "MessageEvent.timestamp"},
    }

    claims = {"version": 1, "contributors": [{"locator": {"provider": "platform"},
        "times": [{"literal": 0, "encoding": "unix_seconds", "meaning": "source_event",
                   "basis": {"producer": "platform", "field": "event.time"}}]}]}
    original = copy.deepcopy(claims)
    marked = event.MessageEvent("marked", origin_statements=claims)
    assert marked.origin_statements["contributors"][0] == original["contributors"][0]
    assert marked.origin_statements["contributors"][1]["times"][0]["meaning"] == "host_stamp"
    monkeypatch.setattr("agent.message_metadata.wall_time", lambda: 0)
    row = stamp_message_timestamp({"role": "user", "origin_statements": claims})
    stamp_message_timestamp(row)
    assert row["origin_statements"]["contributors"][0] == original["contributors"][0]
    assert row["origin_statements"]["contributors"][1]["times"][0]["literal"] == 0
    assert row["origin_statements"]["contributors"][1]["times"][0]["meaning"] == "host_stamp"
    assert len(row["origin_statements"]["contributors"]) == 2
    assert claims == original


def test_absorbed_constituents_union_before_their_rows_disappear():
    survivor = stamp_message_timestamp(
        {"role": "user", "content": "first", "timestamp": 0}
    )
    other = stamp_message_timestamp(
        {"role": "user", "content": "second", "timestamp": "undated"}
    )
    expected = copy.deepcopy(
        survivor["origin_statements"]["contributors"]
        + other["origin_statements"]["contributors"]
    )
    record_absorbed_message(survivor, other)
    record_absorbed_message(survivor, other)
    assert survivor["origin_statements"]["contributors"] == expected
    assert len(survivor["_absorbed_message_uids"]) == 1
    other["origin_statements"]["contributors"][0]["times"].clear()
    assert survivor["origin_statements"]["contributors"] == expected
    record_absorbed_message(
        survivor, {"message_uid": "old-without-claims", "timestamp": 999}
    )
    witness = survivor["origin_statements"]["contributors"][-1]
    assert witness["refs"] == [{"kind": "host_row", "value": "old-without-claims"}]
    assert witness["times"] == []


def test_origin_does_not_make_ordinary_messages_compaction_scaffolding():
    msg = stamp_message_timestamp(
        {"role": "user", "content": "ordinary", "timestamp": 0}
    )
    assert scaffolding_replay_fields(msg) == {}
    assert not is_compaction_scaffolding(msg)
    copy = message_identity(msg)
    assert copy["origin_statements"] == msg["origin_statements"]
    assert copy["origin_statements"] is not msg["origin_statements"]


def test_turn_stage_flush_and_override_keep_explicit_source_claims():
    from agent.message_origin import observed_origin
    from agent.session_persistence import SessionPersistenceMixin, _db_flush_row
    from agent.turn_context import _stage_turn_user_message

    origin = observed_origin(
        0, field="platform.event.time", meaning="source_event", source_id="event-0"
    )
    agent = SimpleNamespace(
        _turn_author={"id": "sender-A"},
        _persist_user_message_override="clean",
        _persist_user_message_timestamp=0,
        _persist_user_message_idx=0,
        _persist_user_origin_statements=origin,
    )
    msg, _ = _stage_turn_user_message(
        agent, "api text", "clean", 0, "event-0", None, None, origin
    )
    assert (
        msg["origin_statements"]["contributors"][0]["transport_sender"]["actor"][
            "identifier"
        ]
        == "sender-A"
    )
    assert msg["origin_statements"]["contributors"][0]["primary_author"] is None
    SessionPersistenceMixin._apply_persist_user_message_override(agent, [msg])
    row = _db_flush_row(agent, msg, True)
    assert row["content"] == "clean" and row["timestamp"] == 0
    assert row["origin_statements"] == msg["origin_statements"]
    assert (
        row["origin_statements"]["contributors"][0]["times"]
        == origin["contributors"][0]["times"]
    )


def test_runtime_marker_and_explicit_relay_claim_do_not_infer_primary_author():
    from agent.message_origin import annotate_turn_origin, observed_origin

    origin = observed_origin(0, field="event.timestamp")
    runtime = annotate_turn_origin(origin, display_kind="internal_notification")
    assert runtime["contributors"][0]["content_class"] == "runtime_self_metadata"
    relay = copy.deepcopy(origin)
    relay["contributors"][0].update(
        content_class="bb_relay",
        class_basis={"producer": "bb", "field": "relay_metadata.origin_statements"},
    )
    merged = annotate_turn_origin(origin, relay_metadata={"origin_statements": relay})
    assert [c["content_class"] for c in merged["contributors"]] == [
        "unknown",
        "bb_relay",
    ]
    assert all(c["primary_author"] is None for c in merged["contributors"])
    assert (
        annotate_turn_origin(origin, relay_metadata={"arbitrary_text": "[bb message]"})
        == origin
    )


def test_real_wire_and_token_estimator_exclude_origin_while_provider_redaction_keeps_it():
    from agent.memory_manager import _redact_for_provider
    from agent.model_metadata import estimate_messages_tokens_rough
    from agent.turn_context import build_api_messages

    agent = SimpleNamespace(
        _current_turn_timestamp=1,
        _copy_reasoning_content_for_api=lambda *_: None,
        _should_sanitize_tool_calls=lambda: False,
        ephemeral_system_prompt=None,
    )
    plain = {"role": "user", "content": "hello"}
    msg = stamp_message_timestamp(copy.deepcopy(plain), timestamp=0)
    wire, _ = build_api_messages(
        agent,
        [msg],
        current_turn_user_idx=0,
        ext_prefetch_cache=None,
        plugin_user_context=None,
        moa_config=None,
        active_system_prompt=None,
    )
    assert wire == [plain]
    assert estimate_messages_tokens_rough([msg]) == estimate_messages_tokens_rough(
        [plain]
    )
    msg["origin_statements"]["contributors"][0]["times"].append(
        {
            "literal": "2026-10-17T09:10:11.123456+07:00",
            "encoding": "rfc3339",
            "meaning": "source_event",
            "basis": {"producer": "hermes", "field": "platform.event.time"},
        }
    )
    redacted = _redact_for_provider([msg])
    assert redacted[0]["origin_statements"] == msg["origin_statements"]


@pytest.mark.parametrize(
    "envelope",
    [
        {"version": 2, "contributors": []},
        {
            "version": 1,
            "contributors": [],
            "receiver": {"session_ingest_started_at": "fake"},
        },
        {"version": 1, "contributors": [{"times": [float("nan")]}]},
    ],
)
def test_native_json_boundary_refuses_receiver_and_unsupported_schema(envelope):
    from agent.message_origin import origin_json

    with pytest.raises(ValueError):
        origin_json(envelope)


def test_interrupted_admission_retains_explicit_claims_and_runtime_marker():
    from agent.message_origin import observed_origin
    from agent.turn_facade_lease import carry_unadmitted_user_message

    origin = observed_origin(0, field="source.time", meaning="source_event")
    result = {"interrupted": True, "messages": []}
    carry_unadmitted_user_message(
        result,
        "api",
        "clean",
        timestamp=0,
        display_kind="internal_notification",
        display_metadata=None,
        platform_id="id-0",
        origin_statements=origin,
    )
    msg = result["messages"][0]
    assert (
        msg["origin_statements"]["contributors"][0]["times"][0]["meaning"]
        == "source_event"
    )
    assert (
        msg["origin_statements"]["contributors"][0]["content_class"]
        == "runtime_self_metadata"
    )
    assert msg["timestamp"] == 0


def test_native_memory_manager_hands_marked_transcript_to_provider():
    """MOCK provider sink only; the native serialized worker and redaction stay real."""
    from threading import Event
    from unittest.mock import Mock
    from agent.memory_manager import MemoryManager
    from agent.memory_provider import MemoryProvider

    delivered, received = Event(), []
    sink = Mock(spec=MemoryProvider)
    sink.name = "MOCK origin receiver"
    sink.get_tool_schemas.return_value = []

    def receive(user_content, assistant_content, **kwargs):
        received.append(kwargs)
        delivered.set()

    sink.sync_turn.side_effect = receive
    manager = MemoryManager()
    manager.add_provider(sink)
    message = stamp_message_timestamp(
        {"role": "user", "content": "A useful source observation", "timestamp": 0}
    )
    original = copy.deepcopy(message)
    try:
        manager.sync_all(
            message["content"],
            "Preserved",
            session_id="native-session",
            messages=[message],
        )
        assert delivered.wait(5)
        assert (
            received[0]["messages"][0]["origin_statements"]
            == original["origin_statements"]
        )
        assert message == original
    finally:
        manager.shutdown_all()
