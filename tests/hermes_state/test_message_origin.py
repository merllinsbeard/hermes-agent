"""Literal origin survives native SQLite reconciliation, copy, repair and replay."""

import copy
import json

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    handle = SessionDB(tmp_path / "state.db")
    handle.create_session("s", "cli")
    yield handle
    handle.close()


def origin(meaning="source_event"):
    return {
        "version": 1,
        "contributors": [
            {
                "locator": {"provider": "hermes"},
                "refs": [{"kind": "host_row", "value": "row-A"}],
                "times": [
                    {
                        "literal": 0,
                        "encoding": "unix_seconds",
                        "meaning": meaning,
                        "basis": {"producer": "hermes", "field": "test.native.time"},
                    }
                ],
                "transport_sender": None,
                "primary_author": None,
                "content_class": "unknown",
                "class_basis": None,
            }
        ],
    }


def test_nullable_column_reconciles_old_database_without_backfill(tmp_path):
    path = tmp_path / "old.db"
    with SessionDB(path) as db:
        db.create_session("s", "cli")
        db.append_message("s", "user", "old", timestamp=0)
        db._conn.execute("ALTER TABLE messages DROP COLUMN origin_statements")
        db._conn.commit()
    with SessionDB(path) as db:
        assert (
            db._conn.execute("SELECT origin_statements FROM messages").fetchone()[0]
            is None
        )
        assert "origin_statements" not in db.get_messages_as_conversation("s")[0]


def test_single_append_display_overwrite_and_all_resume_projections(db):
    db.append_message("s", "user", "same", timestamp=0, origin_statements=origin())
    assert db.set_latest_matching_message_display_kind(
        "s",
        role="user",
        content="same",
        display_kind="other",
        display_metadata={"replacement": True},
    )
    raw = db._conn.execute("SELECT origin_statements FROM messages").fetchone()[0]
    assert json.loads(raw) == origin()
    assert db.get_messages("s")[0]["origin_statements"] == origin()
    for view in (
        db.get_messages_as_conversation("s"),
        *db.get_resume_conversations("s"),
    ):
        assert view[0]["origin_statements"] == origin()
        assert view[0]["timestamp"] == 0
    from gateway.run import _build_gateway_agent_history

    replay, _ = _build_gateway_agent_history(
        db.get_messages_as_conversation("s"), inject_timestamps=False
    )
    assert replay[0]["origin_statements"] == origin()
    assert replay[0]["timestamp"] == 0


def test_replace_reissue_preserves_claims_and_unknown_rows(db):
    db.append_messages_batch(
        "s",
        [
            {
                "role": "user",
                "content": "known",
                "timestamp": 0,
                "origin_statements": origin(),
            },
            {"role": "assistant", "content": "old unknown", "timestamp": 1},
        ],
    )
    restored = db.get_messages_as_conversation("s")
    db.replace_messages("s", restored)
    rows = db.get_messages_as_conversation("s")
    assert rows[0]["origin_statements"] == origin()
    assert "origin_statements" not in rows[1]


@pytest.mark.parametrize("rotate", [False, True])
def test_compaction_handoff_and_concurrent_tail_clone_keep_origins(db, rotate):
    db.append_messages_batch(
        "s",
        [
            {
                "role": "user",
                "content": "q",
                "timestamp": 0,
                "origin_statements": origin("host_stamp"),
            },
            {
                "role": "assistant",
                "content": "a",
                "timestamp": 1,
                "origin_statements": origin(),
            },
        ],
    )
    watermark = db.get_messages("s")[-1]["id"]
    db.append_message(
        "s", "user", "late", timestamp=2, origin_statements=origin("unknown")
    )
    rows = db.get_messages_as_conversation("s")
    handoff = [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
        copy.deepcopy(rows[1]),
    ]
    target = "child" if rotate else "s"
    if rotate:
        db.publish_compression_child(
            parent_session_id="s",
            child_session_id="child",
            source="cli",
            messages=handoff,
            require_compression_lease=False,
            watermark=watermark,
            watermark_ceiling=db.get_messages("s")[-1]["id"],
        )
    else:
        db.archive_and_compact("s", handoff, watermark=watermark)
    copied = db.get_messages_as_conversation(target)
    assert copied[1]["origin_statements"] == origin()
    assert copied[2]["origin_statements"] == origin("unknown")
    assert "origin_statements" not in copied[0]


def test_row_addressed_repair_updates_full_origin_and_replays_it(db):
    msg = {
        "role": "user",
        "content": "first",
        "timestamp": 0,
        "origin_statements": origin("unknown"),
    }
    db.append_messages_batch("s", [msg])
    before = db.get_messages("s")[0]["id"]
    msg["content"] = "folded content"
    msg["origin_statements"] = origin("source_event")
    db.append_messages_batch("s", [msg])
    stored = db.get_messages("s")
    assert len(stored) == 1 and stored[0]["id"] == before
    assert stored[0]["origin_statements"] == origin()
    assert msg["origin_statements"] == origin()
    assert db.get_messages_as_conversation("s")[0]["origin_statements"] == origin()


def test_kept_prefix_adopts_stored_origin_before_a_later_reissue(db):
    db.append_message("s", "user", "known", timestamp=0, origin_statements=origin())
    prefix = [{"role": "user", "content": "known"}]
    db.replace_messages("s", prefix, archive_dropped=True)
    assert prefix[0]["origin_statements"] == origin()
    db.replace_messages("s", prefix)
    assert db.get_messages("s")[0]["origin_statements"] == origin()


def test_different_supplied_origin_is_not_silently_adopted_as_kept_prefix(db):
    db.append_message(
        "s", "user", "same", timestamp=0, origin_statements=origin("unknown")
    )
    replacement = [{"role": "user", "content": "same", "origin_statements": origin()}]
    db.replace_messages("s", replacement, archive_dropped=True)
    assert db.get_messages("s")[0]["origin_statements"] == origin()
    assert any(
        row["origin_statements"] == origin("unknown")
        for row in db.get_messages("s", include_inactive=True)
    )


@pytest.mark.parametrize("boundary", ["final_override", "db_only_override"])
def test_fold_reanchor_override_and_native_flush_keep_both_contributor_claims(db, boundary):
    from agent.agent_runtime_helpers import _merge_consecutive_users
    from agent.session_persistence import SessionPersistenceMixin
    from agent.turn_context_compaction import _reanchor
    from agent.turn_finalizer import _close_transcript_tail

    earlier, current = origin(), origin("provider_record")
    current["contributors"][0]["refs"] = [{"kind": "host_row", "value": "row-B"}]
    current["contributors"][0]["times"][0]["literal"] = "2026-10-17T09:10:11.123456+07:00"
    current["contributors"][0]["times"][0]["encoding"] = "rfc3339"
    expected = copy.deepcopy(earlier["contributors"] + current["contributors"])
    api_text, clean_text = "[API-only guidance] Current request", "Current request"
    prefix = "Unanswered earlier request"
    messages = [
        {"role": "assistant", "content": "Previous reply", "timestamp": 0},
        {"role": "user", "content": prefix, "timestamp": 0,
         "message_uid": "row-A", "origin_statements": earlier},
        {"role": "user", "content": api_text, "timestamp": 7,
         "message_uid": "row-B", "origin_statements": current},
    ]
    agent = SessionPersistenceMixin()
    agent._session_db, agent._session_db_created, agent.session_id = db, True, "s"
    agent._persist_user_message_idx = 2
    agent._persist_user_message_override = clean_text
    agent._persist_user_message_timestamp = 7
    agent._persist_user_origin_statements = copy.deepcopy(current)
    folded, repairs = _merge_consecutive_users(messages)
    assert repairs == 1 and len(folded) == 2
    assert folded[1]["origin_statements"]["contributors"] == expected
    assert _reanchor(agent, folded, api_text) == 1
    assert agent._persist_user_message_idx == 1
    if boundary == "final_override":
        # The real finalizer seam invokes the native override, twice to prove
        # replay/idempotency cannot drop the absorbed contributor either.
        _close_transcript_tail(agent, folded, None, False, False)
        _close_transcript_tail(agent, folded, None, False, False)
    else:
        assert folded[1]["content"] == prefix + "\n\n" + api_text
    assert agent._flush_messages_to_session_db(folded) is True
    stored = db.get_messages_as_conversation("s")
    assert stored[1]["content"] == prefix + "\n\n" + clean_text
    assert stored[1]["origin_statements"]["contributors"] == expected
    assert len(stored[1]["origin_statements"]["contributors"]) == 2
    assert folded[1]["origin_statements"]["contributors"] == expected
    if boundary == "db_only_override":
        assert folded[1]["content"] == prefix + "\n\n" + api_text
    assert agent._flush_messages_to_session_db(folded) is True
    assert len(db.get_messages("s")) == 2
    assert db.get_messages_as_conversation("s")[1]["origin_statements"]["contributors"] == expected
