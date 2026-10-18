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
