"""Producer observations carried by native transcript rows, never provider chat fields.

The envelope contains claims, not authenticated authorship. Persistence never creates
observations: only the live producer at the original clock/field boundary does.
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime
from typing import Any, Mapping, MutableMapping

ORIGIN_STATEMENTS = "origin_statements"


def origin_copy(value: Any) -> dict[str, Any] | None:
    """Detach a JSON envelope without normalizing literals or accepting receiver facts."""
    if value is None:
        return None
    if isinstance(value, str):
        value = json.loads(value)
    if (not isinstance(value, dict) or set(value) != {"version", "contributors"}
            or type(value["version"]) is not int or value["version"] != 1
            or not isinstance(value["contributors"], list)):
        raise ValueError("Unsupported message origin statements")
    return json.loads(json.dumps(value, allow_nan=False))


def origin_json(value: Any) -> str | None:
    """Nullable native TEXT representation; omission stays SQL NULL, including on replay."""
    value = origin_copy(value)
    return None if value is None else json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True)


def observed_origin(value: Any, *, field: str, meaning: str = "unknown",
                    uid: str | None = None, source_id: str | None = None,
                    source_scope: str | None = None) -> dict[str, Any]:
    """One observation before timestamp coercion. Offset/precision and numeric scalar survive."""
    literal = value.isoformat() if isinstance(value, datetime) else value
    times = []
    if type(literal) in (int, float) and (not isinstance(literal, float) or math.isfinite(literal)):
        encoding = "unix_seconds"
    elif isinstance(literal, str):
        encoding = "rfc3339" if re.fullmatch(
            r"\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:[0-5]\d(?:\.\d+)?(?:[Zz]|[+-]\d{2}:\d{2})", literal,
        ) else "opaque"
        if encoding == "rfc3339":
            try:
                datetime.fromisoformat(literal.replace("Z", "+00:00").replace("z", "+00:00"))
            except ValueError:
                encoding = "opaque"
    else:
        encoding = None
    if encoding is not None:
        times.append({"literal": literal, "encoding": encoding, "meaning": meaning,
                      "basis": {"producer": "hermes", "field": field}})
    refs = ([{"kind": "host_row", "value": uid}] if uid else [])
    if source_id:
        refs.append({"kind": "source_message", "value": source_id})
    return {"version": 1, "contributors": [{"locator": {"provider": "hermes", "source_scope": source_scope},
        "refs": refs, "times": times, "transport_sender": None, "primary_author": None,
        "content_class": "unknown", "class_basis": None}]}


def merge_origins(*values: Any) -> dict[str, Any] | None:
    """Union complete contributors; unequal claims and unknown witnesses remain distinct."""
    contributors: dict[str, dict[str, Any]] = {}
    present = False
    for value in values:
        origin = origin_copy(value)
        if origin is None:
            continue
        present = True
        for contributor in origin["contributors"]:
            key = json.dumps(contributor, sort_keys=True, allow_nan=False)
            contributors.setdefault(key, contributor)
    return {"version": 1, "contributors": list(contributors.values())} if present else None


def copy_origin(src: Mapping[str, Any], dst: MutableMapping[str, Any]) -> None:
    """Copy independently of presentation metadata and scaffolding predicates."""
    if src.get(ORIGIN_STATEMENTS) is not None:
        dst[ORIGIN_STATEMENTS] = origin_copy(src[ORIGIN_STATEMENTS])


def annotate_turn_origin(value: Any, *, display_kind: str | None = None,
                         turn_author: Mapping[str, Any] | None = None,
                         relay_metadata: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """Native runtime/explicit BB marker and transport sender, never role-derived authorship.

    Relay metadata alone says nothing about BB: only its explicit origin statements
    carry that claim. Arbitrary text or metadata names are not classified.
    """
    origin = merge_origins(value, (relay_metadata or {}).get(ORIGIN_STATEMENTS))
    if origin is None:
        return None
    for contributor in origin["contributors"]:
        if display_kind and contributor.get("content_class", "unknown") == "unknown":
            contributor["content_class"] = "runtime_self_metadata"
            contributor["class_basis"] = {"producer": "hermes", "field": "persist_user_display_kind"}
        if turn_author and isinstance(turn_author.get("id"), str) and turn_author["id"]:
            if contributor.get("transport_sender") is None:
                contributor["transport_sender"] = {"actor": {"identifier": turn_author["id"]},
                    "basis": {"producer": "hermes", "field": "turn_author.id"}}
    return origin
