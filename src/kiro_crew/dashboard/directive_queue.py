"""Provider-neutral delivery for session directives.

The directive marker (:mod:`kiro_crew.session_directive`) is model-visible text,
so the consumer may only honour it when the tool CALL it arrived under was
recorded as an MCP call served by Kiro Crew's own core server. That recording
comes from the provider's out-of-band ``_meta.kiro`` channel, which is a
kiro-cli engine feature: an ACP backend that does not emit it leaves the gate
with no trusted source, and a gate with no trusted source correctly refuses
every directive. The whole control plane (loops, project changes, cards) then
fails closed on that backend — silently, until #6970 added the diagnostic.

This module is the second delivery path, and it carries the payload OUT OF BAND
rather than through the model's tool result. The MCP tool, having validated its
arguments, POSTs them to the gateway over Kiro Crew's own internal API declaring
its ``X-Session-Key``; the gateway parks the record here; the turn's consumer
claims it. The marker is still emitted (the kiro-cli path is unchanged and
remains authoritative there), but on a backend without ``_meta.kiro`` the marker
is reduced to a HINT that a record may be waiting — its CONTENT is never read.

Why this is not weaker than the marker gate it backs up
------------------------------------------------------
The threat is a model forging a directive. It gains nothing here:

* A record is keyed by the session that published it, and a directive only ever
  affects the session that claims it. A model publishing a record for its OWN
  session has done exactly what calling the tool honestly does.
* Publishing for ANOTHER session is the attack, and the transport already stops
  it: ``token_auth`` kernel-verifies AF_UNIX peers (``SO_PEERCRED`` plus a /proc
  ancestry walk to an HMAC-signed session file, keyed by a trust root the agent
  cannot read) and denies 403 when the peer resolves to a session key other than
  the declared one.
* The payload is the one the TOOL validated, not one lifted from result text, so
  a model editing the marker's JSON changes nothing that gets applied.

Deliberately NOT persisted. A directive is turn-scoped: the turn that requested
it is what gives it meaning. Surviving a gateway restart would let a loop arm, or
a project change land, against a turn that no longer exists — so records live in
memory and are additionally dropped on age.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any

from kiro_crew.session_directive import DIRECTIVE_TOOLS

logger = logging.getLogger(__name__)

#: Records older than this are dropped unclaimed. A directive belongs to the turn
#: that asked for it; a turn does not outlive this by any normal margin, and a
#: record that does has lost the context that made it meaningful.
MAX_AGE_SECS = 300.0

#: Per-session cap. A turn emits one directive per tool call and the consumer
#: claims the queue whole, so depth beyond this means records are not being
#: claimed (a tabless session, a backend emitting neither path). Bounding it keeps
#: an unclaimed queue from growing without limit; the OLDEST is dropped, so a live
#: session always retains its most recent intent.
MAX_PER_SESSION = 8

_lock = threading.Lock()
_pending: dict[str, list[dict[str, Any]]] = {}


def publish(session_key: str, kind: str, args: dict[str, Any]) -> str:
    """Park a validated directive for *session_key*; return its record id.

    Raises :class:`ValueError` for an unknown *kind* or an empty *session_key* —
    the caller is the gateway handler, and an unrecognized kind means the request
    did not come from one of Kiro Crew's own directive tools.
    """
    if kind not in DIRECTIVE_TOOLS:
        raise ValueError(f"unknown directive kind: {kind!r}")
    if not session_key:
        raise ValueError("session_key required")
    rec_id = uuid.uuid4().hex
    record = {
        "id": rec_id,
        "kind": kind,
        "args": dict(args or {}),
        "at": time.monotonic(),
    }
    with _lock:
        queue = _pending.setdefault(session_key, [])
        queue.append(record)
        while len(queue) > MAX_PER_SESSION:
            dropped = queue.pop(0)
            logger.warning(
                "session-directive queue full for %s: dropped unclaimed %s "
                "(cap %d). Nothing claimed these — the session may hold no "
                "consumer.",
                session_key,
                dropped.get("kind"),
                MAX_PER_SESSION,
            )
    return rec_id


def claim(session_key: str) -> list[dict[str, Any]]:
    """Remove and return the fresh directives parked for *session_key*.

    Single-consume by construction: the queue is emptied under the lock, so two
    consumers racing the same session cannot both apply the same record. Stale
    records are dropped rather than returned.
    """
    if not session_key:
        return []
    now = time.monotonic()
    with _lock:
        queue = _pending.pop(session_key, [])
    fresh: list[dict[str, Any]] = []
    for record in queue:
        age = now - float(record.get("at", 0.0))
        if age <= MAX_AGE_SECS:
            fresh.append(record)
            continue
        logger.info(
            "session-directive dropped as stale for %s: %s (age %.0fs > %.0fs)",
            session_key,
            record.get("kind"),
            age,
            MAX_AGE_SECS,
        )
    return fresh


def discard(session_key: str) -> int:
    """Drop any parked directives for *session_key*; return how many.

    The kiro-cli path applies the directive from the marker under a verified
    ``_meta.kiro`` identity. The out-of-band record for that same call is then a
    DUPLICATE, and applying both would arm two loops or render two cards — so the
    marker path calls this to retire its twin.
    """
    if not session_key:
        return 0
    with _lock:
        return len(_pending.pop(session_key, []))


def depth(session_key: str) -> int:
    """Parked record count for *session_key* — diagnostics only, no claim."""
    with _lock:
        return len(_pending.get(session_key, []))


def reset() -> None:
    """Drop every parked record. For tests and gateway shutdown."""
    with _lock:
        _pending.clear()
