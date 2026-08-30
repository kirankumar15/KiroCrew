"""A bounded slot-detail fetch must not resurrect a row's stale on-disk content.

The two branches of ``GET /api/chat/slots/{slot}`` do not agree on which store
decides a row's CONTENT. The unbounded branch returns ``older +
list(slot.messages)``, so the window decides; the bounded branch reads chained
disk history, so disk does. While the two stores agree that difference is
invisible -- and they stop agreeing the moment a row is rewritten IN PLACE and
not yet flushed.

``chat_regenerate`` is exactly that shape: switching variants sets
``_pending_rewrite`` and broadcasts ``chat_variant_switch``, whose client-side
handler dispatches ``refreshSlot``. Once that refresh is bounded, it reads
through the bounded branch and the response carries the PREVIOUS variant -- so
selecting a variant paints the old one straight back over it.

``_append_unflushed_tail`` cannot cover this: it appends rows the disk read is
MISSING, and a rewritten row is not missing. It is present, at the same
``meta.mid``, holding stale text.
"""

from __future__ import annotations

from typing import Any

import pytest
from aiohttp.test_utils import TestClient, TestServer
from chat_test_helpers import _make_app, _make_state

from kiro_crew.dashboard.state import _ChatSlot

#: Enough settled turns that the bound below is a real slice, not the whole corpus.
SETTLED = 12
LIMIT = 4


@pytest.fixture()
def state(tmp_path: Any) -> Any:
    st = _make_state(tmp_path)
    st.push_slots_update = lambda: None  # type: ignore[method-assign]
    return st


def _row(i: int, content: str) -> dict:
    """A durable row carrying the backend's own per-row stamp."""
    return {
        "role": "user" if i % 2 == 0 else "assistant",
        "content": content,
        "cls": "msg msg-u" if i % 2 == 0 else "msg msg-a",
        "meta": {"mid": f"mid-{i}"},
    }


def _slot_with_unflushed_variant(state: Any, name: str = "chat-1") -> tuple[Any, int]:
    """A slot whose newest assistant row was re-selected but not yet flushed.

    Disk holds every row in its ORIGINAL form. The window holds the same rows at
    the same ids, except the newest assistant row, whose content is the variant
    the user just switched to. This is the state ``chat_variant_switch`` is
    broadcast in.
    """
    on_disk = [_row(i, f"m{i}") for i in range(SETTLED)]
    edited_idx = SETTLED - 1
    window = [dict(r) for r in on_disk]
    window[edited_idx] = {**window[edited_idx], "content": "SELECTED VARIANT"}

    slot = _ChatSlot(key=name)
    slot.messages = window
    slot._disk_older_count = 0
    slot._disk_window_len = len(window)
    slot._pending_rewrite = True  # what chat_regenerate sets before broadcasting
    state._slots[name] = slot
    state.conversation_log.read_messages_chained = (  # type: ignore[method-assign]
        lambda _key: [dict(r) for r in on_disk]
    )
    return slot, edited_idx


async def _get(state: Any, query: str, name: str = "chat-1") -> dict:
    async with TestClient(TestServer(_make_app(state))) as client:
        resp = await client.get(f"/api/chat/slots/{name}{query}")
        assert resp.status == 200
        return await resp.json()


class TestBoundedFetchWithAnUnflushedInPlaceEdit:
    @pytest.mark.asyncio
    async def test_bounded_read_returns_the_selected_variant(self, state: Any) -> None:
        """The bound must not swap the live row for its stale disk twin."""
        _slot_with_unflushed_variant(state)
        data = await _get(state, f"?limit={LIMIT}")
        assert len(data["messages"]) == LIMIT
        assert data["messages"][-1]["content"] == "SELECTED VARIANT"

    @pytest.mark.asyncio
    async def test_bounded_and_unbounded_reads_agree(self, state: Any) -> None:
        """The two branches must not disagree about content for the same row.

        This is the invariant; the case above is one way to violate it.
        """
        _slot_with_unflushed_variant(state)
        bounded = await _get(state, f"?limit={LIMIT}")
        unbounded = await _get(state, "")
        assert bounded["messages"][-1]["content"] == unbounded["messages"][-1]["content"]

    @pytest.mark.asyncio
    async def test_untouched_rows_are_not_rewritten(self, state: Any) -> None:
        """Only the edited row changes -- the overlay is not a blanket replace."""
        _, edited_idx = _slot_with_unflushed_variant(state)
        data = await _get(state, f"?limit={SETTLED}")
        contents = [m["content"] for m in data["messages"]]
        expected = [f"m{i}" for i in range(SETTLED)]
        expected[edited_idx] = "SELECTED VARIANT"
        assert contents == expected

    @pytest.mark.asyncio
    async def test_the_row_count_is_unchanged(self, state: Any) -> None:
        """A substitution, not an append: `total` and the slice must not move."""
        _slot_with_unflushed_variant(state)
        data = await _get(state, f"?limit={LIMIT}")
        assert data["total"] == SETTLED
        assert data["has_more"] is True
        assert data["next_before"] == SETTLED - LIMIT

    @pytest.mark.asyncio
    async def test_a_caller_repeated_id_is_left_alone_rather_than_duplicated(
        self, state: Any
    ) -> None:
        """An id that cannot name ONE row is not proof, so decline the swap.

        ``meta.mid`` on an inbound message is caller-supplied -- one is minted only
        when it is absent -- so a client can post the same id twice. Substituting on
        a bare lookup then rewrites EVERY disk row sharing that id to the same
        window row, which duplicates content instead of refreshing it. This is the
        same over-reach ``_append_unflushed_tail`` documents for a set membership
        test, and it is what ``TestSlotDetailPagination`` catches from the other
        direction.
        """
        dup = "mid-shared"
        on_disk = [
            {"role": "user", "content": "first", "cls": "msg msg-u", "meta": {"mid": dup}},
            {"role": "user", "content": "second", "cls": "msg msg-u", "meta": {"mid": dup}},
        ]
        slot = _ChatSlot(key="chat-1")
        # The window holds ONE row at that id -- the shape a bare lookup would
        # smear over both disk rows.
        slot.messages = [
            {"role": "user", "content": "WINDOW COPY", "cls": "msg msg-u", "meta": {"mid": dup}}
        ]
        slot._disk_older_count = 0
        slot._disk_window_len = 1
        state._slots["chat-1"] = slot
        state.conversation_log.read_messages_chained = (  # type: ignore[method-assign]
            lambda _key: [dict(r) for r in on_disk]
        )

        data = await _get(state, "?limit=2")
        contents = [m["content"] for m in data["messages"]]
        # Both disk rows survive as themselves; neither is replaced by the window
        # copy, and neither is duplicated.
        assert contents == ["first", "second"], contents

    @pytest.mark.asyncio
    async def test_an_older_session_row_without_a_matching_id_passes_through(
        self, state: Any
    ) -> None:
        """A chained read spans older sessions whose ids this window never held.

        Those rows must survive untouched rather than be dropped or swapped.
        """
        slot, _ = _slot_with_unflushed_variant(state)
        older = [
            {
                "role": "assistant",
                "content": "FROM AN OLDER SESSION",
                "cls": "msg msg-a",
                "meta": {"mid": "mid-from-another-session"},
            }
        ]
        on_disk = older + [_row(i, f"m{i}") for i in range(SETTLED)]
        slot._disk_older_count = len(older)
        state.conversation_log.read_messages_chained = (  # type: ignore[method-assign]
            lambda _key: [dict(r) for r in on_disk]
        )
        data = await _get(state, f"?limit={SETTLED + len(older)}")
        assert data["messages"][0]["content"] == "FROM AN OLDER SESSION"
        assert data["messages"][-1]["content"] == "SELECTED VARIANT"


class TestASharedIdIsNotProofOfIdentity:
    """Counting an id once on each side does not make the two rows the SAME row.

    A chained disk read spans older sessions, so a caller that reuses a ``meta.mid``
    across two of them produces a disk row and a window row that are unrelated, each
    unique on its own side. Substituting there overwrites PERSISTED content with a
    live row from a different message, which is the corruption this class pins.

    The rows must corroborate first: same role, and the same ``ts`` when both carry
    one. Declining leaves the disk row exactly as it was read.
    """

    @pytest.mark.asyncio
    async def test_a_reused_id_from_an_older_session_is_not_overwritten(self, state):
        reused = "mid-reused"
        on_disk = [
            {
                "role": "assistant",
                "content": "OLD SESSION ANSWER",
                "cls": "msg msg-a",
                "ts": "2026-01-01T00:00:00Z",
                "meta": {"mid": reused},
            },
            {
                "role": "user",
                "content": "later question",
                "cls": "msg msg-u",
                "ts": "2026-06-01T00:00:00Z",
                "meta": {"mid": "mid-other"},
            },
        ]
        slot = _ChatSlot(key="chat-1")
        # The live window reuses the id for a DIFFERENT message, at a different ts.
        slot.messages = [
            {
                "role": "assistant",
                "content": "LIVE ANSWER",
                "cls": "msg msg-a",
                "ts": "2026-08-01T00:00:00Z",
                "meta": {"mid": reused},
            }
        ]
        slot._disk_older_count = 0
        slot._disk_window_len = 1
        slot._pending_rewrite = True
        state._slots["chat-1"] = slot
        state.conversation_log.read_messages_chained = (  # type: ignore[method-assign]
            lambda _key: [dict(r) for r in on_disk]
        )

        data = await _get(state, "?limit=10")

        contents = [m["content"] for m in data["messages"]]
        # The persisted row keeps its own content; the ts disagreement is the proof
        # that these are two different rows wearing one id.
        assert "OLD SESSION ANSWER" in contents, contents
        assert contents.count("LIVE ANSWER") <= 1, contents

    @pytest.mark.asyncio
    async def test_a_role_mismatch_at_one_id_declines(self, state):
        shared = "mid-shared"
        on_disk = [
            {
                "role": "user",
                "content": "DISK USER ROW",
                "cls": "msg msg-u",
                "meta": {"mid": shared},
            }
        ]
        slot = _ChatSlot(key="chat-1")
        slot.messages = [
            {
                "role": "assistant",
                "content": "LIVE ASSISTANT ROW",
                "cls": "msg msg-a",
                "meta": {"mid": shared},
            }
        ]
        slot._disk_older_count = 0
        slot._disk_window_len = 1
        slot._pending_rewrite = True
        state._slots["chat-1"] = slot
        state.conversation_log.read_messages_chained = (  # type: ignore[method-assign]
            lambda _key: [dict(r) for r in on_disk]
        )

        data = await _get(state, "?limit=10")

        contents = [m["content"] for m in data["messages"]]
        assert "DISK USER ROW" in contents, contents

    @pytest.mark.asyncio
    async def test_a_matching_ts_still_substitutes_so_the_variant_fix_survives(self, state):
        # The corroboration must not cost the case the overlay exists for: the SAME
        # row rewritten in place keeps its ts, so it still substitutes.
        same_ts = "2026-03-03T03:03:03Z"
        on_disk = [
            {
                "role": "assistant",
                "content": "STALE VARIANT",
                "cls": "msg msg-a",
                "ts": same_ts,
                "meta": {"mid": "mid-v"},
            }
        ]
        slot = _ChatSlot(key="chat-1")
        slot.messages = [
            {
                "role": "assistant",
                "content": "SELECTED VARIANT",
                "cls": "msg msg-a",
                "ts": same_ts,
                "meta": {"mid": "mid-v"},
            }
        ]
        slot._disk_older_count = 0
        slot._disk_window_len = 1
        slot._pending_rewrite = True
        state._slots["chat-1"] = slot
        state.conversation_log.read_messages_chained = (  # type: ignore[method-assign]
            lambda _key: [dict(r) for r in on_disk]
        )

        data = await _get(state, "?limit=10")

        contents = [m["content"] for m in data["messages"]]
        assert "SELECTED VARIANT" in contents, contents
        assert "STALE VARIANT" not in contents, contents


class TestAVariantSwitchRewritesTheTimestamp:
    """A differing ``ts`` must not by itself reject the case this overlay is for.

    ``api_chat_slot_switch_variant`` assigns ``target["ts"] = chosen["ts"]``, so the
    selected variant wears the VARIANT's timestamp, not the row's. If the inline save
    then fails, disk holds the old variant at the old stamp while the window holds the
    new one at a new stamp -- and a plain ``ts`` equality test would decline exactly
    here, repainting the content the user just switched away from.

    The row's own ``variants`` list is what settles it: a stamp found there is
    positive evidence the two rows are one message at different selections.
    """

    @pytest.mark.asyncio
    async def test_the_selected_variant_survives_a_ts_rewrite(self, state):
        old_ts = "2026-02-01T00:00:00Z"
        new_ts = "2026-02-02T00:00:00Z"
        on_disk = [
            {
                "role": "assistant",
                "content": "STALE VARIANT",
                "cls": "msg msg-a",
                "ts": old_ts,
                "meta": {"mid": "mid-v"},
            }
        ]
        slot = _ChatSlot(key="chat-1")
        # What a switch leaves behind: new content, the variant's OWN ts, and a
        # variants list that still records the stamp disk is sitting on.
        slot.messages = [
            {
                "role": "assistant",
                "content": "SELECTED VARIANT",
                "cls": "msg msg-a",
                "ts": new_ts,
                "variant_idx": 1,
                "variants": [
                    {"content": "STALE VARIANT", "ts": old_ts},
                    {"content": "SELECTED VARIANT", "ts": new_ts},
                ],
                "meta": {"mid": "mid-v"},
            }
        ]
        slot._disk_older_count = 0
        slot._disk_window_len = 1
        slot._pending_rewrite = True
        state._slots["chat-1"] = slot
        state.conversation_log.read_messages_chained = (  # type: ignore[method-assign]
            lambda _key: [dict(r) for r in on_disk]
        )

        data = await _get(state, "?limit=10")

        contents = [m["content"] for m in data["messages"]]
        assert "SELECTED VARIANT" in contents, contents
        assert "STALE VARIANT" not in contents, contents

    @pytest.mark.asyncio
    async def test_a_reused_id_is_still_refused_when_variants_do_not_account_for_it(self, state):
        # The variant exception must not become a blanket pass: a row whose variants
        # list exists but does NOT contain the disk stamp is still two rows sharing
        # an id, and the persisted content stays.
        on_disk = [
            {
                "role": "assistant",
                "content": "OLD SESSION ANSWER",
                "cls": "msg msg-a",
                "ts": "2026-01-01T00:00:00Z",
                "meta": {"mid": "mid-reused"},
            }
        ]
        slot = _ChatSlot(key="chat-1")
        slot.messages = [
            {
                "role": "assistant",
                "content": "LIVE ANSWER",
                "cls": "msg msg-a",
                "ts": "2026-08-01T00:00:00Z",
                "variant_idx": 0,
                "variants": [
                    {"content": "LIVE ANSWER", "ts": "2026-08-01T00:00:00Z"},
                    {"content": "ANOTHER LIVE TRY", "ts": "2026-08-02T00:00:00Z"},
                ],
                "meta": {"mid": "mid-reused"},
            }
        ]
        slot._disk_older_count = 0
        slot._disk_window_len = 1
        slot._pending_rewrite = True
        state._slots["chat-1"] = slot
        state.conversation_log.read_messages_chained = (  # type: ignore[method-assign]
            lambda _key: [dict(r) for r in on_disk]
        )

        data = await _get(state, "?limit=10")

        contents = [m["content"] for m in data["messages"]]
        assert "OLD SESSION ANSWER" in contents, contents

    @pytest.mark.asyncio
    async def test_the_disk_row_may_be_the_side_holding_the_variant_history(self, state):
        # Whichever store was written last holds the fuller list, so the check runs in
        # both directions.
        disk_ts = "2026-05-05T00:00:00Z"
        live_ts = "2026-05-06T00:00:00Z"
        on_disk = [
            {
                "role": "assistant",
                "content": "DISK VARIANT",
                "cls": "msg msg-a",
                "ts": disk_ts,
                "variants": [
                    {"content": "DISK VARIANT", "ts": disk_ts},
                    {"content": "WINDOW VARIANT", "ts": live_ts},
                ],
                "meta": {"mid": "mid-v2"},
            }
        ]
        slot = _ChatSlot(key="chat-1")
        slot.messages = [
            {
                "role": "assistant",
                "content": "WINDOW VARIANT",
                "cls": "msg msg-a",
                "ts": live_ts,
                "meta": {"mid": "mid-v2"},
            }
        ]
        slot._disk_older_count = 0
        slot._disk_window_len = 1
        slot._pending_rewrite = True
        state._slots["chat-1"] = slot
        state.conversation_log.read_messages_chained = (  # type: ignore[method-assign]
            lambda _key: [dict(r) for r in on_disk]
        )

        data = await _get(state, "?limit=10")

        contents = [m["content"] for m in data["messages"]]
        assert "WINDOW VARIANT" in contents, contents
        assert "DISK VARIANT" not in contents, contents


def _mid_row(mid: str, content: str, role: str = "assistant") -> dict:
    return {
        "role": role,
        "content": content,
        "cls": "msg msg-a" if role == "assistant" else "msg msg-u",
        "meta": {"mid": mid},
    }


def _pending_rewrite_slot(
    state: Any,
    on_disk: list[dict],
    window: list[dict],
    *,
    pending: bool = True,
    name: str = "chat-1",
) -> Any:
    """A slot whose window was truncated while disk still holds the old tail."""
    slot = _ChatSlot(key=name)
    slot.messages = [dict(r) for r in window]
    slot._disk_older_count = 0
    slot._disk_window_len = len(window)
    slot._pending_rewrite = pending
    state._slots[name] = slot
    state.conversation_log.read_messages_chained = (  # type: ignore[method-assign]
        lambda _key: [dict(r) for r in on_disk]
    )
    return slot


class TestAPendingRewriteRetiresTheTailItDeleted:
    """A failed rewrite save must not let the deleted turns come back.

    ``chat_regenerate`` truncates the window and only then saves. If that save
    fails, disk still holds the turns the window dropped -- and the overlay cannot
    remove them, because substitution by id deliberately preserves the row count.
    The drop settles LENGTH before the other two helpers reconcile CONTENT.

    The boundary is identity, never offset: ``_disk_older_count`` counts the current
    session's file while the corpus is a chained read spanning older sessions, so
    slicing at that counter would cut inside real history. These tests pin both the
    drop and the prefix it must never touch.
    """

    @pytest.mark.asyncio
    async def test_the_deleted_tail_does_not_reappear(self, state):
        on_disk = [_mid_row(f"mid-{i}", f"m{i}") for i in range(10)]
        window = [dict(r) for r in on_disk[:6]]
        _pending_rewrite_slot(state, on_disk, window)

        data = await _get(state, "?limit=50")

        contents = [m["content"] for m in data["messages"]]
        assert contents == ["m0", "m1", "m2", "m3", "m4", "m5"], contents
        assert data["total"] == 6, data["total"]

    @pytest.mark.asyncio
    async def test_older_session_rows_above_the_window_are_kept(self, state):
        # The regression an offset-based fix would cause: these rows are BELOW the
        # window in the chained read and share no id with it, so a slice at
        # `_disk_older_count` would be measured against the wrong file and delete
        # them. Identity keeps them.
        older = [_mid_row("old-0", "older-0"), _mid_row("old-1", "older-1")]
        current = [_mid_row(f"mid-{i}", f"m{i}") for i in range(6)]
        deleted_tail = [_mid_row("mid-6", "m6"), _mid_row("mid-7", "m7")]
        on_disk = older + current + deleted_tail
        _pending_rewrite_slot(state, on_disk, current)

        data = await _get(state, "?limit=50")

        contents = [m["content"] for m in data["messages"]]
        assert contents[:2] == ["older-0", "older-1"], contents
        assert "m6" not in contents and "m7" not in contents, contents
        assert contents == ["older-0", "older-1", "m0", "m1", "m2", "m3", "m4", "m5"], contents

    @pytest.mark.asyncio
    async def test_an_id_less_row_inside_the_window_region_is_kept(self, state):
        # Carrying an id is what marks a disk row as a flushed window row. A row
        # without one is history this helper has no standing to judge, so it passes
        # through -- a wrong drop would delete a durable turn.
        legacy = {"role": "user", "content": "LEGACY", "cls": "msg msg-u"}
        on_disk = [
            _mid_row("mid-0", "m0"),
            legacy,
            _mid_row("mid-1", "m1"),
            _mid_row("mid-6", "m6"),
        ]
        window = [_mid_row("mid-0", "m0"), _mid_row("mid-1", "m1")]
        _pending_rewrite_slot(state, on_disk, window)

        data = await _get(state, "?limit=50")

        contents = [m["content"] for m in data["messages"]]
        assert "LEGACY" in contents, contents
        assert "m6" not in contents, contents

    @pytest.mark.asyncio
    async def test_no_pending_rewrite_drops_nothing(self, state):
        # Without the flag the two stores are not expected to disagree on length,
        # and a disk row the window simply has not loaded is not a deleted row.
        on_disk = [_mid_row(f"mid-{i}", f"m{i}") for i in range(10)]
        window = [dict(r) for r in on_disk[:6]]
        _pending_rewrite_slot(state, on_disk, window, pending=False)

        data = await _get(state, "?limit=50")

        contents = [m["content"] for m in data["messages"]]
        assert "m9" in contents, contents
        assert len(contents) == 10, contents

    @pytest.mark.asyncio
    async def test_a_window_sharing_no_id_with_disk_drops_nothing(self, state):
        # No shared identity means no provable boundary, so the corpus is left alone
        # rather than guessed at.
        on_disk = [_mid_row(f"disk-{i}", f"d{i}") for i in range(4)]
        window = [_mid_row("live-0", "live-0")]
        _pending_rewrite_slot(state, on_disk, window)

        data = await _get(state, "?limit=50")

        contents = [m["content"] for m in data["messages"]]
        for i in range(4):
            assert f"d{i}" in contents, contents
