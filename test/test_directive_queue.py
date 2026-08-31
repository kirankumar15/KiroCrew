"""Provider-neutral session-directive delivery.

The marker path can only be trusted when the provider stamps the tool call with
``_meta.kiro`` identity. A backend that omits it leaves the forgery gate with no
trusted source, so the gate refuses every directive and the whole control plane
(loops, project changes, cards) fails closed. These cover the out-of-band path
that carries the validated payload to the gateway instead, and — the part that
matters most with several chat slots live at once — that a record can only ever
be claimed by the session it was published for.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web

from kiro_crew.dashboard import directive_queue
from kiro_crew.dashboard.handlers.sessions import api_session_directive


@pytest.fixture(autouse=True)
def _clean_queue():
    """Every test starts and ends with an empty store — the module holds process
    state, so a leaked record would make a later test pass for the wrong reason."""
    directive_queue.reset()
    yield
    directive_queue.reset()


class TestPublishAndClaim:
    def test_a_published_directive_is_claimable(self):
        directive_queue.publish("sess-a", "monitor_start", {"message": "check CI"})
        claimed = directive_queue.claim("sess-a")
        assert len(claimed) == 1
        assert claimed[0]["kind"] == "monitor_start"
        assert claimed[0]["args"] == {"message": "check CI"}

    def test_claim_is_single_consume(self):
        """Two consumers racing one session must not both apply the same record —
        that is two armed loops from one request."""
        directive_queue.publish("sess-a", "monitor_start", {"message": "x"})
        assert len(directive_queue.claim("sess-a")) == 1
        assert directive_queue.claim("sess-a") == []

    def test_claim_of_an_unknown_session_is_empty_not_an_error(self):
        assert directive_queue.claim("never-published") == []

    def test_unknown_kind_is_refused(self):
        """The only legitimate publishers are Kiro Crew's own directive tools, so
        an unrecognized kind means the request did not come from one."""
        with pytest.raises(ValueError, match="unknown directive kind"):
            directive_queue.publish("sess-a", "rm_rf_everything", {})

    def test_empty_session_key_is_refused(self):
        with pytest.raises(ValueError, match="session_key required"):
            directive_queue.publish("", "monitor_start", {})

    def test_args_are_copied_not_aliased(self):
        """The publisher's dict must not stay live inside the store: a later
        mutation would change what gets applied."""
        args = {"message": "original"}
        directive_queue.publish("sess-a", "monitor_start", args)
        args["message"] = "mutated after publish"
        assert directive_queue.claim("sess-a")[0]["args"]["message"] == "original"


class TestSessionIsolation:
    """The concurrency case: several chat slots arming at once.

    Records are keyed by the session that published them, and a directive only
    ever affects the session that claims it — so nothing can land in the wrong
    slot. Nothing here is shared between the two sessions, which is the point.
    """

    def test_two_sessions_do_not_see_each_others_records(self):
        directive_queue.publish("slot-a", "monitor_start", {"message": "for A"})
        directive_queue.publish("slot-b", "monitor_start", {"message": "for B"})

        claimed_a = directive_queue.claim("slot-a")
        claimed_b = directive_queue.claim("slot-b")

        assert [r["args"]["message"] for r in claimed_a] == ["for A"]
        assert [r["args"]["message"] for r in claimed_b] == ["for B"]

    def test_one_session_claiming_does_not_drain_another(self):
        directive_queue.publish("slot-a", "monitor_start", {"message": "for A"})
        directive_queue.publish("slot-b", "suggest_followup", {"items": []})

        directive_queue.claim("slot-a")

        assert directive_queue.depth("slot-b") == 1

    def test_discard_is_scoped_to_one_session(self):
        directive_queue.publish("slot-a", "monitor_start", {"message": "a"})
        directive_queue.publish("slot-b", "monitor_start", {"message": "b"})

        assert directive_queue.discard("slot-a") == 1

        assert directive_queue.claim("slot-a") == []
        assert len(directive_queue.claim("slot-b")) == 1

    def test_several_directives_for_one_session_are_claimed_in_order(self):
        """One turn may emit more than one directive (arm a loop, then offer a
        card). Order is the order they were requested."""
        directive_queue.publish("slot-a", "monitor_start", {"message": "first"})
        directive_queue.publish("slot-a", "suggest_followup", {"items": []})
        kinds = [r["kind"] for r in directive_queue.claim("slot-a")]
        assert kinds == ["monitor_start", "suggest_followup"]


class TestDiscard:
    def test_discard_drops_without_applying(self):
        """The kiro-cli path applies from the verified marker; the out-of-band
        twin must be retired or the effect lands twice."""
        directive_queue.publish("sess-a", "monitor_start", {"message": "x"})
        assert directive_queue.discard("sess-a") == 1
        assert directive_queue.claim("sess-a") == []

    def test_discard_of_nothing_is_zero(self):
        assert directive_queue.discard("sess-a") == 0

    def test_discard_of_empty_key_is_zero(self):
        assert directive_queue.discard("") == 0


class TestBounds:
    def test_the_queue_is_capped_and_keeps_the_newest(self):
        """An unclaimed queue must not grow without limit, and a live session's
        most recent intent is the one worth keeping."""
        for i in range(directive_queue.MAX_PER_SESSION + 3):
            directive_queue.publish("sess-a", "monitor_start", {"message": f"m{i}"})

        claimed = directive_queue.claim("sess-a")
        assert len(claimed) == directive_queue.MAX_PER_SESSION
        # The three oldest were dropped, so the first survivor is m3.
        assert claimed[0]["args"]["message"] == "m3"
        assert claimed[-1]["args"]["message"] == f"m{directive_queue.MAX_PER_SESSION + 2}"

    def test_a_stale_record_is_dropped_rather_than_applied(self):
        """A directive belongs to the turn that asked for it. Applying one long
        after that turn ended would arm a loop nobody is waiting on."""
        directive_queue.publish("sess-a", "monitor_start", {"message": "x"})
        with patch.object(
            directive_queue.time,
            "monotonic",
            return_value=time.monotonic() + directive_queue.MAX_AGE_SECS + 1,
        ):
            assert directive_queue.claim("sess-a") == []

    def test_a_fresh_record_survives_the_age_check(self):
        directive_queue.publish("sess-a", "monitor_start", {"message": "x"})
        with patch.object(
            directive_queue.time,
            "monotonic",
            return_value=time.monotonic() + (directive_queue.MAX_AGE_SECS / 2),
        ):
            assert len(directive_queue.claim("sess-a")) == 1


def _request(headers: dict, body: object, can_read: bool = True, local: bool = True):
    """A request double for the endpoint.

    ``local`` models what the auth middleware stamped on the request: an
    internal-secret / kernel-verified peer caller (True) versus a
    cookie-authenticated browser caller on a ``local_only=False`` deployment
    (False). The ``.get`` mapping is backed by a REAL dict on purpose — a bare
    ``MagicMock.get`` returns a truthy mock, which would silently satisfy the
    handler's locality re-assert and make every test here vacuous.
    """
    req = MagicMock(spec=web.Request)
    req.headers = headers
    req.can_read_body = can_read
    req.json = AsyncMock(return_value=body)
    req.app = {"state": MagicMock()}
    _scope: dict = {"internal_auth": True} if local else {}
    req.get = _scope.get
    return req


class TestEndpoint:
    @pytest.mark.asyncio
    async def test_happy_path_parks_the_record_for_the_declared_session(self):
        resp = await api_session_directive(
            _request(
                {"X-Session-Key": "slot-a"},
                {"kind": "monitor_start", "args": {"message": "go"}},
            )
        )
        assert resp.status == 200
        claimed = directive_queue.claim("slot-a")
        assert len(claimed) == 1
        assert claimed[0]["args"] == {"message": "go"}

    @pytest.mark.asyncio
    async def test_missing_session_key_is_400_with_a_code(self):
        resp = await api_session_directive(_request({}, {"kind": "monitor_start", "args": {}}))
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_unknown_kind_is_400_and_parks_nothing(self):
        resp = await api_session_directive(
            _request({"X-Session-Key": "slot-a"}, {"kind": "not_a_directive"})
        )
        assert resp.status == 400
        assert directive_queue.depth("slot-a") == 0

    @pytest.mark.asyncio
    async def test_cookie_caller_cannot_park_a_directive_for_a_chosen_session(self):
        """The locality re-assert. On a ``local_only=False`` deployment the auth
        middleware admits a cookie/token caller onto this strict route, and such
        a caller picks its own ``X-Session-Key`` — which would be a cross-session
        mutation once the named turn's consumer applies the record. 403, and the
        queue must stay empty."""
        resp = await api_session_directive(
            _request(
                {"X-Session-Key": "victim-slot"},
                {"kind": "monitor_start", "args": {"message": "go"}},
                local=False,
            )
        )
        assert resp.status == 403
        assert directive_queue.depth("victim-slot") == 0

    @pytest.mark.asyncio
    async def test_kernel_verified_peer_without_the_secret_is_accepted(self):
        """``peer_verified`` alone is enough: the kernel attested the AF_UNIX
        peer's ancestry resolves to the DECLARED key, which is stronger evidence
        for this route than the shared secret."""
        req = _request(
            {"X-Session-Key": "slot-a"},
            {"kind": "monitor_start", "args": {"message": "go"}},
            local=False,
        )
        req.get = {"peer_verified": True}.get
        resp = await api_session_directive(req)
        assert resp.status == 200
        assert directive_queue.depth("slot-a") == 1

    @pytest.mark.asyncio
    async def test_malformed_body_is_400_and_parks_nothing(self):
        req = _request({"X-Session-Key": "slot-a"}, None)
        req.json = AsyncMock(side_effect=ValueError("not json"))
        resp = await api_session_directive(req)
        assert resp.status == 400
        assert directive_queue.depth("slot-a") == 0

    @pytest.mark.asyncio
    async def test_non_dict_args_degrade_to_empty_not_a_crash(self):
        resp = await api_session_directive(
            _request(
                {"X-Session-Key": "slot-a"},
                {"kind": "reset_conversation", "args": "not-a-dict"},
            )
        )
        assert resp.status == 200
        assert directive_queue.claim("slot-a")[0]["args"] == {}


class TestEmitHelperPublishes:
    """``control._emit_directive`` must park the payload as well as return the
    marker — the marker alone is what a provider-less backend cannot use."""

    def test_emit_publishes_and_still_returns_the_marker(self):
        from kiro_crew.mcp_tools import control

        posted: list[tuple] = []

        with patch.object(
            control.mcp_core, "_post", side_effect=lambda p, b: posted.append((p, b))
        ):
            out = control._emit_directive("monitor_start", {"message": "hi"}, "Monitor requested.")

        assert posted == [
            ("/api/session-directive", {"kind": "monitor_start", "args": {"message": "hi"}})
        ]
        from kiro_crew import session_directive

        assert session_directive.has_marker(out)

    def test_a_refused_oversized_directive_is_never_published(self):
        """``encode`` refuses an over-limit payload and tells the model nothing was
        applied. Publishing anyway would leave a record that contradicts that."""
        from kiro_crew import session_directive
        from kiro_crew.mcp_tools import control

        posted: list[tuple] = []
        huge = {"message": "x" * (session_directive.MAX_DIRECTIVE_CHARS + 100)}

        with patch.object(
            control.mcp_core, "_post", side_effect=lambda p, b: posted.append((p, b))
        ):
            out = control._emit_directive("monitor_start", huge, "Monitor requested.")

        assert session_directive.is_refusal(out)
        assert posted == []

    def test_a_publish_failure_does_not_break_the_tool(self):
        """An older gateway with no such route, or one that is down, must not turn
        a working tool call into an error — the marker path may still work."""
        from kiro_crew import session_directive
        from kiro_crew.mcp_tools import control

        with patch.object(control.mcp_core, "_post", side_effect=RuntimeError("gateway down")):
            out = control._emit_directive("monitor_start", {"message": "hi"}, "Monitor requested.")

        assert session_directive.has_marker(out)
