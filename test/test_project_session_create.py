from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from chat_test_helpers import _make_state

from kiro_crew.dashboard import chat_runner
from kiro_crew.dashboard.chat import api_chat_slot_create
from kiro_crew.dashboard.state import _ChatSlot
from kiro_crew.project_manifest import create_project_manifest
from kiro_crew.project_registry import ProjectRegistry


def _app(state, registry: ProjectRegistry, *, owner: bool = True) -> web.Application:
    @web.middleware
    async def identity(request: web.Request, handler):
        request["app"] = ""
        request["user"] = "local-app" if owner else "guest"
        return await handler(request)

    app = web.Application(middlewares=[identity])
    app["state"] = state
    app["project_registry"] = registry
    app.router.add_post("/api/chat/slots", api_chat_slot_create)
    return app


def _record_slot_frames(state) -> list[list[dict]]:
    frames: list[list[dict]] = []

    def capture(payload):
        if payload.get("_type") == "slots":
            frames.append(json.loads(payload["slots"]))

    state._broadcast = capture
    return frames


@pytest.mark.asyncio
async def test_create_attaches_project_before_first_broadcast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
    bundle = tmp_path / "payments"
    manifest = create_project_manifest(bundle, name="Payments")
    registry = ProjectRegistry(tmp_path / "projects")
    registry.add_local(bundle)
    state = _make_state(tmp_path / "sessions")
    frames = _record_slot_frames(state)

    async with TestClient(TestServer(_app(state, registry))) as client:
        response = await client.post(
            "/api/chat/slots", json={"name": "payments-chat", "project_id": manifest.id}
        )
        assert response.status == 200
        payload = await response.json()
    assert payload["project_id"] == manifest.id
    assert payload["project"] == str(bundle.resolve())
    assert len(frames) == 1
    announced = next(slot for slot in frames[0] if slot["key"] == "payments-chat")
    assert announced["project_id"] == manifest.id
    assert announced["project"] == str(bundle.resolve())
    assert "Project: Payments" in state._slots["payments-chat"]._project_brief


@pytest.mark.asyncio
async def test_create_rejects_unknown_project_without_creating_slot(tmp_path: Path) -> None:
    state = _make_state(tmp_path / "sessions")
    registry = ProjectRegistry(tmp_path / "projects")

    async with TestClient(TestServer(_app(state, registry))) as client:
        response = await client.post(
            "/api/chat/slots",
            json={
                "name": "orphan",
                "project_id": "018f4f4a-760f-7a8b-a5d4-5a7e0f130d4e",
            },
        )
        assert response.status == 404
        assert (await response.json())["code"] == "project_not_found"
    assert "orphan" not in state._slots


@pytest.mark.asyncio
async def test_create_project_attachment_is_owner_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "payments"
    manifest = create_project_manifest(bundle, name="Payments")
    registry = ProjectRegistry(tmp_path / "projects")
    registry.add_local(bundle)
    state = _make_state(tmp_path / "sessions")
    events: list[dict[str, object]] = []
    monkeypatch.setattr(
        "kiro_crew.dashboard.chat_handlers.sel",
        lambda: SimpleNamespace(log_api_access=lambda **event: events.append(event)),
    )

    async with TestClient(TestServer(_app(state, registry, owner=False))) as client:
        response = await client.post(
            "/api/chat/slots", json={"name": "forbidden", "project_id": manifest.id}
        )
        assert response.status == 403
        assert (await response.json())["code"] == "owner_only"
    assert "forbidden" not in state._slots
    assert events == [
        {
            "caller": "guest",
            "operation": "project_attach",
            "outcome": "denied",
            "source": "dashboard",
            "resources": "non_owner_block",
        }
    ]


@pytest.mark.asyncio
async def test_create_project_attachment_fails_closed_when_audit_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "payments"
    manifest = create_project_manifest(bundle, name="Payments")
    registry = ProjectRegistry(tmp_path / "projects")
    registry.add_local(bundle)
    state = _make_state(tmp_path / "sessions")
    attachment_resolutions: list[str] = []

    from kiro_crew.project_sessions import resolve_project_attachment as real_resolve

    def track_resolution(*args, **kwargs):
        attachment_resolutions.append(str(args[0]))
        return real_resolve(*args, **kwargs)

    monkeypatch.setattr(
        "kiro_crew.dashboard.chat_handlers.resolve_project_attachment",
        track_resolution,
    )
    monkeypatch.setattr(
        "kiro_crew.dashboard.chat_handlers.sel",
        lambda: SimpleNamespace(
            log_api_access=lambda **_event: (_ for _ in ()).throw(OSError("audit unavailable"))
        ),
    )

    async with TestClient(TestServer(_app(state, registry))) as client:
        response = await client.post(
            "/api/chat/slots", json={"name": "unaudited", "project_id": manifest.id}
        )
        assert response.status == 503
        assert (await response.json())["code"] == "project_audit_unavailable"
    assert "unaudited" not in state._slots
    assert attachment_resolutions == []


@pytest.mark.asyncio
async def test_create_cannot_attach_project_to_existing_conversation(tmp_path: Path) -> None:
    bundle = tmp_path / "payments"
    manifest = create_project_manifest(bundle, name="Payments")
    registry = ProjectRegistry(tmp_path / "projects")
    registry.add_local(bundle)
    state = _make_state(tmp_path / "sessions")
    existing = state.get_or_create_slot("existing")
    existing.append("user", "Already working", broadcast=False)

    async with TestClient(TestServer(_app(state, registry))) as client:
        response = await client.post(
            "/api/chat/slots", json={"name": "existing", "project_id": manifest.id}
        )
        assert response.status == 409
        assert (await response.json())["code"] == "project_rebind_requires_new_session"
    assert existing.project_id == ""


@pytest.mark.asyncio
async def test_create_rejects_non_string_project_id(tmp_path: Path) -> None:
    state = _make_state(tmp_path / "sessions")
    registry = ProjectRegistry(tmp_path / "projects")

    async with TestClient(TestServer(_app(state, registry))) as client:
        response = await client.post("/api/chat/slots", json={"name": "invalid", "project_id": 0})
        assert response.status == 400
        assert (await response.json())["code"] == "project_invalid_request"
    assert "invalid" not in state._slots


@pytest.mark.asyncio
async def test_restored_project_attachment_refreshes_workspace_and_brief_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slot = _ChatSlot("restored")
    slot.project_id = "018f4f4a-760f-7a8b-a5d4-5a7e0f130d4e"
    slot.project = "/stale/workspace"
    workspace = tmp_path / "current-workspace"
    attachment = SimpleNamespace(workspace_dir=workspace, brief="Current Project brief")
    monkeypatch.setattr(
        "kiro_crew.project_sessions.resolve_project_attachment",
        lambda _project_id: attachment,
    )

    await chat_runner._refresh_project_attachment(slot)

    assert slot.project == str(workspace)
    assert slot._project_brief == "Current Project brief"


def test_restored_project_attachment_refresh_precedes_session_binding() -> None:
    source = inspect.getsource(chat_runner._run_chat)

    assert source.index("await _refresh_project_attachment(slot)") < source.index(
        "state.sessions.get_or_create("
    )
