"""The sources.sync_status COLUMN is the single source of truth.

A source's sync state used to live in two places: the ``sources.sync_status``
column and a ``sync_status`` key inside the ``properties`` JSON blob. Writers
were split across the two -- most transitions wrote the column only, while the
watcher's 'missing' marker went into the blob only -- and readers were split the
same way, so each side saw a state the other had not written.

These tests pin the converged contract on the watcher paths that read and write
it: the pre-scan skip reads the column, and the missing marker is written to the
column and can be left again.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from kiro_crew.knowledge.store import KnowledgeStore
from kiro_crew.knowledge.watcher import KnowledgeWatcher


@pytest.fixture()
def store(tmp_path):
    s = KnowledgeStore(str(tmp_path / "knowledge.db"))
    yield s
    s.close()


def _watcher(store) -> KnowledgeWatcher:
    pipeline = MagicMock()
    # No embedder configured -> _scan skips the self-heal re-embed branch.
    pipeline.embedder = None
    watcher = KnowledgeWatcher(store=store, pipeline=pipeline)
    # Discovery registers workspace folders from live config; irrelevant here
    # and it would put rows in the table the assertions do not expect.
    watcher._discover_drop_folder = AsyncMock()  # type: ignore[method-assign]
    watcher._discover_project_docs = AsyncMock()  # type: ignore[method-assign]
    watcher._maybe_reembed_stale = AsyncMock()  # type: ignore[method-assign]
    return watcher


def _status(store, sid: str) -> str:
    return store.db.execute("SELECT sync_status FROM sources WHERE id = ?", (sid,)).fetchone()[
        "sync_status"
    ]


def _props(store, sid: str) -> dict:
    raw = store.db.execute("SELECT properties FROM sources WHERE id = ?", (sid,)).fetchone()[
        "properties"
    ]
    return json.loads(raw or "{}")


class TestFolderPreScanSkip:
    @pytest.mark.asyncio
    async def test_a_paused_folder_is_not_walked(self, store, tmp_path):
        """A pause recorded in the column stops the sweep.

        The skip used to read the properties copy, so a pause the column knew
        about still walked and delete-reconciled the whole folder every sweep.
        """
        folder = tmp_path / "vault"
        folder.mkdir()
        sid = store.add_source("vault", "local_folder", str(folder))
        store.db.execute("UPDATE sources SET sync_status = 'paused' WHERE id = ?", (sid,))
        store.db.commit()
        assert "sync_status" not in _props(store, sid), "the JSON copy must not exist"

        watcher = _watcher(store)
        scan = AsyncMock(return_value={})
        watcher._folder_watcher.scan_source = scan  # type: ignore[method-assign]
        await watcher._scan()

        scan.assert_not_called()

    @pytest.mark.asyncio
    async def test_an_unconfirmed_folder_is_not_walked(self, store, tmp_path):
        folder = tmp_path / "vault"
        folder.mkdir()
        sid = store.add_source(
            "vault", "local_folder", str(folder), properties={"sync_status": "pending_confirmation"}
        )
        assert _status(store, sid) == "pending_confirmation"

        watcher = _watcher(store)
        scan = AsyncMock(return_value={})
        watcher._folder_watcher.scan_source = scan  # type: ignore[method-assign]
        await watcher._scan()

        scan.assert_not_called()

    @pytest.mark.asyncio
    async def test_an_active_folder_is_still_walked(self, store, tmp_path):
        """The guard rejects only the two states, so 'active' still scans."""
        folder = tmp_path / "vault"
        folder.mkdir()
        store.add_source("vault", "local_folder", str(folder), properties={"sync_status": "active"})

        watcher = _watcher(store)
        scan = AsyncMock(return_value={})
        watcher._folder_watcher.scan_source = scan  # type: ignore[method-assign]
        await watcher._scan()

        scan.assert_called_once()


class TestSingleFileMissingMarker:
    @pytest.mark.asyncio
    async def test_a_vanished_file_marks_the_column_missing(self, store, tmp_path):
        """The Library renders the column, so that is where 'missing' belongs.

        Marking it in the properties JSON instead left the visible state stale:
        a file that was gone went on reading 'synced'.
        """
        gone = tmp_path / "gone.md"
        gone.write_text("# gone")
        sid = store.add_source("gone.md", "local_file", str(gone))
        store.db.execute("UPDATE sources SET sync_status = 'synced' WHERE id = ?", (sid,))
        store.db.commit()
        gone.unlink()

        await _watcher(store)._scan()

        assert _status(store, sid) == "missing"
        assert "sync_status" not in _props(store, sid), "no second copy is written"

    @pytest.mark.asyncio
    async def test_a_returning_file_leaves_missing(self, store, tmp_path):
        """'missing' must be a state a source can leave.

        An unchanged file is not re-ingested, so nothing else moves the column
        back and the source would read missing for as long as it existed.
        """
        back = tmp_path / "back.md"
        back.write_text("# back")
        # A stored mtime in the future keeps the change branch out of it: this
        # test is about the recovery write, not about re-ingestion.
        sid = store.add_source(
            "back.md", "local_file", str(back), properties={"mtime": 1 << 40, "content_hash": "abc"}
        )
        store.db.execute("UPDATE sources SET sync_status = 'missing' WHERE id = ?", (sid,))
        store.db.commit()

        await _watcher(store)._scan()

        assert _status(store, sid) == "synced"

    @pytest.mark.asyncio
    async def test_a_present_file_keeps_its_status(self, store, tmp_path):
        """The recovery write fires only for a row that reads 'missing'."""
        here = tmp_path / "here.md"
        here.write_text("# here")
        sid = store.add_source(
            "here.md", "local_file", str(here), properties={"mtime": 1 << 40, "content_hash": "abc"}
        )
        store.db.execute("UPDATE sources SET sync_status = 'error' WHERE id = ?", (sid,))
        store.db.commit()

        await _watcher(store)._scan()

        assert _status(store, sid) == "error"

    @pytest.mark.asyncio
    async def test_recovery_does_not_overwrite_a_status_that_moved(self, store, tmp_path):
        """The recovery write loses to a transition that landed mid-sweep.

        'missing' comes from the snapshot taken at the top of the sweep. A manual
        sync that fails while the sweep runs writes 'error', and stamping
        'synced' over it would report content the store does not have.
        """
        back = tmp_path / "raced.md"
        back.write_text("# raced")
        sid = store.add_source(
            "raced.md",
            "local_file",
            str(back),
            properties={"mtime": 1 << 40, "content_hash": "abc"},
        )
        store.db.execute("UPDATE sources SET sync_status = 'missing' WHERE id = ?", (sid,))
        store.db.commit()

        real_update = store.update_source

        def a_manual_sync_fails_first(source_id, **fields):
            if fields.get("sync_status") == "synced":
                store.db.execute(
                    "UPDATE sources SET sync_status = 'error' WHERE id = ?", (source_id,)
                )
                store.db.commit()
            return real_update(source_id, **fields)

        store.update_source = a_manual_sync_fails_first  # type: ignore[method-assign]
        await _watcher(store)._scan()

        assert _status(store, sid) == "error"

    @pytest.mark.asyncio
    async def test_a_failed_reingest_is_not_recorded_as_synced(self, store, tmp_path):
        """'synced' is a claim about content, so it waits for the re-ingest.

        A returning file whose content CHANGED is re-ingested, and the pipeline
        writes the column itself. Clearing 'missing' to 'synced' before that read
        would leave the claim standing when the read fails -- the source would
        report holding content it never ingested.
        """
        back = tmp_path / "changed.md"
        back.write_text("# changed")
        sid = store.add_source(
            "changed.md", "local_file", str(back), properties={"mtime": 1, "content_hash": "stale"}
        )
        store.db.execute("UPDATE sources SET sync_status = 'missing' WHERE id = ?", (sid,))
        store.db.commit()

        watcher = _watcher(store)
        watcher.pipeline.ingest_file = AsyncMock(side_effect=RuntimeError("read failed"))
        await watcher._scan()

        watcher.pipeline.ingest_file.assert_awaited_once()
        assert _status(store, sid) == "missing"
