"""The MDM-managed tier, and the tighten-only composition rule beneath it.

Covers the top precedence tier added to ``load_security_policy`` and the
composition primitive every lower tier now goes through:

* the tier is **inert** with no managed file, so every standalone install is
  unaffected (this is the property that makes the tier free to ship);
* the trust checks on a *present* managed file -- regular file, root-owned, not
  group/world-writable, not a symlink, size-bounded -- and the fact that each
  failure **raises** instead of falling through to a lower, possibly permissive
  tier, which is the entire security property;
* both document encodings the tier accepts (macOS ``.plist``, JSON elsewhere);
* ``_intersect_ceilings``: a subordinate may add restrictions and may not remove
  one, and everything outside ``controls`` stays the authority's;
* ``BreakGlass``: the dated, authority-issued grant that is the ONLY way a lower
  tier outranks the authority.

**Nothing here touches a real managed path.** The autouse fixture points
``_managed_policy_path`` and ``_policy_home_path`` at nonexistent files under
``tmp_path`` before every test, so a dev machine that happens to have
``/etc/kirocrew/security_policy.json`` cannot make an assertion here read a
document this module never wrote. There is deliberately no environment override
for the managed path, so the function seam is the only way to aim it.
"""

from __future__ import annotations

import json
import os
import plistlib
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from kiro_crew import platform_compat
from kiro_crew.platform import governance
from kiro_crew.platform import governance_health as health
from kiro_crew.platform import governance_profiles as gp
from kiro_crew.platform.context import PlatformCompositionError
from kiro_crew.platform.governance import (
    BREAK_GLASS_TIERS,
    SIGNATURE_UNSIGNED,
    SIGNATURE_UNVERIFIED,
    TIER_BUNDLED,
    TIER_CENTRAL,
    TIER_ENV,
    TIER_HOME,
    TIER_MANAGED,
    BreakGlass,
    load_security_policy,
    resolve,
    resolve_ordinal,
)

_POLICY_ENV = "KIROCREW_SECURITY_POLICY"

#: The real path resolver, captured at import time -- BEFORE the autouse fixture
#: replaces the module attribute. The "no env override" test needs the genuine
#: function, and there is no other way back to it once the seam is patched.
_REAL_MANAGED_POLICY_PATH = governance._managed_policy_path

#: The uid check only runs on POSIX -- off POSIX ``_assert_managed_file_trusted``
#: returns early and documents that the ACL is the OS's to enforce. Tests whose
#: subject IS the uid comparison on a real file are therefore POSIX-only; the
#: faked-stat tests below force ``IS_POSIX`` so the mode-bit branch is exercised
#: on a Windows runner too, and no platform's answer is left untested.
_POSIX_ONLY = pytest.mark.skipif(
    not hasattr(os, "getuid"), reason="uid ownership semantics; os.getuid is POSIX-only"
)

#: A managed file under ``tmp_path`` is owned by whoever runs pytest, and a test
#: cannot chown to root -- so the "owned by a non-root uid" case is natural here
#: and needs no patching at all, which is why those tests exercise the real
#: ``os.fstat``. Unless the suite runs AS root, where the file would legitimately
#: pass the check.
_NOT_ROOT = pytest.mark.skipif(
    hasattr(os, "getuid") and os.getuid() == 0,
    reason="running as root makes a tmp_path file legitimately root-owned",
)


# ──────────────────────────────────────────────────────────────────────────
# Helpers -- same document-building idiom as test_governance_policy.py /
# test_governance_distribution.py: a minimal valid body, tagged by
# ``identity.issuer`` so a precedence assertion can name the document that won
# without parsing a control out of it.
# ──────────────────────────────────────────────────────────────────────────


def _doc(marker: str = "", **extra: object) -> dict:
    body: dict = {"version": 1, "boot": {"fail_closed": True}}
    if marker:
        body["identity"] = {"issuer": marker}
    body.update(extra)
    return body


def _write_policy(path: Path, marker: str = "", **extra: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_doc(marker, **extra)), encoding="utf-8")
    return path


def _write_plist_policy(path: Path, marker: str = "", **extra: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        plistlib.dump(_doc(marker, **extra), handle)
    return path


def _sized_policy_bytes(total: int) -> bytes:
    """JSON for a valid policy whose encoded length is EXACTLY *total* bytes.

    Padding rides on ``identity.issuer`` rather than a filler top-level key
    because ``parse_policy`` is fail-closed on an unknown top-level scope, so a
    filler key would be refused for the wrong reason.
    """
    body = _doc("x")
    pad = total - len(json.dumps(body).encode("utf-8"))
    assert pad >= 0, "requested size is below the minimum valid document"
    body["identity"]["issuer"] = "x" * (1 + pad)
    raw = json.dumps(body).encode("utf-8")
    assert len(raw) == total
    return raw


def _future(days: int = 7) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()


def _past(days: int = 7) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def _point_managed(monkeypatch, path: object) -> None:
    monkeypatch.setattr(governance, "_managed_policy_path", lambda: path)


def _point_home(monkeypatch, path: Path) -> None:
    monkeypatch.setattr(governance, "_policy_home_path", lambda: path)


def _fake_managed_stat(
    monkeypatch, *, uid: int = 0, extra_mode: int = 0, perm: int = None
) -> None:
    """Report a doctored ``os.fstat`` for the managed file's descriptor.

    Needed because a test cannot create a root-owned file: without this the uid
    check fires first and the mode-bit branch below it is unreachable. The fake
    delegates to the real ``os.fstat`` and rewrites only the fields under
    test, so ``S_ISREG`` and every other predicate still answer about the real
    file rather than about a hand-built tuple.

    ``perm`` REPLACES the permission bits (the file type is preserved, so
    ``S_ISREG`` is unaffected). Pass it whenever the assertion depends on the
    permissions being exactly something: ``extra_mode`` alone ORs onto whatever the
    filesystem reports, and Windows honours only the read-only bit, so a
    ``chmod(0o644)`` there leaves 0o666 behind and a "no group write" case would
    fail on the platform rather than on the code.
    """
    real_fstat = os.fstat

    def fake(fd: int) -> os.stat_result:
        st = real_fstat(fd)
        mode = st.st_mode
        if perm is not None:
            mode = stat.S_IFMT(mode) | perm
        return os.stat_result(
            (
                mode | extra_mode,
                st.st_ino,
                st.st_dev,
                st.st_nlink,
                uid,
                st.st_gid,
                st.st_size,
                int(st.st_atime),
                int(st.st_mtime),
                int(st.st_ctime),
            )
        )

    monkeypatch.setattr(os, "fstat", fake)
    # The mode-bit branch sits behind the POSIX gate, so the gate must be on for
    # the branch to be reachable at all on a Windows runner.
    monkeypatch.setattr(platform_compat, "IS_POSIX", True)


class _ExplodingPath(type(Path())):  # type: ignore[misc]
    """A path whose ``exists()`` raises OSError, as an unreadable parent would."""

    def exists(self, *args: object, **kwargs: object) -> bool:  # type: ignore[override]
        raise OSError("permission denied")


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _hermetic_governance_globals(monkeypatch, tmp_path):
    """Pin every process global and env var this module's subject reads.

    The two path seams are aimed at nonexistent files under ``tmp_path`` so the
    tier starts INERT in every test and a real ``/etc/kirocrew`` (or a developer's
    own home policy) can never be the document an assertion here reads. The env
    vars are deleted because each selects a tier: one left set by a CI image would
    make every precedence assertion read a file this module never wrote.
    ``governance_health`` and ``governance_profiles`` keep worker-lifetime state,
    so both are reset on the way in and on the way out.
    """
    for var in (
        _POLICY_ENV,
        "KIROCREW_ADMISSION_POLICY",
        "KIROCREW_POLICY_URL",
        "KIROCREW_POLICY_HEADERS",
        "KIROCREW_POLICY_CACHE_ONLY",
    ):
        monkeypatch.delenv(var, raising=False)
    _point_managed(monkeypatch, tmp_path / "absent" / "managed.json")
    _point_home(monkeypatch, tmp_path / "absent" / "home.json")
    health.reset()
    gp.reset_store()
    yield
    health.reset()
    gp.reset_store()


@pytest.fixture
def trusted_managed(monkeypatch):
    """Treat the managed file as trusted, for tests whose subject is NOT the guard.

    A test cannot create a root-owned file, so every happy-path assertion about
    the tier's PRECEDENCE would otherwise refuse for a reason it is not testing.
    The guard itself is exercised for real -- unpatched -- by
    ``TestManagedFileTrustIsChecked``.
    """
    monkeypatch.setattr(governance, "_assert_managed_file_trusted", lambda fd, path: None)


# ──────────────────────────────────────────────────────────────────────────
# (a) Inert with no managed file
# ──────────────────────────────────────────────────────────────────────────
class TestTheTierIsInertByDefault:
    def test_absent_managed_file_reads_as_none(self):
        assert governance._read_managed_policy() is None

    def test_unknown_platform_has_no_managed_path_and_stays_inert(self, monkeypatch):
        # ``_managed_policy_path`` answers None on a platform with no managed
        # channel; the reader must treat that as inert, not as an error.
        _point_managed(monkeypatch, None)
        assert governance._read_managed_policy() is None

    def test_standalone_home_policy_still_governs_with_no_managed_file(self, monkeypatch, tmp_path):
        _point_home(monkeypatch, _write_policy(tmp_path / "home.json", "operator"))
        ceiling = load_security_policy()
        assert ceiling is not None
        assert ceiling.tier == TIER_HOME
        assert ceiling.identity_issuer == "operator"

    def test_no_policy_at_any_tier_is_still_ungoverned(self):
        assert load_security_policy() is None

    def test_an_absent_managed_file_leaves_break_glass_empty(self, monkeypatch, tmp_path):
        _point_home(monkeypatch, _write_policy(tmp_path / "home.json", "operator"))
        ceiling = load_security_policy()
        assert ceiling is not None
        assert ceiling.break_glass == BreakGlass()
        assert ceiling.break_glass.summary() == "no break-glass grant"

    def test_an_unreadable_parent_directory_leaves_the_tier_inert(self, monkeypatch, tmp_path):
        # A permissions quirk on a path nobody configured is not a managed policy,
        # so the tier stays inert rather than aborting boot.
        _point_managed(monkeypatch, _ExplodingPath(tmp_path / "managed.json"))
        assert governance._read_managed_policy() is None

    def test_tier_names_are_distinct_and_break_glass_excludes_the_authorities(self):
        assert len({TIER_MANAGED, TIER_CENTRAL, TIER_ENV, TIER_BUNDLED, TIER_HOME}) == 5
        # An authority cannot grant itself an override, so neither authority tier
        # is nameable in a break_glass block.
        assert TIER_MANAGED not in BREAK_GLASS_TIERS
        assert TIER_CENTRAL not in BREAK_GLASS_TIERS
        assert BREAK_GLASS_TIERS == {TIER_ENV, TIER_BUNDLED, TIER_HOME}

    def test_the_managed_path_is_not_environment_overridable(self, monkeypatch, tmp_path):
        # The absence of an override is the tier's whole trust claim: an env var is
        # per-process and redefinable by whoever launches the process, so an MDM
        # could set one but never pin one. Calls the REAL resolver (captured at
        # import) with plausible override names set, and asserts none of them wins.
        planted = tmp_path / "planted.json"
        for var in (
            "KIROCREW_MANAGED_POLICY",
            "KIROCREW_MANAGED_SECURITY_POLICY",
            "KIROCREW_MANAGED_POLICY_PATH",
        ):
            monkeypatch.setenv(var, str(planted))
        resolved = _REAL_MANAGED_POLICY_PATH()
        assert resolved is None or resolved != planted
        # Only two constants and None. Windows is deliberately None: the obvious
        # implementation there reads %ProgramData% from the environment, which is
        # exactly the per-process override this test exists to forbid, and the
        # ownership check that would catch a redirect cannot run without a uid.
        assert resolved in (
            None,
            governance._MANAGED_POLICY_MACOS,
            governance._MANAGED_POLICY_LINUX,
        )

    def test_windows_has_no_managed_tier_rather_than_an_overridable_one(self, monkeypatch):
        """The tier must not advertise a guarantee it cannot enforce.

        A ``%ProgramData%``-derived path would be resolved from a variable the
        launching user controls, and ``_assert_managed_file_trusted`` returns early on
        non-POSIX with no ownership check -- so a standard user could install their own
        document as the TOP authority. Absent beats falsely authoritative.
        """
        import kiro_crew.platform_compat as pc

        monkeypatch.setattr(pc, "IS_MACOS", False, raising=False)
        monkeypatch.setattr(pc, "IS_LINUX", False, raising=False)
        monkeypatch.setattr(pc, "IS_WINDOWS", True, raising=False)
        monkeypatch.setenv("ProgramData", "C:\\Users\\me\\evil")
        assert _REAL_MANAGED_POLICY_PATH() is None


# ──────────────────────────────────────────────────────────────────────────
# (b) The managed document outranks every local channel
# ──────────────────────────────────────────────────────────────────────────
class TestTheManagedTierOutranksLocalChannels:
    def test_managed_outranks_the_env_tier(self, monkeypatch, tmp_path, trusted_managed):
        _point_managed(monkeypatch, _write_policy(tmp_path / "managed.json", "mdm"))
        monkeypatch.setenv(_POLICY_ENV, str(_write_policy(tmp_path / "env.json", "local-env")))
        ceiling = load_security_policy()
        assert ceiling is not None
        assert ceiling.tier == TIER_MANAGED
        assert ceiling.identity_issuer == "mdm"

    def test_managed_outranks_the_home_tier(self, monkeypatch, tmp_path, trusted_managed):
        _point_managed(monkeypatch, _write_policy(tmp_path / "managed.json", "mdm"))
        _point_home(monkeypatch, _write_policy(tmp_path / "home.json", "operator"))
        ceiling = load_security_policy()
        assert ceiling is not None
        assert ceiling.tier == TIER_MANAGED
        assert ceiling.identity_issuer == "mdm"

    def test_managed_outranks_the_bundled_tier(self, monkeypatch, tmp_path, trusted_managed):
        _point_managed(monkeypatch, _write_policy(tmp_path / "managed.json", "mdm"))
        ceiling = load_security_policy(bundled_loader=lambda: _doc("companion"))
        assert ceiling is not None
        assert ceiling.tier == TIER_MANAGED
        assert ceiling.identity_issuer == "mdm"

    def test_managed_outranks_env_and_home_together(self, monkeypatch, tmp_path, trusted_managed):
        _point_managed(monkeypatch, _write_policy(tmp_path / "managed.json", "mdm"))
        monkeypatch.setenv(_POLICY_ENV, str(_write_policy(tmp_path / "env.json", "local-env")))
        _point_home(monkeypatch, _write_policy(tmp_path / "home.json", "operator"))
        ceiling = load_security_policy()
        assert ceiling is not None
        assert ceiling.tier == TIER_MANAGED
        assert ceiling.identity_issuer == "mdm"

    def test_a_lower_tier_cannot_undo_a_managed_denial(
        self, monkeypatch, tmp_path, trusted_managed
    ):
        # The precedence claim only matters if it survives composition, so the env
        # document below governs the SAME scope and tries to open it up.
        _point_managed(
            monkeypatch,
            _write_policy(
                tmp_path / "managed.json",
                "mdm",
                commands={"mode": "deny", "deny": ["git push*"]},
            ),
        )
        monkeypatch.setenv(
            _POLICY_ENV,
            str(
                _write_policy(
                    tmp_path / "env.json", "local-env", commands={"mode": "deny", "deny": []}
                )
            ),
        )
        ceiling = load_security_policy()
        assert ceiling is not None
        assert not resolve(ceiling, None, "commands", "git push origin main").permitted
        assert ceiling.pinned_command_patterns() == ("git push*",)


# ──────────────────────────────────────────────────────────────────────────
# (c)-(g) Trust checks on a PRESENT managed file -- each one raises
# ──────────────────────────────────────────────────────────────────────────
class TestManagedFileTrustIsChecked:
    @_POSIX_ONLY
    @_NOT_ROOT
    def test_a_non_root_owned_managed_file_is_refused(self, monkeypatch, tmp_path):
        # Deliberately UNPATCHED: a file under tmp_path is owned by whoever runs
        # pytest, so this exercises the real uid comparison against a real fstat
        # rather than a hand-built stat result.
        _point_managed(monkeypatch, _write_policy(tmp_path / "managed.json", "mdm"))
        with pytest.raises(PlatformCompositionError, match="not root"):
            governance._read_managed_policy()

    @_POSIX_ONLY
    @_NOT_ROOT
    def test_a_non_root_managed_file_does_not_fall_through_to_the_home_tier(
        self, monkeypatch, tmp_path
    ):
        # THE security property. Falling through here would restore exactly the
        # override this tier exists to remove, so the load must raise -- not return
        # the permissive local ceiling.
        _point_managed(monkeypatch, _write_policy(tmp_path / "managed.json", "mdm"))
        _point_home(monkeypatch, _write_policy(tmp_path / "home.json", "operator"))
        with pytest.raises(PlatformCompositionError, match="not root"):
            load_security_policy()

    @_POSIX_ONLY
    @_NOT_ROOT
    def test_a_non_root_managed_file_does_not_fall_through_to_the_env_tier(
        self, monkeypatch, tmp_path
    ):
        _point_managed(monkeypatch, _write_policy(tmp_path / "managed.json", "mdm"))
        monkeypatch.setenv(_POLICY_ENV, str(_write_policy(tmp_path / "env.json", "local-env")))
        with pytest.raises(PlatformCompositionError, match="not root"):
            load_security_policy()

    @_POSIX_ONLY
    @_NOT_ROOT
    def test_a_non_root_managed_file_does_not_fall_through_to_ungoverned(
        self, monkeypatch, tmp_path
    ):
        # No lower tier at all: the answer is still a refusal, never the
        # editable-defaults None an ungoverned host gets.
        _point_managed(monkeypatch, _write_policy(tmp_path / "managed.json", "mdm"))
        with pytest.raises(PlatformCompositionError):
            load_security_policy()

    def test_a_group_writable_managed_file_is_refused(self, monkeypatch, tmp_path):
        _point_managed(monkeypatch, _write_policy(tmp_path / "managed.json", "mdm"))
        _fake_managed_stat(monkeypatch, uid=0, extra_mode=stat.S_IWGRP)
        with pytest.raises(PlatformCompositionError, match="group- or world-writable"):
            governance._read_managed_policy()

    def test_a_world_writable_managed_file_is_refused(self, monkeypatch, tmp_path):
        _point_managed(monkeypatch, _write_policy(tmp_path / "managed.json", "mdm"))
        _fake_managed_stat(monkeypatch, uid=0, extra_mode=stat.S_IWOTH)
        with pytest.raises(PlatformCompositionError, match="group- or world-writable"):
            governance._read_managed_policy()

    def test_a_group_writable_managed_file_does_not_fall_through(self, monkeypatch, tmp_path):
        _point_managed(monkeypatch, _write_policy(tmp_path / "managed.json", "mdm"))
        _point_home(monkeypatch, _write_policy(tmp_path / "home.json", "operator"))
        _fake_managed_stat(monkeypatch, uid=0, extra_mode=stat.S_IWGRP)
        with pytest.raises(PlatformCompositionError, match="group- or world-writable"):
            load_security_policy()

    def test_a_root_owned_unwritable_managed_file_is_accepted(self, monkeypatch, tmp_path):
        # Positive control for the two refusals above: with uid 0 and no
        # group/world write bits the same fake PASSES, so those tests are failing
        # on the bits under test and not on the fake itself.
        path = _write_policy(tmp_path / "managed.json", "mdm")
        # 0o644 is pinned in the FAKE, not via chmod: Windows would leave the real
        # mode at 0o666 and this control would fail on the platform, not the guard.
        _point_managed(monkeypatch, path)
        _fake_managed_stat(monkeypatch, uid=0, perm=0o644)
        data = governance._read_managed_policy()
        assert data is not None
        assert data["identity"] == {"issuer": "mdm"}

    def test_the_uid_check_is_skipped_off_posix(self, monkeypatch, tmp_path):
        # Off POSIX there is no uid to compare, so the ownership branch is skipped and
        # only the regular-file test survives. That is exactly why
        # ``_managed_policy_path`` returns None on Windows: a tier whose trust check
        # cannot run must not be reachable, or it would lend its authority to a
        # user-writable file. This test pins the skip itself, which any future
        # non-POSIX platform would also take.
        _point_managed(monkeypatch, _write_policy(tmp_path / "managed.json", "mdm"))
        monkeypatch.setattr(platform_compat, "IS_POSIX", False)
        data = governance._read_managed_policy()
        assert data is not None
        assert data["identity"] == {"issuer": "mdm"}

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="os.mkfifo is POSIX-only")
    def test_a_fifo_at_the_managed_path_is_refused(self, monkeypatch, tmp_path):
        # A FIFO has no bounded size, so it is rejected before a byte is read.
        # O_NONBLOCK in the open is what keeps this from hanging with no writer.
        fifo = tmp_path / "managed.json"
        os.mkfifo(fifo)
        _point_managed(monkeypatch, fifo)
        with pytest.raises(PlatformCompositionError, match="not a regular file"):
            governance._read_managed_policy()

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="os.mkfifo is POSIX-only")
    def test_a_fifo_is_refused_before_the_ownership_check(self, monkeypatch, tmp_path):
        # Ordering matters: the regular-file test must come first, or a FIFO would
        # be reported as an ownership problem and an operator would chown it.
        fifo = tmp_path / "managed.json"
        os.mkfifo(fifo)
        _point_managed(monkeypatch, fifo)
        with pytest.raises(PlatformCompositionError) as excinfo:
            governance._read_managed_policy()
        assert "not root" not in str(excinfo.value)

    @pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="O_NOFOLLOW is POSIX-only")
    def test_a_symlink_at_the_managed_path_is_refused(self, monkeypatch, tmp_path):
        # The open carries O_NOFOLLOW, so a symlink planted at the managed path
        # cannot redirect the ceiling at a file the user does own. The target
        # EXISTS, so the earlier ``path.exists()`` probe does not short-circuit --
        # O_NOFOLLOW is what refuses it.
        target = _write_policy(tmp_path / "attacker.json", "attacker")
        link = tmp_path / "managed.json"
        link.symlink_to(target)
        _point_managed(monkeypatch, link)
        with pytest.raises(PlatformCompositionError, match="could not be opened"):
            governance._read_managed_policy()

    @pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="O_NOFOLLOW is POSIX-only")
    def test_a_symlinked_managed_path_does_not_fall_through(self, monkeypatch, tmp_path):
        target = _write_policy(tmp_path / "attacker.json", "attacker")
        link = tmp_path / "managed.json"
        link.symlink_to(target)
        _point_managed(monkeypatch, link)
        _point_home(monkeypatch, _write_policy(tmp_path / "home.json", "operator"))
        with pytest.raises(PlatformCompositionError):
            load_security_policy()

    def test_an_oversize_managed_file_is_refused(self, monkeypatch, tmp_path, trusted_managed):
        path = tmp_path / "managed.json"
        path.write_bytes(_sized_policy_bytes(governance._MANAGED_POLICY_MAX_BYTES + 1))
        _point_managed(monkeypatch, path)
        with pytest.raises(PlatformCompositionError, match="exceeds"):
            governance._read_managed_policy()

    def test_a_managed_file_at_exactly_the_limit_is_accepted(
        self, monkeypatch, tmp_path, trusted_managed
    ):
        # The bound is a bound, not a blanket refusal, so the boundary is asserted
        # from both sides.
        path = tmp_path / "managed.json"
        path.write_bytes(_sized_policy_bytes(governance._MANAGED_POLICY_MAX_BYTES))
        _point_managed(monkeypatch, path)
        data = governance._read_managed_policy()
        assert data is not None
        assert data["version"] == 1

    def test_an_oversize_managed_file_does_not_fall_through(
        self, monkeypatch, tmp_path, trusted_managed
    ):
        path = tmp_path / "managed.json"
        path.write_bytes(_sized_policy_bytes(governance._MANAGED_POLICY_MAX_BYTES + 1))
        _point_managed(monkeypatch, path)
        _point_home(monkeypatch, _write_policy(tmp_path / "home.json", "operator"))
        with pytest.raises(PlatformCompositionError, match="exceeds"):
            load_security_policy()

    def test_a_managed_file_that_is_not_an_object_is_refused(
        self, monkeypatch, tmp_path, trusted_managed
    ):
        path = tmp_path / "managed.json"
        path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        _point_managed(monkeypatch, path)
        with pytest.raises(PlatformCompositionError, match="not a JSON/plist object"):
            governance._read_managed_policy()

    def test_unparseable_managed_json_is_refused(self, monkeypatch, tmp_path, trusted_managed):
        path = tmp_path / "managed.json"
        path.write_text("{not json", encoding="utf-8")
        _point_managed(monkeypatch, path)
        with pytest.raises(PlatformCompositionError, match="unreadable"):
            governance._read_managed_policy()

    def test_a_structurally_invalid_managed_document_raises_at_its_own_tier(
        self, monkeypatch, tmp_path, trusted_managed
    ):
        # Readable bytes, refused by parse_policy: a fleet that placed a document
        # here meant it to govern, so a bad one aborts rather than degrading.
        path = tmp_path / "managed.json"
        path.write_text(json.dumps({"version": 99, "boot": {}}), encoding="utf-8")
        _point_managed(monkeypatch, path)
        _point_home(monkeypatch, _write_policy(tmp_path / "home.json", "operator"))
        with pytest.raises(PlatformCompositionError, match="version"):
            load_security_policy()


# ──────────────────────────────────────────────────────────────────────────
# (h) Both document encodings
# ──────────────────────────────────────────────────────────────────────────
class TestManagedDocumentEncodings:
    def test_a_macos_plist_managed_document_parses(self, monkeypatch, tmp_path, trusted_managed):
        # macOS publishes a managed configuration profile as a plist in the
        # ``dev.kirocrew`` preference domain, so the reader keys on the suffix.
        _point_managed(monkeypatch, _write_plist_policy(tmp_path / "dev.kirocrew.plist", "mdm"))
        data = governance._read_managed_policy()
        assert data is not None
        assert data["version"] == 1
        assert data["identity"] == {"issuer": "mdm"}

    def test_a_plist_managed_document_governs(self, monkeypatch, tmp_path, trusted_managed):
        _point_managed(
            monkeypatch,
            _write_plist_policy(
                tmp_path / "dev.kirocrew.plist",
                "mdm",
                commands={"mode": "deny", "deny": ["git push*"]},
            ),
        )
        _point_home(monkeypatch, _write_policy(tmp_path / "home.json", "operator"))
        ceiling = load_security_policy()
        assert ceiling is not None
        assert ceiling.tier == TIER_MANAGED
        assert ceiling.identity_issuer == "mdm"
        assert not resolve(ceiling, None, "commands", "git push origin").permitted

    def test_json_bytes_at_a_plist_path_are_refused(self, monkeypatch, tmp_path, trusted_managed):
        # The suffix picks the parser, so a mismatch fails closed instead of being
        # sniffed: a document whose encoding is ambiguous is not trusted.
        path = tmp_path / "dev.kirocrew.plist"
        path.write_text(json.dumps(_doc("mdm")), encoding="utf-8")
        _point_managed(monkeypatch, path)
        with pytest.raises(PlatformCompositionError, match="unreadable"):
            governance._read_managed_policy()

    def test_plist_bytes_at_a_json_path_are_refused(self, monkeypatch, tmp_path, trusted_managed):
        path = tmp_path / "managed.json"
        with path.open("wb") as handle:
            plistlib.dump(_doc("mdm"), handle)
        _point_managed(monkeypatch, path)
        with pytest.raises(PlatformCompositionError, match="unreadable"):
            governance._read_managed_policy()


# ──────────────────────────────────────────────────────────────────────────
# (i) + (j) Tighten-only composition
# ──────────────────────────────────────────────────────────────────────────
class TestASubordinateMayOnlyTighten:
    def _compose(self, monkeypatch, tmp_path, authority: dict, subordinate: dict):
        """Load with a managed *authority* and a home *subordinate*."""
        managed = tmp_path / "managed.json"
        managed.write_text(json.dumps({**_doc("mdm"), **authority}), encoding="utf-8")
        _point_managed(monkeypatch, managed)
        home = tmp_path / "home.json"
        home.write_text(json.dumps({**_doc("operator"), **subordinate}), encoding="utf-8")
        _point_home(monkeypatch, home)
        ceiling = load_security_policy()
        assert ceiling is not None
        return ceiling

    def test_a_subordinate_addition_to_a_governed_scope_is_applied(
        self, monkeypatch, tmp_path, trusted_managed
    ):
        ceiling = self._compose(
            monkeypatch,
            tmp_path,
            authority={"commands": {"mode": "deny", "deny": ["git push*"]}},
            subordinate={"commands": {"mode": "deny", "deny": ["rm -rf*"]}},
        )
        # deny∪ -- both denials bind.
        assert not resolve(ceiling, None, "commands", "git push origin").permitted
        assert not resolve(ceiling, None, "commands", "rm -rf /").permitted
        assert resolve(ceiling, None, "commands", "ls -la").permitted

    def test_a_subordinate_cannot_widen_a_deny_list_by_omission(
        self, monkeypatch, tmp_path, trusted_managed
    ):
        ceiling = self._compose(
            monkeypatch,
            tmp_path,
            authority={"commands": {"mode": "deny", "deny": ["git push*"]}},
            subordinate={"commands": {"mode": "deny", "deny": []}},
        )
        assert not resolve(ceiling, None, "commands", "git push origin").permitted

    def test_a_subordinate_cannot_widen_an_allowlist(self, monkeypatch, tmp_path, trusted_managed):
        ceiling = self._compose(
            monkeypatch,
            tmp_path,
            authority={"tools": {"mode": "allow", "allow": ["read", "grep"]}},
            subordinate={"tools": {"mode": "allow", "allow": ["read", "grep", "execute_bash"]}},
        )
        # allow∩ -- the extra entry the subordinate added does not appear.
        assert resolve(ceiling, None, "tools", "read").permitted
        assert not resolve(ceiling, None, "tools", "execute_bash").permitted

    def test_a_subordinate_allowlist_narrows_when_it_is_smaller(
        self, monkeypatch, tmp_path, trusted_managed
    ):
        ceiling = self._compose(
            monkeypatch,
            tmp_path,
            authority={"tools": {"mode": "allow", "allow": ["read", "grep"]}},
            subordinate={"tools": {"mode": "allow", "allow": ["read"]}},
        )
        assert resolve(ceiling, None, "tools", "read").permitted
        assert not resolve(ceiling, None, "tools", "grep").permitted

    def test_a_subordinate_cannot_flip_a_deny_scope_open_with_allow_mode(
        self, monkeypatch, tmp_path, trusted_managed
    ):
        # Rule 1 makes allow-mode ignore deny entirely WITHIN one ruleset, so a
        # subordinate switching mode is the obvious widening attempt. Composition
        # is an AND of the two rulesets, not a mode handover, so it fails.
        ceiling = self._compose(
            monkeypatch,
            tmp_path,
            authority={"commands": {"mode": "deny", "deny": ["git push*"]}},
            subordinate={"commands": {"mode": "allow", "allow": ["git push*", "ls*"]}},
        )
        assert not resolve(ceiling, None, "commands", "git push origin").permitted

    def test_a_subordinate_cannot_relax_an_ordinal(self, monkeypatch, tmp_path, trusted_managed):
        ceiling = self._compose(
            monkeypatch,
            tmp_path,
            authority={"sandbox": {"min_level": "strict"}},
            subordinate={"sandbox": {"min_level": "off"}},
        )
        control = resolve_ordinal(ceiling, None, "sandbox.min_level")
        assert control is not None
        assert control.value == "strict"

    def test_a_subordinate_may_tighten_an_ordinal(self, monkeypatch, tmp_path, trusted_managed):
        ceiling = self._compose(
            monkeypatch,
            tmp_path,
            authority={"sandbox": {"min_level": "standard"}},
            subordinate={"sandbox": {"min_level": "strict"}},
        )
        control = resolve_ordinal(ceiling, None, "sandbox.min_level")
        assert control is not None
        assert control.value == "strict"

    def test_a_subordinate_cannot_relax_an_approval_ordinal(
        self, monkeypatch, tmp_path, trusted_managed
    ):
        ceiling = self._compose(
            monkeypatch,
            tmp_path,
            authority={"approval_mode": "interactive"},
            subordinate={"approval_mode": "yolo"},
        )
        control = resolve_ordinal(ceiling, None, "approval_mode")
        assert control is not None
        assert control.value == "interactive"

    def test_a_subordinate_cannot_re_enable_a_disabled_capability(
        self, monkeypatch, tmp_path, trusted_managed
    ):
        ceiling = self._compose(
            monkeypatch,
            tmp_path,
            authority={"capabilities": {"script_hooks": {"enabled": False}}},
            subordinate={"capabilities": {"script_hooks": {"enabled": True}}},
        )
        gate = ceiling.get("capabilities.script_hooks")
        assert gate is not None
        assert gate.enabled is False  # type: ignore[attr-defined]

    def test_a_subordinate_may_disable_a_capability_the_authority_enabled(
        self, monkeypatch, tmp_path, trusted_managed
    ):
        ceiling = self._compose(
            monkeypatch,
            tmp_path,
            authority={"capabilities": {"script_hooks": {"enabled": True}}},
            subordinate={"capabilities": {"script_hooks": {"enabled": False}}},
        )
        gate = ceiling.get("capabilities.script_hooks")
        assert gate is not None
        assert gate.enabled is False  # type: ignore[attr-defined]

    def test_a_scope_only_the_subordinate_governs_carries_through(
        self, monkeypatch, tmp_path, trusted_managed
    ):
        # An ungoverned scope is unrestricted, so ADDING governance to it is a
        # tightening, not an escape -- the subordinate's control survives whole.
        ceiling = self._compose(
            monkeypatch,
            tmp_path,
            authority={"commands": {"mode": "deny", "deny": ["git push*"]}},
            subordinate={"tools": {"mode": "allow", "allow": ["read"]}},
        )
        assert resolve(ceiling, None, "tools", "read").permitted
        assert not resolve(ceiling, None, "tools", "execute_bash").permitted
        assert not resolve(ceiling, None, "commands", "git push origin").permitted

    def test_a_scope_only_the_authority_governs_is_not_repealed_by_omission(
        self, monkeypatch, tmp_path, trusted_managed
    ):
        ceiling = self._compose(
            monkeypatch,
            tmp_path,
            authority={"tools": {"mode": "allow", "allow": ["read"]}},
            subordinate={"commands": {"mode": "deny", "deny": ["rm -rf*"]}},
        )
        assert not resolve(ceiling, None, "tools", "execute_bash").permitted

    def test_composition_is_the_same_for_the_env_tier(self, monkeypatch, tmp_path, trusted_managed):
        # The subordinate's identity does not change the algebra: whichever of
        # tiers 3-5 is present composes the same way.
        _point_managed(
            monkeypatch,
            _write_policy(
                tmp_path / "managed.json", "mdm", tools={"mode": "allow", "allow": ["read"]}
            ),
        )
        monkeypatch.setenv(
            _POLICY_ENV,
            str(
                _write_policy(
                    tmp_path / "env.json",
                    "local-env",
                    tools={"mode": "allow", "allow": ["read", "execute_bash"]},
                )
            ),
        )
        ceiling = load_security_policy()
        assert ceiling is not None
        assert resolve(ceiling, None, "tools", "read").permitted
        assert not resolve(ceiling, None, "tools", "execute_bash").permitted


# ──────────────────────────────────────────────────────────────────────────
# (k) Boot flags compose strictest-wins
# ──────────────────────────────────────────────────────────────────────────
class TestBootFlagsComposeStrictestWins:
    def _boot(self, monkeypatch, tmp_path, authority: dict, subordinate: dict):
        managed = tmp_path / "managed.json"
        managed.write_text(
            json.dumps({"version": 1, "boot": authority, "identity": {"issuer": "mdm"}}),
            encoding="utf-8",
        )
        _point_managed(monkeypatch, managed)
        home = tmp_path / "home.json"
        home.write_text(
            json.dumps({"version": 1, "boot": subordinate, "identity": {"issuer": "operator"}}),
            encoding="utf-8",
        )
        _point_home(monkeypatch, home)
        ceiling = load_security_policy()
        assert ceiling is not None
        return ceiling.boot

    def test_a_strict_subordinate_tightens_a_loose_authority(
        self, monkeypatch, tmp_path, trusted_managed
    ):
        boot = self._boot(
            monkeypatch,
            tmp_path,
            authority={"require_sandbox": False, "allow_terminal": True, "fail_closed": False},
            subordinate={"require_sandbox": True, "allow_terminal": False, "fail_closed": True},
        )
        assert boot.require_sandbox is True  # OR
        assert boot.allow_terminal is False  # AND
        assert boot.fail_closed is True  # OR

    def test_a_loose_subordinate_cannot_relax_a_strict_authority(
        self, monkeypatch, tmp_path, trusted_managed
    ):
        boot = self._boot(
            monkeypatch,
            tmp_path,
            authority={"require_sandbox": True, "allow_terminal": False, "fail_closed": True},
            subordinate={"require_sandbox": False, "allow_terminal": True, "fail_closed": False},
        )
        assert boot.require_sandbox is True
        assert boot.allow_terminal is False
        assert boot.fail_closed is True

    def test_allow_terminal_needs_both_tiers_to_agree(self, monkeypatch, tmp_path, trusted_managed):
        boot = self._boot(
            monkeypatch,
            tmp_path,
            authority={"allow_terminal": True},
            subordinate={"allow_terminal": True},
        )
        assert boot.allow_terminal is True


# ──────────────────────────────────────────────────────────────────────────
# (l) Everything outside ``controls`` stays the authority's
# ──────────────────────────────────────────────────────────────────────────
class TestNonControlFieldsStayWithTheAuthority:
    @pytest.fixture
    def composed(self, monkeypatch, tmp_path, trusted_managed):
        _point_managed(
            monkeypatch,
            _write_policy(
                tmp_path / "managed.json",
                "mdm",
                updates={"source": "git@fleet.example:kirocrew.git", "min_version": "9.9.9"},
                break_glass={"tiers": ["bundled"], "expires": _future()},
            ),
        )
        home = tmp_path / "home.json"
        home.write_text(
            json.dumps(
                {
                    "version": 1,
                    "boot": {"fail_closed": True},
                    # A signed-but-unprovable identity, so the composed
                    # signature_state can be told apart from the authority's.
                    "identity": {"issuer": "operator", "signature": "deadbeef"},
                    "updates": {"source": "git@attacker.example:x.git", "min_version": "0.0.1"},
                    "break_glass": {"tiers": ["env"], "expires": _future()},
                }
            ),
            encoding="utf-8",
        )
        _point_home(monkeypatch, home)
        ceiling = load_security_policy()
        assert ceiling is not None
        return ceiling

    def test_identity_stays_the_authoritys(self, composed):
        assert composed.identity_issuer == "mdm"
        assert composed.identity_signature == ""

    def test_signature_state_stays_the_authoritys(self, composed):
        # The managed document carries no signature; the home one carries an
        # unprovable one. A subordinate must not relabel the ceiling's provenance
        # in either direction.
        assert composed.signature_state == SIGNATURE_UNSIGNED
        assert composed.signature_state != SIGNATURE_UNVERIFIED

    def test_update_pins_stay_the_authoritys(self, composed):
        assert composed.updates.source == "git@fleet.example:kirocrew.git"
        assert composed.updates.min_version == "9.9.9"

    def test_distribution_pins_stay_the_authoritys(self, composed):
        # Neither document declares a source (declaring one would make the central
        # tier fetch), so the assertion is that the composed value is the
        # authority's -- a subordinate cannot redirect where the NEXT document
        # comes from.
        assert composed.distribution == governance.PolicyDistribution()

    def test_the_tier_label_stays_the_authoritys(self, composed):
        assert composed.tier == TIER_MANAGED

    def test_break_glass_stays_the_authoritys(self, composed):
        # A subordinate cannot widen its own escape hatch: the home document named
        # "env", and that grant is discarded.
        assert composed.break_glass.tiers == (TIER_BUNDLED,)
        assert not composed.break_glass.grants(TIER_ENV)
        assert composed.break_glass.grants(TIER_BUNDLED)

    def test_a_subordinate_fallback_applies_only_when_the_authority_declares_none(
        self, monkeypatch, tmp_path, trusted_managed
    ):
        # The one documented exception: a fallback only ever narrows what an
        # unusable profile file would have allowed, so the subordinate's is
        # honoured when the authority declared nothing.
        _point_managed(monkeypatch, _write_policy(tmp_path / "managed.json", "mdm"))
        _point_home(
            monkeypatch,
            _write_policy(
                tmp_path / "home.json",
                "operator",
                fallback={"tools": {"mode": "allow", "allow": ["read"]}},
            ),
        )
        ceiling = load_security_policy()
        assert ceiling is not None
        assert ceiling.tier == TIER_MANAGED
        fallback = ceiling.fallback_profile
        assert fallback is not None
        assert resolve(None, fallback, "tools", "read").permitted
        assert not resolve(None, fallback, "tools", "grep").permitted

    def test_the_authoritys_fallback_wins_when_both_declare_one(
        self, monkeypatch, tmp_path, trusted_managed
    ):
        _point_managed(
            monkeypatch,
            _write_policy(
                tmp_path / "managed.json",
                "mdm",
                fallback={"tools": {"mode": "allow", "allow": ["grep"]}},
            ),
        )
        _point_home(
            monkeypatch,
            _write_policy(
                tmp_path / "home.json",
                "operator",
                fallback={"tools": {"mode": "allow", "allow": ["read"]}},
            ),
        )
        ceiling = load_security_policy()
        assert ceiling is not None
        fallback = ceiling.fallback_profile
        assert fallback is not None
        assert resolve(None, fallback, "tools", "grep").permitted
        assert not resolve(None, fallback, "tools", "read").permitted


# ──────────────────────────────────────────────────────────────────────────
# (m) BreakGlass -- the dated, authority-issued override
# ──────────────────────────────────────────────────────────────────────────
class TestBreakGlassParsing:
    def test_an_absent_block_grants_nothing(self):
        assert BreakGlass.from_dict({}) == BreakGlass()
        assert not BreakGlass.from_dict({}).grants(TIER_ENV)

    def test_a_grant_with_no_expires_field_is_inert(self):
        # An undated grant is a permanent hole that outlives its incident, so the
        # fail-closed direction is "grants nothing".
        block = BreakGlass.from_dict({"tiers": ["env"]})
        assert block.tiers == (TIER_ENV,)
        assert not block.grants(TIER_ENV)
        assert "NO EXPIRY (inert)" in block.summary()

    def test_an_unparseable_expires_is_inert(self):
        assert not BreakGlass.from_dict({"tiers": ["env"], "expires": "next tuesday"}).grants(
            TIER_ENV
        )

    def test_an_unexpired_grant_releases_only_the_named_tier(self):
        block = BreakGlass.from_dict({"tiers": ["env"], "expires": _future()})
        assert block.grants(TIER_ENV)
        assert not block.grants(TIER_HOME)
        assert not block.grants(TIER_BUNDLED)

    def test_an_expired_grant_releases_nothing(self):
        assert not BreakGlass.from_dict({"tiers": ["env"], "expires": _past()}).grants(TIER_ENV)

    def test_a_naive_expiry_is_read_as_utc(self):
        # An operator writing a bare date means a date; comparing it against an
        # aware ``now`` would otherwise raise.
        assert BreakGlass.from_dict({"tiers": ["env"], "expires": "2999-01-01"}).grants(TIER_ENV)
        assert not BreakGlass.from_dict({"tiers": ["env"], "expires": "2000-01-01"}).grants(
            TIER_ENV
        )

    def test_a_z_suffixed_expiry_parses(self):
        block = BreakGlass.from_dict({"tiers": ["env"], "expires": "2999-01-01T00:00:00Z"})
        assert block.grants(TIER_ENV)

    def test_grants_accepts_an_explicit_now(self):
        block = BreakGlass.from_dict({"tiers": ["env"], "expires": "2026-01-01T00:00:00Z"})
        assert block.grants(TIER_ENV, now=datetime(2025, 1, 1, tzinfo=timezone.utc))
        assert not block.grants(TIER_ENV, now=datetime(2027, 1, 1, tzinfo=timezone.utc))

    def test_the_summary_names_the_tiers_and_the_window(self):
        block = BreakGlass.from_dict({"tiers": ["home", "env"], "expires": "2999-01-01"})
        assert block.summary() == "break-glass for env,home until 2999-01-01"

    def test_an_unknown_tier_name_raises(self):
        # A misspelling is refused rather than dropped: silently ignoring it leaves
        # an operator believing a recovery channel is open during exactly the
        # incident they need it for.
        with pytest.raises(PlatformCompositionError, match="unknown tier"):
            BreakGlass.from_dict({"tiers": ["enviroment"], "expires": _future()})

    @pytest.mark.parametrize("tier", [TIER_MANAGED, TIER_CENTRAL])
    def test_an_authority_cannot_grant_itself_an_override(self, tier):
        with pytest.raises(PlatformCompositionError, match="unknown tier"):
            BreakGlass.from_dict({"tiers": [tier], "expires": _future()})

    def test_a_string_tiers_value_raises(self):
        with pytest.raises(PlatformCompositionError, match="must be a list"):
            BreakGlass.from_dict({"tiers": "env", "expires": _future()})

    def test_a_non_object_break_glass_block_raises(self):
        with pytest.raises(PlatformCompositionError, match="'break_glass' must be an object"):
            governance.parse_policy(_doc("mdm", break_glass=["env"]))

    def test_an_unknown_tier_in_a_document_raises_at_parse(self):
        with pytest.raises(PlatformCompositionError, match="unknown tier"):
            governance.parse_policy(
                _doc("mdm", break_glass={"tiers": ["nope"], "expires": _future()})
            )


class TestBreakGlassAtLoad:
    def _load(self, monkeypatch, tmp_path, *, block: object):
        _point_managed(
            monkeypatch,
            _write_policy(
                tmp_path / "managed.json",
                "mdm",
                commands={"mode": "deny", "deny": ["git push*"]},
                break_glass=block,
            ),
        )
        monkeypatch.setenv(
            _POLICY_ENV,
            str(
                _write_policy(
                    tmp_path / "env.json", "local-env", commands={"mode": "deny", "deny": []}
                )
            ),
        )
        ceiling = load_security_policy()
        assert ceiling is not None
        return ceiling

    def test_an_unexpired_grant_makes_the_env_document_replace_the_authority(
        self, monkeypatch, tmp_path, trusted_managed
    ):
        ceiling = self._load(monkeypatch, tmp_path, block={"tiers": ["env"], "expires": _future()})
        # REPLACE, not intersect: the released tier is returned whole, which is
        # what makes it a usable recovery lever for a bad authority document.
        assert ceiling.tier == TIER_ENV
        assert ceiling.identity_issuer == "local-env"
        assert resolve(ceiling, None, "commands", "git push origin").permitted

    def test_an_expired_grant_does_not_release_the_env_tier(
        self, monkeypatch, tmp_path, trusted_managed
    ):
        ceiling = self._load(monkeypatch, tmp_path, block={"tiers": ["env"], "expires": _past()})
        assert ceiling.tier == TIER_MANAGED
        assert ceiling.identity_issuer == "mdm"
        assert not resolve(ceiling, None, "commands", "git push origin").permitted

    def test_a_grant_with_no_expires_releases_nothing(self, monkeypatch, tmp_path, trusted_managed):
        ceiling = self._load(monkeypatch, tmp_path, block={"tiers": ["env"]})
        assert ceiling.tier == TIER_MANAGED
        assert not resolve(ceiling, None, "commands", "git push origin").permitted

    def test_a_grant_for_a_different_tier_does_not_release_the_env_tier(
        self, monkeypatch, tmp_path, trusted_managed
    ):
        ceiling = self._load(monkeypatch, tmp_path, block={"tiers": ["home"], "expires": _future()})
        assert ceiling.tier == TIER_MANAGED
        assert not resolve(ceiling, None, "commands", "git push origin").permitted

    def test_a_released_tier_carries_its_own_tier_label(
        self, monkeypatch, tmp_path, trusted_managed
    ):
        # The audit trail has to name the document that actually governed, so the
        # returned ceiling is labelled with the RELEASED tier, not the authority.
        ceiling = self._load(monkeypatch, tmp_path, block={"tiers": ["env"], "expires": _future()})
        assert ceiling.tier == TIER_ENV
        assert ceiling.tier != TIER_MANAGED

    def test_a_break_glass_naming_managed_aborts_the_load(
        self, monkeypatch, tmp_path, trusted_managed
    ):
        # An authority granting itself an override is not a sentence this model can
        # express, so the document is refused rather than partly honoured.
        with pytest.raises(PlatformCompositionError, match="unknown tier"):
            self._load(monkeypatch, tmp_path, block={"tiers": ["managed"], "expires": _future()})

    def test_a_break_glass_naming_central_aborts_the_load(
        self, monkeypatch, tmp_path, trusted_managed
    ):
        with pytest.raises(PlatformCompositionError, match="unknown tier"):
            self._load(monkeypatch, tmp_path, block={"tiers": ["central"], "expires": _future()})

    def test_a_grant_releases_the_home_tier_when_it_is_the_subordinate(
        self, monkeypatch, tmp_path, trusted_managed
    ):
        _point_managed(
            monkeypatch,
            _write_policy(
                tmp_path / "managed.json",
                "mdm",
                commands={"mode": "deny", "deny": ["git push*"]},
                break_glass={"tiers": ["home"], "expires": _future()},
            ),
        )
        _point_home(monkeypatch, _write_policy(tmp_path / "home.json", "operator"))
        ceiling = load_security_policy()
        assert ceiling is not None
        assert ceiling.tier == TIER_HOME
        assert ceiling.identity_issuer == "operator"
        assert resolve(ceiling, None, "commands", "git push origin").permitted

    def test_a_grant_releases_the_bundled_tier_when_it_is_the_subordinate(
        self, monkeypatch, tmp_path, trusted_managed
    ):
        _point_managed(
            monkeypatch,
            _write_policy(
                tmp_path / "managed.json",
                "mdm",
                commands={"mode": "deny", "deny": ["git push*"]},
                break_glass={"tiers": ["bundled"], "expires": _future()},
            ),
        )
        ceiling = load_security_policy(bundled_loader=lambda: _doc("companion"))
        assert ceiling is not None
        assert ceiling.tier == TIER_BUNDLED
        assert ceiling.identity_issuer == "companion"

    def test_a_grant_naming_a_tier_that_is_not_present_changes_nothing(
        self, monkeypatch, tmp_path, trusted_managed
    ):
        # The grant is for the bundled tier, but no bundled document exists, so the
        # env document below is still merely a subordinate.
        ceiling = self._load(
            monkeypatch, tmp_path, block={"tiers": ["bundled"], "expires": _future()}
        )
        assert ceiling.tier == TIER_MANAGED
        assert not resolve(ceiling, None, "commands", "git push origin").permitted
