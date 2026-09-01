"""The OS-level sandbox mask covers the crew data home's governance tree.

``security.sensitive_home_dirs()`` is the agent-TOOL gate: it is what
``is_sensitive_path`` refuses for a file_read/file_write tool call. The dir lists in
``sandbox.py`` are a separate, OS-level gate, and a spawned shell command reaches a path
fenced only by the first one. These tests pin the reconciliation between them:

* every crew-home entry on the tool gate has one of three sandbox dispositions,
* the ceilings are exposed READ-ONLY rather than hidden, in every mode,
* the deliberate read-write exceptions are exactly the declared set, in every mode.

The third is the one worth failing loudly: an entry that quietly moves from "masked" to
"exception" is a ceiling the agent can rewrite again.
"""

from __future__ import annotations

import json
import os
import re
import sys

import pytest

from kiro_crew import sandbox, security

_POSIX_ONLY = pytest.mark.skipif(sys.platform == "win32", reason="POSIX launcher only")

_MODES = ("standard", "cc", "strict")
_CREW_PREFIXES = (".kiro/crew", ".kirocrew")


def _home() -> str:
    return os.path.expanduser("~")


def _launcher_sets(mode: str) -> tuple[set[str], set[str], set[str]]:
    """``(hidden_dirs, readonly, hidden_files)`` as the generated launcher declares them."""
    script = sandbox._build_launcher_script(mode)

    def _grab(name: str) -> set[str]:
        match = re.search(rf"{name} = (\[.*?\])\n", script, re.S)
        assert match, f"{name} missing from the launcher"
        return set(json.loads(match.group(1)))

    return _grab("SENSITIVE_DIRS"), _grab("READONLY_DIRS"), _grab("SENSITIVE_FILES")


def _crew_sensitive_paths() -> list[str]:
    """Absolute crew-data-home paths the tool gate declares sensitive."""
    home = _home()
    return [
        os.path.join(home, rel)
        for rel in security.sensitive_home_dirs()
        if rel.startswith(_CREW_PREFIXES)
    ]


def _expected_exceptions() -> set[str]:
    home = _home()
    return {
        os.path.join(home, f"{prefix}/{leaf}")
        for prefix in _CREW_PREFIXES
        for leaf in sandbox._CREW_SANDBOX_VISIBLE_LEAVES
    }


class TestKeystonesAreSealedInEveryMode:
    """The ceiling files the issue named, on every backend the launcher feeds."""

    #: Named individually rather than looped from the module tuple: the point is that
    #: THESE paths are covered, so a test derived from the same tuple the production
    #: code reads would pass just as happily after someone emptied it.
    KEYSTONES = (
        "security_policy.json",
        "admission_policy.json",
        "app_admission.json",
        "profiles",
        "denied_commands.json",
        "computer_use.json",
        "oauth_endpoints.json",
        "aws_service_consent.json",
    )

    @_POSIX_ONLY
    @pytest.mark.parametrize("mode", _MODES)
    @pytest.mark.parametrize("prefix", _CREW_PREFIXES)
    @pytest.mark.parametrize("leaf", KEYSTONES)
    def test_linux_seals_the_ceiling_read_only(self, mode: str, prefix: str, leaf: str) -> None:
        hidden, readonly, _files = _launcher_sets(mode)
        target = os.path.join(_home(), prefix, leaf)

        assert target in readonly, f"{leaf} is writable through the {mode} sandbox"
        # Hiding a ceiling inverts its effect: an absent policy file resolves to the
        # permissive standalone default, and a script cron's ``boot_platform()`` runs
        # inside this namespace.
        assert target not in hidden, f"{leaf} must stay READABLE, not be masked"

    @pytest.mark.parametrize("mode", _MODES)
    @pytest.mark.parametrize("prefix", _CREW_PREFIXES)
    @pytest.mark.parametrize("leaf", KEYSTONES)
    def test_macos_denies_writes_to_the_ceiling(self, mode: str, prefix: str, leaf: str) -> None:
        profile = sandbox._build_seatbelt_profile(mode)
        target = os.path.join(_home(), prefix, leaf)

        assert f'(deny file-write* (literal "{target}"))' in profile
        assert f'(deny file-write* (subpath "{target}"))' in profile
        # A hardlink at a non-denied path would otherwise reach the same inode.
        assert f'(deny file-link (subpath "{target}"))' in profile
        assert f'(deny file-read* (subpath "{target}"))' not in profile

    @_POSIX_ONLY
    @pytest.mark.parametrize("mode", _MODES)
    def test_the_seal_survives_a_file_shaped_ceiling(self, mode: str) -> None:
        """The read-only loop must not guard on ``isdir``.

        ``security_policy.json`` is a plain file. An ``isdir`` guard skips it silently —
        no error, and the ceiling stays writable.
        """
        script = sandbox._build_launcher_script(mode)
        loop = script.split("for d in READONLY_DIRS:", 1)[1].split("\n\n", 1)[0]

        assert "os.path.exists(target)" in loop
        assert "os.path.isdir(target)" not in loop
        assert "_MS_REMOUNT | _MS_BIND | _MS_RDONLY" in loop


class TestSecretsAreMaskedInEveryMode:
    """Crew-home leaves with no in-sandbox reader are bind-masked, not merely sealed."""

    MASKED = (
        "token_signing.key",
        "refresh_chains.json",
        "kas",
        "mcp-apps",
        "ledger",
        "backup",
        "browser-cookies.txt",
        "playwright-storage-state.json",
        "playwright-extension-token",
        "ops_mission_control_secrets.json",
        "whatsapp",
        "apps/aws-control/data",
        "workspace/md-notebook/pat",
    )

    @_POSIX_ONLY
    @pytest.mark.parametrize("mode", _MODES)
    @pytest.mark.parametrize("prefix", _CREW_PREFIXES)
    @pytest.mark.parametrize("leaf", MASKED)
    def test_linux_masks_it(self, mode: str, prefix: str, leaf: str) -> None:
        hidden, _readonly, files = _launcher_sets(mode)
        target = os.path.join(_home(), prefix, leaf)

        # Both lists, because the child classifies by kind: a file entry reaching only
        # the directory loop is skipped by its ``isdir`` guard and stays readable.
        assert target in hidden, f"{leaf} readable through the {mode} sandbox"
        assert target in files, f"{leaf} would be skipped if it is a file"

    @pytest.mark.parametrize("mode", _MODES)
    @pytest.mark.parametrize("prefix", _CREW_PREFIXES)
    @pytest.mark.parametrize("leaf", MASKED)
    def test_macos_denies_reads(self, mode: str, prefix: str, leaf: str) -> None:
        profile = sandbox._build_seatbelt_profile(mode)
        target = os.path.join(_home(), prefix, leaf)

        assert f'(deny file-read* (subpath "{target}"))' in profile
        assert f'(deny file-link (subpath "{target}"))' in profile


class TestTheReconciliationIsComplete:
    """No crew-home entry on the tool gate is left with no sandbox disposition."""

    @_POSIX_ONLY
    @pytest.mark.parametrize("mode", _MODES)
    def test_every_crew_sensitive_path_is_masked_sealed_or_a_declared_exception(
        self, mode: str
    ) -> None:
        hidden, readonly, _files = _launcher_sets(mode)
        exceptions = _expected_exceptions()

        unaccounted = [
            path
            for path in _crew_sensitive_paths()
            if path not in hidden and path not in readonly and path not in exceptions
        ]
        assert not unaccounted, (
            "crew-home paths the tool gate fences but the OS sandbox does not, and that "
            f"are not declared exceptions either: {unaccounted}"
        )

    @_POSIX_ONLY
    @pytest.mark.parametrize("mode", _MODES)
    def test_the_read_write_exceptions_are_exactly_the_declared_set(self, mode: str) -> None:
        """A path drifting into the exception set is a ceiling the agent can rewrite.

        ``run`` is expected on the read-only list rather than fully read-write: it holds
        the launcher itself, so it must stay readable, and the voice-runtime rules
        already seal it. Every other exception is genuinely unrestricted.
        """
        hidden, readonly, _files = _launcher_sets(mode)
        unrestricted = {
            path
            for path in _crew_sensitive_paths()
            if path not in hidden and path not in readonly
        }
        declared = _expected_exceptions()

        assert unrestricted <= declared, (
            f"undeclared read-write crew paths in {mode}: {sorted(unrestricted - declared)}"
        )
        for path in declared - unrestricted:
            assert path in readonly, (
                f"{path} is declared an exception but is neither read-write nor sealed"
            )

    def test_the_exception_set_names_only_paths_the_tool_gate_fences(self) -> None:
        """An exception for a path nothing fences is dead weight that reads as a hole."""
        home = _home()
        fenced = {os.path.join(home, rel) for rel in security.sensitive_home_dirs()}

        for path in _expected_exceptions():
            assert path in fenced, f"{path} is exempted from a gate that never covered it"

    def test_the_three_dispositions_do_not_overlap(self) -> None:
        hidden = set(sandbox._CREW_HIDDEN_LEAVES)
        readonly = set(sandbox._CREW_READONLY_LEAVES)
        visible = set(sandbox._CREW_SANDBOX_VISIBLE_LEAVES)

        assert not hidden & readonly
        assert not hidden & visible
        assert not readonly & visible

    @pytest.mark.parametrize("mode", _MODES)
    def test_every_mode_carries_the_derived_set(self, mode: str) -> None:
        """The governance tree is masked at every level, the way ``.vault`` already is.

        A per-mode carve-out here would mean ``standard`` — the default — leaves the
        ceiling exposed, which is the configuration almost every install runs.
        """
        listing = {
            "standard": sandbox._STANDARD_DIRS,
            "cc": sandbox._CC_DIRS,
            "strict": sandbox._STRICT_DIRS,
        }[mode]

        for entry in sandbox._CREW_HIDDEN_DIRS:
            assert entry in listing, f"{entry} missing from the {mode} dir list"


class TestThirdPartyCredentialsKeepTheirExistingTiering:
    """The reconciliation must not quietly re-tier the non-crew credential entries."""

    def test_standard_still_exposes_the_developer_workflow_dirs(self) -> None:
        """``standard`` deliberately leaves ``.aws``/``.kube`` readable.

        Masking them here would break an ordinary build running in the default mode; the
        crew governance tree is masked at every level precisely because nothing
        legitimate reads it.
        """
        assert ".aws" not in sandbox._STANDARD_DIRS
        assert ".kube" not in sandbox._STANDARD_DIRS
        assert ".aws" in sandbox._CC_DIRS
        assert ".aws" in sandbox._STRICT_DIRS

    def test_the_sso_cookie_store_is_masked_at_the_credential_tiers(self) -> None:
        """``.midway`` is a live bearer credential, the same class as ``.aws``."""
        assert ".midway" in sandbox._STRICT_DIRS
        assert ".midway" in sandbox._CC_DIRS
        assert ".midway" not in sandbox._STANDARD_DIRS

    def test_the_agent_runtime_auth_stores_stay_visible(self) -> None:
        """kiro-cli / amazon-q identity stores are fenced at the tool gate only.

        The agent runtime is itself spawned inside this sandbox and resolves its own
        access token from that store, so masking it would break the agent's model auth
        rather than protect anything. ``security.py`` states this explicitly.
        """
        for listing in (sandbox._STRICT_DIRS, sandbox._CC_DIRS, sandbox._STANDARD_DIRS):
            assert ".local/share/kiro-cli" not in listing
            assert ".local/share/amazon-q" not in listing
