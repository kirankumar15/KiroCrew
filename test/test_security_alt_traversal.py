"""Tests for the alternate-traversal pass in ``security.py``.

``find`` is not the only program that factors a fenced path into a root plus a
name and hands the result to a reader. These tests pin the shapes ``fd``,
``grep -r``, ``rg`` and ``du`` can spell, the legitimate forms of each that must
keep working, and the residuals that are deliberately left open.
"""

from __future__ import annotations

import os

import pytest

from kiro_crew import security
from kiro_crew.security import is_sensitive_bash_command

#: A crew-home directory that HOLDS fenced leaves (``.env``,
#: ``token_signing.key``) without being fenced itself -- the root every blocked
#: case below traverses.
CREW = "~/.kiro/crew"

#: The legacy data-home prefix, fenced by the same leaf list.
CREW_LEGACY = "~/.kirocrew"


def _denied(command: str) -> bool:
    return is_sensitive_bash_command(command) is not None


# ── fd: a positional regex, a root, and find's -exec under another name ──


@pytest.mark.parametrize(
    "command",
    [
        # The issue's headline shapes.
        f"fd '^\\.env$' {CREW} -x cat",
        f"fd -e key . {CREW} -X cat",
        # The Debian/Ubuntu binary name for the same tool.
        f"fdfind '^\\.env$' {CREW} -x cat",
        # Long spellings of both exec forms.
        f"fd . {CREW} --exec cat",
        f"fd . {CREW} --exec-batch cat",
        # The reader does not have to be `cat`.
        f"fd . {CREW} -x base64",
        f"fd . {CREW} -x head -c 100",
        # A path-qualified program word.
        f"/usr/bin/fd . {CREW} -x cat",
        # A quoted root, which shlex unwraps.
        f"fd . '{CREW}' -x cat",
        # $HOME instead of a tilde.
        "fd . $HOME/.kiro/crew -x cat",
        # The legacy data-home carries the same leaves.
        f"fd . {CREW_LEGACY} -x cat",
        # The root arrives through a flag rather than as a positional.
        f"fd --search-path {CREW} '^\\.env$' -x cat",
        f"fd --search-path={CREW} nothing --exec-batch cat",
        # No exec flag, but the name list is piped into a reader.
        f"fd . {CREW} | xargs cat",
        f"fd . {CREW} | xargs -0 head -c 100",
        f"fd . {CREW} | parallel cat",
        # xargs flags that take a VALUE put it where the payload would sit, so
        # the payload cannot be found by stopping at the first non-flag token.
        f"fd . {CREW} | xargs -n 1 cat",
        f"fd . {CREW} | xargs -P 4 cat",
        f"fd . {CREW} | xargs -I {{}} cat {{}}",
        # An `env` wrapper hides the program word from a naive first-token read.
        f"env fd . {CREW} -x cat",
        f"env FOO=1 fd . {CREW} -x cat",
        f"env grep -r secret {CREW}",
        # The workspace root holds the Notes vault's PAT.
        f"grep -r secret {CREW}/workspace",
    ],
)
def test_fd_traversal_into_crew_home_is_denied(command: str) -> None:
    assert _denied(command), command


@pytest.mark.parametrize(
    "command",
    [
        # A listing is not a read: names are not the secret.
        f"fd . {CREW}",
        f"fd '^\\.env$' {CREW}",
        # `cat` on the PIPE prints the name list on stdin; it does not open the
        # files those names point to, so it is not a sink.
        f"fd . {CREW} | cat",
        f"fd . {CREW} | wc -l",
        # An ordinary project tree holds no fenced leaf.
        "fd '^main.py$' ./src -x cat",
        "fd -e py . src",
        "fd -e ts . website/src -x npx prettier --check",
        "fd . ~/Documents",
        "fd . ~/projects/app -x cat",
        # A crew subdirectory that holds no fenced leaf stays readable.
        f"fd . {CREW}/workspace/memory -x cat",
    ],
)
def test_fd_without_delivery_or_fence_is_allowed(command: str) -> None:
    assert not _denied(command), command


# ── grep -r / rg: the reader IS the traversal, so there is no sink to find ──


@pytest.mark.parametrize(
    "command",
    [
        f"grep -r secret {CREW}",
        f"grep -R secret {CREW}",
        # Clustered short flags -- the spelling a person actually types.
        f"grep -rn secret {CREW}",
        f"grep -rl secret {CREW}",
        f"grep -irn secret {CREW}",
        # Long spellings.
        f"grep --recursive secret {CREW}",
        f"grep --dereference-recursive secret {CREW}",
        # grep's aliases, including the one that is recursive with no flag.
        f"egrep -r secret {CREW}",
        f"fgrep -r secret {CREW}",
        f"rgrep secret {CREW}",
        # ripgrep recurses with no flag at all.
        f"rg secret {CREW}",
        # `-l` still opens every file to decide whether to print its name.
        f"rg -l secret {CREW}",
        f"rg --files-with-matches secret {CREW}",
        # `--files` is a pure lister, so it needs a sink -- and here it has one.
        f"rg --files {CREW} | xargs cat",
        f"rg --files {CREW} | parallel cat",
        # Reached through a sequencer rather than as the whole line.
        f"true && grep -r secret {CREW}",
        f"cd /tmp; grep -r secret {CREW}",
        f"grep -r secret {CREW_LEGACY}",
    ],
)
def test_recursive_read_rooted_above_fence_is_denied(command: str) -> None:
    assert _denied(command), command


@pytest.mark.parametrize(
    "command",
    [
        # Not recursive: a single named file is the normalizer pass's business,
        # and this one is not fenced.
        "grep -n secret ~/projects/notes.txt",
        "grep secret ./src/main.py",
        # Recursive, but rooted where no fenced leaf lives.
        "grep -r TODO ./src",
        "grep -r secret ~/projects/app",
        "grep -r secret /tmp/scratch",
        f"grep -r TODO {CREW}/workspace/memory",
        "rg secret ./website/src",
        # A pure lister with no sink discloses nothing.
        f"rg --files {CREW}",
        f"rg --files {CREW} | wc -l",
        "rg --files ./src | xargs cat",
        # The words appear as data, not as a program.
        "echo grep -r",
        f"echo 'grep -r secret {CREW}'",
    ],
)
def test_non_recursive_or_unfenced_reads_are_allowed(command: str) -> None:
    assert not _denied(command), command


# ── du: a size lister used as a path producer ──


@pytest.mark.parametrize(
    "command",
    [
        f"du -a {CREW} | xargs cat",
        # An intervening stage does not hide the sink.
        f"du -a {CREW} | awk '{{print $2}}' | xargs cat",
    ],
)
def test_size_lister_with_reader_sink_is_denied(command: str) -> None:
    assert _denied(command), command


@pytest.mark.parametrize(
    "command",
    [
        f"du -a {CREW}",
        f"du -sh {CREW}",
        "du -a ~/projects | xargs cat",
    ],
)
def test_size_lister_without_sink_or_fence_is_allowed(command: str) -> None:
    assert not _denied(command), command


# ── Residuals: named here so a later change cannot quietly assume coverage ──


@pytest.mark.parametrize(
    "command",
    [
        # `locate` has NO root operand -- the database supplies the path -- so the
        # root-containment clause every rule above rests on has nothing to test.
        # Recognising only the leaf names the fence DECLARES would still miss
        # `id_rsa` (`.ssh` is fenced as a whole directory, so no leaf name is
        # declared for it) while reading as covered, so it is left open and named
        # rather than half-closed.
        "locate id_rsa | xargs cat",
        "plocate id_rsa | xargs cat",
        # A name list delivered through a command substitution rather than xargs.
        f"cat $(fd '^\\.env$' {CREW})",
    ],
)
def test_documented_residuals_are_not_yet_covered(command: str) -> None:
    """Pins the residuals the module's block comment names.

    A failure here is GOOD news -- it means a later change closed the shape. Move
    the case up to the denied set and delete it from the block comment's residual
    list; do not relax the pass to keep this test passing.
    """
    assert not _denied(command), command


# ── Unit-level behaviour of the pass's own helpers ──


def test_reader_sink_requires_the_payload_to_be_a_reader() -> None:
    stages = security._alt_pipeline_stages("fd . x | xargs cat")
    assert security._alt_has_reader_sink(stages)
    stages = security._alt_pipeline_stages("fd . x | xargs rm")
    assert not security._alt_has_reader_sink(stages)


def test_reader_sink_survives_an_xargs_flag_that_takes_a_value() -> None:
    """`-n 1` puts its value where the payload would sit."""
    stages = security._alt_pipeline_stages("fd . x | xargs -n 1 cat")
    assert security._alt_has_reader_sink(stages)


def test_stage_head_skips_assignments_and_an_env_wrapper() -> None:
    program, operands = security._alt_stage_head(["env", "FOO=1", "fd", ".", "/tmp"])
    assert program == "fd"
    assert operands == [".", "/tmp"]
    program, operands = security._alt_stage_head(["FOO=1", "grep", "-r", "x"])
    assert program == "grep"
    assert operands == ["-r", "x"]


def test_bare_pipe_to_a_reader_is_not_a_sink() -> None:
    """`| cat` prints the NAME list, it does not open the named files."""
    stages = security._alt_pipeline_stages("fd . x | cat")
    assert not security._alt_has_reader_sink(stages)


def test_pipeline_split_respects_quoting() -> None:
    stages = security._alt_pipeline_stages("grep -r 'a|b' ./src")
    assert len(stages) == 1
    assert security._alt_stage_head(stages[0]) == ("grep", ["-r", "a|b", "./src"])


def test_grep_recursion_is_read_off_clustered_short_flags() -> None:
    assert security._grep_is_recursive(["-rn", "secret", "."])
    assert security._grep_is_recursive(["--recursive", "secret", "."])
    assert not security._grep_is_recursive(["-n", "secret", "."])
    # Everything after `--` is an operand, not a flag.
    assert not security._grep_is_recursive(["--", "-r", "."])


def test_root_check_accepts_a_flag_value_as_a_candidate_root() -> None:
    """A root can arrive as a flag's value, so every operand is tested.

    Tracking which flags take a value would need a table per tool, and each
    omission from such a table would be a MISS.
    """
    home = os.path.expanduser("~")
    assert security._alt_root_reaching_fence([f"--search-path={home}/.kiro/crew"])
    assert security._alt_root_reaching_fence([f"{home}/.kiro/crew"])
    assert not security._alt_root_reaching_fence(["--max-depth", "2", "secret"])


def test_pass_is_reachable_from_the_public_gate() -> None:
    """The pass must be wired into ``is_sensitive_bash_command``, not just defined."""
    reason = is_sensitive_bash_command(f"grep -r secret {CREW}")
    assert reason is not None
    assert "recursive traversal" in reason
