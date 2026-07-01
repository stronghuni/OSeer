"""Offline tests for prompt grounding helpers."""

from __future__ import annotations

from oseer.config import Settings
from oseer.prompts import terminal
from oseer.prompts.terminal import _program_note, _referenced_paths, build_messages
from oseer.schemas import EnvSnapshot


def _env(**kw) -> EnvSnapshot:
    base = dict(os="macOS", shell="/bin/zsh",
                tools_available={"brew": True, "apt-get": False, "git": True})
    base.update(kw)
    return EnvSnapshot(**base)


def test_program_note_present_tool():
    note = _program_note("brew install jq", _env())
    assert note is not None and "IS installed" in note and "brew" in note


def test_program_note_absent_tool():
    note = _program_note("apt-get install curl", _env())
    assert note is not None and "NOT installed" in note and "127" in note


def test_program_note_skips_sudo_and_env_prefix():
    note = _program_note("sudo apt-get update", _env())
    assert note is not None and "apt-get" in note  # judged apt-get, not sudo


def test_program_note_path_invocation_returns_none():
    assert _program_note("./local-bin/tool run", _env()) is None


def test_program_note_unknown_tool_returns_none():
    assert _program_note("frobnicate --now", _env()) is None


def test_referenced_paths_skips_synthetic_cwd():
    # A cwd that isn't a real local dir → cannot verify → None (no fabricated existence).
    assert _referenced_paths("rm -rf /nope/build", "/this/does/not/exist") is None


def test_referenced_paths_reports_real_existence(tmp_path):
    (tmp_path / "keep.txt").write_text("x")
    (tmp_path / "adir").mkdir()
    # Paths are signalled by a slash / dot / ~ ; bare words are deliberately not treated
    # as paths (so e.g. `git status` doesn't report 'status' as a missing file).
    out = _referenced_paths("rm keep.txt adir/ gone.txt", str(tmp_path))
    assert "keep.txt (exists: file)" in out
    assert "adir/ (exists: directory)" in out
    assert "gone.txt (does NOT exist)" in out


def test_referenced_paths_ignores_bare_subcommands(tmp_path):
    # A bare word that is not a path must not be reported (avoids false 'does NOT exist').
    assert _referenced_paths("git status", str(tmp_path)) is None


def test_build_messages_includes_grounding_and_contract():
    msgs = build_messages("apt-get install curl", _env(), Settings())
    system = msgs[0]["content"]
    assert "NOT installed" in system                      # program note
    assert '"safe", "caution", "destructive"' in system   # risk must be a string
    assert msgs[1]["content"].startswith("Action: execute_bash")


def test_env_fingerprint_changes_with_tools():
    a = terminal.env_fingerprint(_env(tools_available={"git": True}))
    b = terminal.env_fingerprint(_env(tools_available={"git": False}))
    assert a != b
