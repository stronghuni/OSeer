"""Offline tests for the environment probe and secret sanitization."""

from __future__ import annotations

import os

from oseer.config import Grounding, Settings
from oseer.environment import REDACTED, EnvironmentProbe, sanitize_env


def test_sanitize_redacts_secret_named_vars():
    raw = {
        "PATH": "/usr/bin:/bin",
        "AWS_SECRET_ACCESS_KEY": "AKIAsupersecret",
        "GITHUB_TOKEN": "ghp_xxx",
        "MY_PASSWORD": "hunter2",
        "DB_CREDENTIAL": "conn-string",
        "NODE_ENV": "production",
        "SESSION_COOKIE": "abc",
    }
    clean = sanitize_env(raw)

    # Non-secret vars pass through untouched.
    assert clean["PATH"] == "/usr/bin:/bin"
    assert clean["NODE_ENV"] == "production"

    # Anything that looks secret is redacted by NAME, regardless of value.
    for secret in ["AWS_SECRET_ACCESS_KEY", "GITHUB_TOKEN", "MY_PASSWORD",
                   "DB_CREDENTIAL", "SESSION_COOKIE"]:
        assert clean[secret] == REDACTED, secret

    # No secret values leak through.
    joined = " ".join(clean.values())
    for leaked in ["AKIAsupersecret", "ghp_xxx", "hunter2", "conn-string"]:
        assert leaked not in joined


def test_grounding_none_sends_nothing():
    probe = EnvironmentProbe(Settings(grounding=Grounding.none))
    snap = probe.snapshot()
    assert snap.grounding == "none"
    assert snap.os == ""
    assert snap.cwd == ""
    assert snap.env_vars == {}
    assert snap.tools_available == {}


def test_grounding_minimal_has_os_and_tools_but_no_cwd_or_env():
    probe = EnvironmentProbe(Settings(grounding=Grounding.minimal))
    snap = probe.snapshot()
    assert snap.os != ""                 # OS detected
    assert snap.tools_available          # tool map populated
    assert snap.cwd == ""                # but no cwd contents
    assert snap.cwd_listing == []
    assert snap.env_vars == {}           # and no env vars


def test_grounding_full_captures_cwd_and_git(tmp_path, monkeypatch):
    (tmp_path / "hello.txt").write_text("hi")
    (tmp_path / "subdir").mkdir()
    monkeypatch.chdir(tmp_path)

    probe = EnvironmentProbe(Settings(grounding=Grounding.full))
    snap = probe.snapshot()

    assert snap.cwd == str(tmp_path)
    assert "hello.txt" in snap.cwd_listing
    assert "subdir/" in snap.cwd_listing  # directories get a trailing slash
    # env vars are present and never contain a real secret value
    assert REDACTED not in "".join(k for k in snap.env_vars)  # names aren't redacted


def test_full_env_vars_are_sanitized(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    # RELEVANT_ENV_VARS is an allowlist, so inject a name that is both relevant AND secret-looking.
    monkeypatch.setattr("oseer.environment.RELEVANT_ENV_VARS",
                        ["PATH", "OPENAI_API_KEY"])
    probe = EnvironmentProbe(Settings(grounding=Grounding.full))
    snap = probe.snapshot()
    assert snap.env_vars.get("OPENAI_API_KEY") == REDACTED
    assert "sk-should-not-leak" not in " ".join(snap.env_vars.values())


def test_snapshot_is_cached_within_ttl():
    probe = EnvironmentProbe(Settings(grounding=Grounding.minimal, env_ttl=1000))
    first = probe.snapshot()
    second = probe.snapshot()
    assert first is second  # same object returned from cache
    third = probe.snapshot(refresh=True)
    assert third is not first
