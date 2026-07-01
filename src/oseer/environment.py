"""Read-only probe of the real terminal environment.

Everything here is strictly read-only and drawn from an allowlist of commands — OSeer
never executes the command it is asked to predict. The snapshot grounds the world model
in *this* machine (e.g. macOS/zsh) rather than a generic Linux box.

Because OSeer sends this context to a hosted model, environment variables are aggressively
sanitized: any var whose name looks secret has its value redacted, and grounding depth is
configurable (``full`` / ``minimal`` / ``none``).
"""

from __future__ import annotations

import os
import re
import subprocess
import time

from .config import Grounding, Settings, get_settings
from .schemas import EnvSnapshot

# Tools an agent commonly reaches for; presence/absence strongly shapes predictions.
CURATED_TOOLS = [
    # shells & core
    "bash", "zsh", "fish", "sh", "sed", "awk", "grep", "find", "tar", "unzip",
    "ssh", "scp", "rsync", "curl", "wget", "make", "tmux", "watch",
    # vcs / hosting
    "git", "gh", "svn",
    # node / js
    "node", "npm", "npx", "pnpm", "yarn", "bun", "deno",
    # python
    "python3", "python", "pip", "pip3", "uv", "poetry", "pipenv", "conda",
    # other languages
    "go", "cargo", "rustc", "gcc", "clang", "java", "mvn", "gradle",
    "ruby", "gem", "bundle", "php", "composer", "dotnet",
    # containers / infra / cloud
    "docker", "podman", "kubectl", "helm", "terraform", "ansible",
    "aws", "gcloud", "az",
    # data / misc
    "psql", "mysql", "sqlite3", "redis-cli", "mongosh", "rg", "jq", "yq", "brew", "apt",
]

PACKAGE_MANAGERS = [
    "brew", "apt", "apt-get", "yum", "dnf", "pacman", "apk",
    "npm", "pnpm", "yarn", "bun", "pip", "pip3", "uv", "poetry",
    "cargo", "go", "gem", "composer",
]

# Env vars worth sending (non-secret, prediction-relevant). Names are matched case-insensitively.
RELEVANT_ENV_VARS = [
    "SHELL", "TERM", "LANG", "PWD", "HOME", "USER", "PATH",
    "NODE_ENV", "VIRTUAL_ENV", "CONDA_DEFAULT_ENV", "PYENV_VERSION",
    "npm_config_registry", "DOCKER_HOST", "KUBECONFIG",
]

# Any env var whose NAME matches this is treated as secret; its value is redacted.
_SECRET_NAME = re.compile(
    r"(TOKEN|KEY|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH|SESSION|COOKIE|PRIVATE|SIGNATURE|CERT)",
    re.IGNORECASE,
)
REDACTED = "***redacted***"

_MAX_LISTING = 60  # cap cwd listing entries sent to the model


def sanitize_env(raw: dict[str, str]) -> dict[str, str]:
    """Redact values of secret-looking vars; pass through the rest unchanged.

    Applied to the *full* environment so secrets are stripped even if a relevant var
    (e.g. an odd ``PATH`` alias) happens to look sensitive.
    """
    clean: dict[str, str] = {}
    for name, value in raw.items():
        clean[name] = REDACTED if _SECRET_NAME.search(name) else value
    return clean


def _run(cmd: list[str], timeout: float = 4.0) -> str:
    """Run a read-only command, returning stripped stdout ('' on any failure)."""
    try:
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _which(tool: str) -> bool:
    from shutil import which

    return which(tool) is not None


class EnvironmentProbe:
    """Collects and caches (with a TTL) an :class:`EnvSnapshot`."""

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._cache: EnvSnapshot | None = None
        self._cached_at: float = 0.0

    def snapshot(self, cwd: str | None = None, refresh: bool = False) -> EnvSnapshot:
        now = time.time()
        fresh = (
            self._cache is not None
            and not refresh
            and (now - self._cached_at) < self._settings.env_ttl
            and (cwd is None or cwd == self._cache.cwd)
        )
        if fresh:
            return self._cache  # type: ignore[return-value]

        snap = self._collect(cwd=cwd, now=now)
        self._cache = snap
        self._cached_at = now
        return snap

    # -- collection -----------------------------------------------------------

    def _collect(self, cwd: str | None, now: float) -> EnvSnapshot:
        grounding = self._settings.grounding
        workdir = cwd or os.getcwd()

        snap = EnvSnapshot(grounding=grounding.value, captured_at=now)
        if grounding is Grounding.none:
            return snap

        # OS / kernel — always cheap and highly relevant.
        uname = os.uname()
        snap.os = uname.sysname
        snap.kernel = uname.release
        if uname.sysname == "Darwin":
            product = _run(["sw_vers", "-productName"])
            version = _run(["sw_vers", "-productVersion"])
            snap.os = product or "macOS"
            snap.os_version = version
        else:
            snap.os_version = _run(["sh", "-c", ". /etc/os-release 2>/dev/null && echo $PRETTY_NAME"])

        snap.shell = os.environ.get("SHELL", "")
        if snap.shell:
            version_lines = _run([snap.shell, "--version"]).splitlines()
            snap.shell_version = version_lines[0] if version_lines else ""

        snap.tools_available = {t: _which(t) for t in CURATED_TOOLS}
        snap.package_managers = [p for p in PACKAGE_MANAGERS if snap.tools_available.get(p, _which(p))]

        if grounding is Grounding.minimal:
            return snap

        # --- full grounding: cwd contents, git status, env vars ---
        snap.cwd = workdir
        snap.cwd_listing = self._listing(workdir)
        snap.git = self._git_summary(workdir)
        snap.env_vars = self._relevant_env()
        return snap

    def _listing(self, workdir: str) -> list[str]:
        try:
            entries = sorted(os.listdir(workdir))
        except OSError:
            return []
        out: list[str] = []
        for name in entries[:_MAX_LISTING]:
            full = os.path.join(workdir, name)
            suffix = "/" if os.path.isdir(full) else ""
            out.append(f"{name}{suffix}")
        if len(entries) > _MAX_LISTING:
            out.append(f"... (+{len(entries) - _MAX_LISTING} more)")
        return out

    def _git_summary(self, workdir: str) -> str | None:
        inside = _run(["git", "-C", workdir, "rev-parse", "--is-inside-work-tree"])
        if inside != "true":
            return None
        branch = _run(["git", "-C", workdir, "branch", "--show-current"])
        status = _run(["git", "-C", workdir, "status", "--porcelain"])
        dirty = len([ln for ln in status.splitlines() if ln.strip()])
        state = "clean" if dirty == 0 else f"{dirty} uncommitted change(s)"
        parts = [f"branch={branch or 'DETACHED'}", state]

        # Remote URL grounds push/pull predictions (avoids hallucinated remotes).
        remote = _run(["git", "-C", workdir, "remote", "get-url", "origin"])
        if remote:
            parts.append(f"origin={remote}")

        # Ahead/behind vs upstream grounds push (non-fast-forward) predictions.
        counts = _run(["git", "-C", workdir, "rev-list", "--left-right", "--count", "@{u}...HEAD"])
        if counts and "\t" in counts:
            behind, ahead = counts.split("\t", 1)
            parts.append(f"ahead={ahead.strip()} behind={behind.strip()}")
        elif not _run(["git", "-C", workdir, "rev-parse", "--abbrev-ref", "@{u}"]):
            parts.append("no-upstream")

        return " ".join(parts)

    def _relevant_env(self) -> dict[str, str]:
        wanted = {n.lower() for n in RELEVANT_ENV_VARS}
        selected = {k: v for k, v in os.environ.items() if k.lower() in wanted}
        return sanitize_env(selected)
