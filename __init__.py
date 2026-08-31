"""hermes-classifier-mode — Claude Code `auto` permission mode for Hermes Agent.

A `pre_tool_call` plugin that routes shell commands through a local LLM
classifier (Ollama by default) before they execute, mirroring Claude Code's
`--permission-mode auto`: autonomy with a second model judging each action,
instead of "ask me every time" or "skip all checks".

Verdicts:
  allow   -> command proceeds (None returned to the hook dispatcher)
  block   -> command vetoed; the model sees the classifier's reason
  approve -> escalated to the existing human approval gate
             ([o]nce/[s]ession/[a]lways/[d]eny) — fail-closed

Fast paths so the common case never pays model latency:
  1. Static instant-allow: single read-only commands (git status, ls, cat, ...)
  2. Static instant-block: high-precision catastrophic patterns
     (fork bombs, curl|sh, disk wipes) — works even with Ollama down
  3. Local classifier for everything else
  4. Classifier unreachable/unparseable -> escalate to human (never fail-open)

Configuration lives in config.yaml under `classifier_mode:` (settings, never
secrets). Defaults work with zero config when Ollama runs on :11434.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (config.yaml: classifier_mode section)
# ---------------------------------------------------------------------------

_DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "ollama_url": "http://localhost:11434",
    "model": "qwen3:30b",
    "timeout_s": 20,
    # How long Ollama keeps the model warm between verdicts (seconds).
    "keep_alive_s": 600,
    # Deny rule that overrides the classifier: force human approval.
    # Entries are regexes matched against the command string.
    "force_approve_patterns": [],
    # Allow rule that overrides the classifier: skip classification entirely.
    "force_allow_patterns": [],
}


def _load_config() -> Dict[str, Any]:
    cfg = dict(_DEFAULTS)
    try:
        from hermes_cli.config import load_config

        raw = load_config() or {}
        section = raw.get("classifier_mode")
        if isinstance(section, dict):
            cfg.update({k: v for k, v in section.items() if v is not None})
    except Exception as e:  # config unavailable -> defaults
        logger.debug("classifier-mode: config load failed, using defaults: %s", e)
    return cfg


# ---------------------------------------------------------------------------
# Fast path 1: static read-only allow
# ---------------------------------------------------------------------------

# Single-command, pipe/redirection-free read-only invocations. Matched as the
# WHOLE command (after stripping env-var prefixes) against `^<word>( .*)?$`.
_READ_ONLY = {
    # vcs
    "git status", "git log", "git show", "git diff", "git branch",
    "git remote", "git blame", "git describe", "git rev-parse",
    "git ls-files", "git ls-remote", "git config --get",
    # inspection
    "ls", "cat", "head", "tail", "wc", "file", "stat", "du", "df",
    "which", "whereis", "whoami", "hostname", "date", "uname", "env",
    "printenv", "pwd", "tree", "find", "locate", "mdfind",
    # dev tools (read invocations)
    "cargo check", "cargo metadata", "cargo tree", "cargo doc --no-deps",
    "python --version", "python3 --version", "node --version",
    "npm ls", "npm view", "rustc --version", "go version", "uv --version",
    "ollama list", "hermes status", "hermes doctor",
}

_READ_ONLY_RE = re.compile(
    r"^(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)*("  # optional FOO=bar env prefixes
    + "|".join(re.escape(c) for c in sorted(_READ_ONLY, key=len, reverse=True))
    + r")(?=\s|$)"
)


def _static_allow(command: str) -> bool:
    """True for unambiguous single read-only commands (no pipes/redirects/
    substitution/other shell operators)."""
    if any(ch in command for ch in "|;&`$><\n"):
        return False
    return bool(_READ_ONLY_RE.match(command.strip()))


# ---------------------------------------------------------------------------
# Fast path 2: static catastrophic block (high precision, defense-in-depth)
# ---------------------------------------------------------------------------

_STATIC_BLOCK_RES = [
    # remote code execution
    re.compile(r"curl[^|]*\|\s*(ba)?sh\b"),
    re.compile(r"wget[^|]*\|\s*(ba)?sh\b"),
    re.compile(r"curl[^|]*\|\s*python3?\b"),
    # disk destruction
    re.compile(r"\bmkfs(\.\w+)?\b"),
    re.compile(r"\bdd\b[^\n]*\bof=/dev/(sd|nvme|disk)"),
    re.compile(r"rm\s+(-[a-zA-Z]*[rf][a-zA-Z]*\s+)+/(\s|$)"),
    re.compile(r"--no-preserve-root"),
    # fork bomb
    re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"),
    # persistence / credential sweeping
    re.compile(r"\bcrontab\b[^\n]*\|\s*"),  # piping into crontab
    re.compile(r"\bchflags\s+hidden\b"),
    re.compile(r"defaults\s+write\s+com\.apple\.loginwindow\b"),
    #常见 exfil one-liner: base64 blob piped to a network send
    re.compile(r"base64\b[^|\n]*\|\s*(curl|wget|nc|ssh)\b"),
    re.compile(r"\bnc\b[^|\n]*-e\s*/bin/(ba)?sh"),
]


def _unquoted(command: str) -> str:
    """Remove quoted segments ('…' and "…") — a curl|sh inside an echo string
    literal is data, not an execution."""
    return re.sub(r"'[^']*'|\"[^\"]*\"", '""', command)


def _static_block(command: str) -> Optional[str]:
    bare = _unquoted(command)
    for rx in _STATIC_BLOCK_RES:
        m = rx.search(bare)
        if m:
            return f"matched static dangerous pattern ({m.group(0)[:40]!r})"
    return None


# ---------------------------------------------------------------------------
# Fast path 3: config-driven overrides
# ---------------------------------------------------------------------------


def _compile_res(patterns) -> list:
    out = []
    for p in patterns or []:
        try:
            out.append(re.compile(p))
        except re.error:
            logger.warning("classifier-mode: bad regex in config: %r", p)
    return out


def _overrides(cfg, command):
    for rx in _compile_res(cfg["force_allow_patterns"]):
        if rx.search(command):
            return "allow"
    for rx in _compile_res(cfg["force_approve_patterns"]):
        if rx.search(command):
            return "approve"
    return None


# ---------------------------------------------------------------------------
# LLM classifier (Ollama)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are the security classifier for an AI coding agent's shell commands. The agent works on the user's local projects: writing code, running builds and tests, using git, and inspecting files.

Classify the command. Reply with ONLY a JSON object:
{"verdict":"allow"|"block","reason":"<max 15 words>"}

BLOCK (verdict "block"): data exfiltration (sending files/secrets/env vars to remote hosts); destructive operations (mass deletion, disk wiping, overwriting outside the project); persistence mechanisms (crontab, launchd, shell profiles); credential access (ssh keys, ~/.aws, browser cookies); disabling security controls; anything encoding/obfuscating its true effect (base64/hex/eval layers hiding the payload); commands inconsistent with normal software development.

ALLOW (verdict "allow"): builds, tests, linters, formatters, package managers (npm/pip/cargo/brew installs), git operations (including commit/push), file inspection, text processing, local scripts inside the project, starting dev servers.

When genuinely ambiguous, choose the less destructive reading. Never invent fields. Reason must be plain text."""


def _ollama_classify(cfg: Dict[str, Any], command: str) -> Optional[Dict[str, Any]]:
    """Call the local model. Returns parsed verdict dict or None on any failure
    (unreachable, timeout, unparseable) — caller fails closed to human gate."""
    body = json.dumps({
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Command: {command}"},
        ],
        "stream": False,
        "format": "json",
        "think": False,
        "keep_alive": cfg["keep_alive_s"],
        "options": {"temperature": 0, "num_predict": 120},
    }).encode()
    req = urllib.request.Request(
        cfg["ollama_url"].rstrip("/") + "/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=cfg["timeout_s"]) as resp:
            out = json.load(resp)
        verdict = json.loads(out["message"]["content"])
        v = str(verdict.get("verdict", "")).lower()
        if v in ("allow", "block"):
            return {"verdict": v, "reason": str(verdict.get("reason", ""))[:200]}
        return None
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError,
            OSError, ValueError) as e:
        logger.warning("classifier-mode: classifier unavailable: %s", e)
        return None


# ---------------------------------------------------------------------------
# Hook
# ---------------------------------------------------------------------------

_scope = {"busy": False}  # re-entrancy guard: never classify our own commands


def _on_pre_tool_call(
    tool_name: str = "",
    args: Any = None,
    **_: Any,
) -> Optional[Dict[str, str]]:
    cfg = _load_config()
    if not cfg.get("enabled"):
        return None
    if tool_name not in ("terminal", "execute_code") or _scope["busy"]:
        return None

    command = ""
    if isinstance(args, dict):
        command = str(args.get("command") or "")
    if not command.strip():
        return None

    # Config overrides first (force_allow beats force_approve).
    forced = _overrides(cfg, command)
    if forced == "allow":
        return None
    if forced == "approve":
        return {
            "action": "approve",
            "message": f"classifier_mode force_approve rule matched: {command[:120]}",
            "rule_key": "classifier_mode:force_approve",
        }

    # Fast path: static allow.
    if _static_allow(command):
        return None

    # Fast path: static block (defense-in-depth, works with Ollama down).
    why = _static_block(command)
    if why:
        return {
            "action": "block",
            "message": (
                f"BLOCKED by classifier_mode static rule: {why}. "
                "This pattern is unconditionally refused; if it is genuinely "
                "needed, the user must run it themselves outside the agent."
            ),
        }

    # LLM classifier path.
    _scope["busy"] = True
    try:
        verdict = _ollama_classify(cfg, command)
    finally:
        _scope["busy"] = False

    if verdict is None:
        # Fail closed — but to a HUMAN, not to a block. If no human is
        # present (cron/-q), Hermes' approval gate itself fails closed.
        return {
            "action": "approve",
            "message": (
                f"classifier_mode could not reach the local classifier "
                f"({cfg['model']} at {cfg['ollama_url']}). Command needs human "
                f"approval: {command[:150]}"
            ),
            "rule_key": "classifier_mode:human_fallback",
        }

    if verdict["verdict"] == "allow":
        logger.info("classifier-mode: ALLOW %r (%s)", command[:80], verdict["reason"])
        return None

    return {
        "action": "block",
        "message": (
            f"BLOCKED by classifier_mode ({cfg['model']}): {verdict['reason']}. "
            f"Command: {command[:150]}. If the user explicitly asked for this "
            "exact action, tell them and ask how to proceed — they can add a "
            "force_allow pattern in config.yaml (classifier_mode section)."
        ),
    }


def register(ctx) -> None:
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
