#!/usr/bin/env python3
"""Anonymize real Hermes commands for the committed benchmark dataset.

Scrubbed (case-insensitive, path- and value-preserving where safe):
  - usernames, mac device names (macbook-*, Wesley, wesleymatos*)
  - absolute home paths /Users/<name> -> ~/user
  - hostnames/domains -> host1.example, host2.example (consistent per original)
  - email addresses -> user1@example
  - Bearer tokens, API keys (sk-..., ghp_..., AKIA..., long hex/base64)
  - IPv4 addresses -> 10.0.0.N
  - URLs keep their STRUCTURE (scheme+path shape) but host is scrubbed

Not scrubbed: command structure, flags, package names, tool names — those
are the signal the classifier is benchmarked on.
"""
import re
import sys
from pathlib import Path

IN = Path("/tmp/hermes_real_commands.tsv")   # raw, /tmp only
OUT = Path(__file__).resolve().parent / "hermes_real_commands_anon.tsv"

# consistent pseudonyms
host_map: dict[str, str] = {}
email_map: dict[str, str] = {}
counters = {"host": 0, "email": 0, "ip": 0}

RE_USER_PATH = re.compile(r"/Users/[A-Za-z0-9_.-]+(?=/|$)")
# also catch short-username variants directly (defense in depth for the dataset)
RE_USER_NAME = re.compile(r"/Users/[a-z]{2,6}\b")
RE_DEVICE = re.compile(r"\b([A-Za-z0-9-]*macbook[A-Za-z0-9-]*|MacBook[- ]?(?:Pro|Air)?[A-Za-z0-9-]*)\b", re.I)
RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
RE_HOST = re.compile(r"\b(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,}\b")
RE_TOKEN = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{8,}|ghp_[A-Za-z0-9]{8,}|github_pat_[A-Za-z0-9_]{8,}|"
    r"AKIA[0-9A-Z]{8,}|xox[bap]-[A-Za-z0-9-]{8,}|Bearer\s+\S+|"
    r"[A-Fa-f0-9]{32,}|[A-Za-z0-9+/]{40,}={0,2})\b")
# long base64-ish runs: only when NOT path-shaped on either side (dots, slashes,
# dashes and word chars on either side make it a filename/flag, not a token)
RE_LONGRUN = re.compile(r"(?<![\w./@-])(?!.*\.)[A-Za-z0-9+/]{40,}={0,2}(?![\w./@-])")
RE_IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
RE_NAME = re.compile(r"\bwesley\b|\bwesleymatos\b|\bwesleymatosdev\b", re.I)


def scrub(cmd: str) -> str:
    cmd = RE_USER_PATH.sub("~/user", cmd)
    cmd = RE_USER_NAME.sub("/Users/the-user", cmd)
    cmd = RE_DEVICE.sub("macbook-device", cmd)
    cmd = RE_NAME.sub("the-user", cmd)
    # ordering matters: filenames like `madeline-redesign` were caught by the
    # long-run pass because they LOOK like base64. Only redact runs that are
    # not preceded/followed by path chars — and restore dot-suffixed names.
    cmd = RE_TOKEN.sub("REDACTED_TOKEN", cmd)
    cmd = RE_LONGRUN.sub("REDACTED_B64", cmd)

    def email_sub(m):
        e = m.group(0).lower()
        if e not in email_map:
            counters["email"] += 1
            email_map[e] = f"user{counters['email']}@example"
        return email_map[e]
    cmd = RE_EMAIL.sub(email_sub, cmd)

    def ip_sub(m):
        counters["ip"] += 1
        return f"10.0.0.{counters['ip'] % 250 + 1}"
    cmd = RE_IP.sub(ip_sub, cmd)

    def host_sub(m):
        h = m.group(0).lower()
        if h in ("localhost", "0.0.0.0", "example.com", "example.org"):
            return h  # structural, harmless
        if h not in host_map:
            counters["host"] += 1
            host_map[h] = f"host{counters['host']}.example"
        return host_map[h]
    # run host scrub last so redacted tokens/emails aren't mangled
    return RE_HOST.sub(host_sub, cmd)


def main():
    if not IN.exists():
        sys.exit("raw tsv missing at /tmp — re-extract from state.db first")
    n = 0
    with IN.open() as f, OUT.open("w") as out:
        for line in f:
            line = line.rstrip("\n")
            if "\t" not in line:
                continue
            src, cmd = line.split("\t", 1)
            out.write(f"{src}\t{scrub(cmd)}\n")
            n += 1
    print(f"anonymized {n} commands -> {OUT}")
    print(f"unique hosts pseudonymized: {counters['host']}, emails: {counters['email']}, ips: {counters['ip']}")
    # sanity: no residual real-looking long tokens
    residual = [l for l in OUT.read_text().splitlines() if RE_TOKEN.search(l)]
    print(f"residual token-pattern lines: {len(residual)}")


if __name__ == "__main__":
    main()
