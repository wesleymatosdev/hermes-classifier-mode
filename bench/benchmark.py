#!/usr/bin/env python3
"""Benchmark hermes-classifier-mode against real labeled + real observed data.

Data:
  - HF AmirAliGharesoufloo/CommandInjection: shell-shaped malicious payloads
  - HF jiacheng-ye/nl2bash: real-world benign bash
  - bench/hermes_real_commands.tsv: 6,355 ACTUAL commands Hermes executed for
    Wesley across telegram/cli/whatsapp/cron/subagent (ground truth: all were
    intended to run — so any "block" here is a disruption, though the user
    (Wesley) should eyeball the flagged list before trusting it as benign)

Usage: python3 bench/benchmark.py [--max-model N] [--sources telegram,cli]
"""
import argparse
import csv
import html
import json
import statistics
import sys
import time
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import importlib.util

_spec = importlib.util.spec_from_file_location("hcm", ROOT / "__init__.py")
cm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cm)

BENCH = Path(__file__).resolve().parent


def load_malicious():
    out, seen = [], set()
    with open(BENCH / "cmdinj.csv", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            s = html.unescape(urllib.parse.unquote(row["sentence"])).strip()
            if s and len(s) < 250 and any(c in s for c in ";|`$(<>&"):
                if s not in seen:
                    seen.add(s)
                    out.append(s)
    return out


def load_benign(limit):
    return [l for l in (BENCH / "benign.txt").read_text().splitlines() if l.strip()][:limit]


def load_hermes_real(sources):
    out = []
    for line in (BENCH / "hermes_real_commands.tsv").read_text().splitlines():
        if "\t" not in line:
            continue
        src, cmd = line.split("\t", 1)
        if not sources or src in sources:
            out.append((src, cmd.replace("⏎", "\n")))
    return out


def judge(command):
    t0 = time.perf_counter()
    out = cm._on_pre_tool_call(tool_name="terminal", args={"command": command})
    dt = time.perf_counter() - t0
    if out is None:
        return "allow", ("static-allow" if cm._static_allow(command) else "llm"), dt
    msg = out.get("message", "")
    if "static rule" in msg:
        return "block", "static-block", dt
    if out["action"] == "approve":
        return "approve", "llm-fallback", dt
    return "block", "llm", dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-model", type=int, default=10000)
    ap.add_argument("--sources", default="",
                    help="comma list of hermes sources to include (default: all)")
    args = ap.parse_args()
    sources = set(args.sources.split(",")) if args.sources else None

    mal = load_malicious()
    benign = load_benign(400)
    real = load_hermes_real(sources)
    print(f"dataset: {len(mal)} malicious | {len(benign)} nl2bash benign | "
          f"{len(real)} real hermes commands\n")

    results = []  # (class, label, verdict, layer, dt)
    n_model = 0
    for cmd in mal:
        v, layer, dt = judge(cmd)
        results.append(("hf-mal", "mal", v, layer, dt))
        if layer in ("llm", "llm-fallback"):
            n_model += 1
            if n_model >= args.max_model:
                break
    for cmd in benign:
        v, layer, dt = judge(cmd)
        results.append(("nl2bash", "ben", v, layer, dt))
    for src, cmd in real:
        v, layer, dt = judge(cmd)
        results.append((f"hermes:{src}", "ben(expected)", v, layer, dt))

    # ---- aggregate ----
    def summarize(rows, name):
        if not rows:
            return
        total = len(rows)
        allowed = sum(1 for r in rows if r[2] == "allow")
        blocked = sum(1 for r in rows if r[2] == "block")
        approved = sum(1 for r in rows if r[2] == "approve")
        static = sum(1 for r in rows if r[3].startswith("static"))
        lat = [r[4] for r in rows]
        model_lat = [r[4] for r in rows if r[3] in ("llm", "llm-fallback")]
        print(f"  {name:<22} n={total:<5} allow={allowed:<5} block={blocked:<5} "
              f"approve={approved:<4} static-fastpath={static/total:.0%} "
              f"median={statistics.median(lat):.3f}s"
              + (f" model-median={statistics.median(model_lat):.2f}s" if model_lat else ""))

    print("=" * 100)
    print("MALICIOUS (CommandInjection, should block):")
    summarize([r for r in results if r[1] == "mal"], "hf-malicious")
    print("\nBENIGN (nl2bash, should allow):")
    summarize([r for r in results if r[0] == "nl2bash"], "nl2bash")
    print("\nREAL HERMES HISTORY (should allow; blocks = friction Wesley would feel):")
    by_src = {}
    for r in results:
        if r[0].startswith("hermes:"):
            by_src.setdefault(r[0], []).append(r)
    for src in sorted(by_src):
        summarize(by_src[src], src.replace("hermes:", ""))

    # ---- false positives, eyeball list ----
    fps = [r for r in results if r[1].startswith("ben") and r[2] == "block"]
    print(f"\nTOTAL blocks on benign/real commands: {len(fps)}")
    print("top 25 (most disruptive first — real hermes commands):")
    real_fps = [r for r in fps if r[0].startswith("hermes:")]
    for r in real_fps[:25]:
        i = results.index(r)
        cmd = real[i - len(mal) - len(benign)][1]
        print(f"  [{r[0].replace('hermes:','')}] {cmd[:110]}")

    # save full log
    with open(BENCH / "results_full.json", "w") as f:
        json.dump([{"class": r[0], "label": r[1], "verdict": r[2], "layer": r[3],
                    "dt": round(r[4], 4)} for r in results], f, indent=1)
    print(f"\nfull results: bench/results_full.json")


if __name__ == "__main__":
    main()
