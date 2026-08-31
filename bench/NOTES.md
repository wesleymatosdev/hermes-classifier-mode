# Classifier Fix + Benchmark — Step-by-Step Lab Notebook

Goal: build content-grade documentation of how we diagnose and fix the
classifier, with honest numbers, multiple models, and anonymized data.

## Ground rules for this work

- All real Hermes commands used in benchmarks are ANONYMIZED (paths,
  hostnames, usernames, tokens scrubbed) before commit. Raw tsv stays in /tmp.
- Every fix gets its own commit (STANDING RULE) so the diff trail doubles as
  the narrative for the write-up.
- Benchmark: same suite across all candidate models. No cherry-picking.

## Dataset (bench/)

| file | what | size |
|---|---|---|
| `cmdinj.csv` | HF `AmirAliGharesoufloo/CommandInjection` decoded, filtered to shell-shaped payloads | 381 unique |
| `benign.txt` | HF `jiacheng-ye/nl2bash` real-world bash | 399 |
| `hermes_real_commands.tsv` | REAL commands Hermes ran for Wesley (telegram/cli/whatsapp/cron/subagent), anonymized at commit | 6,355 |
| `hermes_real_commands_raw.tsv` | UNREDACTED original — /tmp only, NEVER committed | — |

## Baseline (qwen3:30b, v0.1 classifier) — MEASURED, from bench/results.txt

```
malicious (381):  recall 58.5%   ← misses injections
nl2bash (399):    FP rate 47.4%  ← half of benign bash gets blocked
real hermes cmds: ~2,000+ blocked = unusable friction
latency:          static 0.33ms | model median 0.53s | fastpath coverage 3%
```

**Diagnosis so far (before any fix):**
1. nl2bash contains sysadmin commands (`crontab`, `chmod ~`, `find ~ -exec`)
   that the system prompt's block rules hit — prompt is too aggressive.
2. CommandInjection fragments (`;id`, `|id`) are not complete commands; the
   classifier sees `;id` and has no idea what precedes it. Some "misses" are
   actually unanswerable inputs, not model failures. Fix = classifier prompt
   should treat bare fragments conservatively (block) OR benchmark should
   only include compound-attack samples — decide by looking at actual misses.
3. Real Hermes commands include heredocs, multi-line scripts, long pipes —
   nothing like the toy examples in the system prompt.

## Fix plan (each step = 1 commit + 1 benchmark run)

- [ ] Step 1: Anonymize the real-commands dataset (bench/make_anon.py), commit dataset
- [ ] Step 2: Rewrite system prompt: role = "coding-agent sidekick", calibrate
      against real distribution (heredocs, brew, docker, sqlite3, jq...),
      fragments=block rule, explicit allow-by-default for dev-tool verbs
- [ ] Step 3: Few-shot examples in prompt (10-12, drawn from real misses/FPs, anonymized)
- [ ] Step 4: Re-benchmark qwen3:30b → measure delta from v0.1 baseline
- [ ] Step 5: Same suite on Ornith-1.5-35B-A3B (Q5_K_M) + any other local candidates
- [ ] Step 6: Pick winner on (accuracy, FP-rate on REAL cmds, latency); update
      plugin default + README table
- [ ] Step 7: If Ornith wins on speed but loses on JSON discipline: add a
      fence-stripping pre-parse (it wrapped JSON in ```json fences in early tests)

## Log

### 2026-08-31 08:2x — baseline captured
Full 780-command run done earlier: numbers above. Misses + FPs saved in
`bench/results.txt` and `results_full.json`. Next: examine actual misses to
decide the fragment question with data, not vibes.

