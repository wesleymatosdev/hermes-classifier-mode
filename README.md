# hermes-classifier-mode

Claude Code's [`--permission-mode auto`](https://code.claude.com/docs/en/permission-modes) for [Hermes Agent](https://github.com/NousResearch/hermes-agent): a **local LLM classifier reviews every shell command and `execute_code` script before it runs** — autonomy with a second model judging each action, instead of approval prompts (`default`) or skipping all checks (`--yolo`).

Hermes has no classifier-gated permission mode; this plugin adds one as a `pre_tool_call` hook. Runs entirely on-device via Ollama.

## How a verdict is reached

| Order | Layer | Latency | Behavior |
|---|---|---|---|
| 1 | `force_allow` / `force_approve` config regexes | ~0 | `force_allow` skips classification only for commands built entirely from the safe charset (letters, digits, spaces, `_ @ : . , / + -`); `force_approve` forces the human gate |
| 2 | Static read-only allow (`git status`, `ls`, `cat`, ...) | ~0 | proceed, no model call |
| 3 | Static catastrophic block (fork bombs, `curl\|sh`, disk wipes, base64-exfil one-liners) | ~0 | veto even if Ollama is down |
| 4 | Local LLM classifier | ~0.25s warm | `allow` → proceed, `block` → veto with reason **and a safer alternative** the agent can run instead (or `ask-user` when none exists) |
| 5 | Classifier unreachable / unparseable | — | **escalate to the human approval gate** (fail-closed: cron and `-q` runs deny, per Hermes' own gate semantics) |

Model vetoes include the reason, so the agent can self-correct or surface the decision to you.

Layer 4 verdicts are **deterministic**: each request pins a sampling seed derived from the command text (stable across restarts), and successful verdicts are cached per `(model, command)` for `verdict_cache_s` seconds (default 24h, `0` disables). The same command always gets the same verdict — temperature 0 alone does not stop a local model from inventing a different free-form reason on every call.

## Install

```bash
git clone https://github.com/wesleymatosdev/hermes-classifier-mode ~/.hermes/plugins/hermes-classifier-mode
# requires a local Ollama; recommended model (benchmark: 7/7, ~0.25s warm):
ollama pull qwen3:30b
hermes gateway restart   # or restart your CLI session
```

## Configuration

Settings in `config.yaml` (never `.env` — no secrets here):

```yaml
classifier_mode:
  enabled: true
  model: qwen3:30b              # any Ollama model; must support /api/chat
  ollama_url: http://localhost:11434
  timeout_s: 20
  keep_alive_s: 600             # keep the model warm between verdicts
  force_allow_patterns:         # regexes that skip classification only for
                                # commands built from the safe charset
                                # (letters/digits/spaces/_@:.,/+-);
                                # setting this key REPLACES the plugin's
                                # built-in defaults — copy them here if you
                                # want to keep them alongside your own
    - "^brew (install|upgrade) "
  force_approve_patterns:       # regexes that always need a human
    - "\\bgit push\\b.*\\b--force\\b"
    - "^docker (rmi|system prune)"
```

Before a `force_allow` regex is evaluated, the whole command must be built
from the safe charset: letters, digits, spaces, and the punctuation
`_ @ : . , / + -`. Any other byte — quotes, `$`, backticks, parentheses,
braces, brackets, operators, `#`, glob characters, tabs/newlines, non-ASCII —
keeps the command on the normal classifier/human-approval path. Shells keep
gaining execution forms spelled entirely from plain word bytes: three
adversarial reviews of a construct-by-construct lexer each found bypasses
(zsh's `=(cmd)` process substitution and `*(e:cmd:)` glob qualifier execute
arbitrary payloads with no operator byte; a live second line hides behind a
`#` comment break), so the fast path now simply refuses anything that isn't
plain words instead of enumerating syntax. Benign commands that quote or
expand anything pay classifier latency — that over-scrutiny is the
deliberate trade for closing the whole class at once.

## Benchmark (M4 Max 48GB, 2026-08-31)

qwen3:30b, `think: false`, JSON-constrained decoding, temp 0:

- 7/7 correct verdicts on destructive/exfil/benign suite
- ~0.25s per verdict warm, ~5s cold load (amortized by `keep_alive`)
- thinking mode enabled: seconds of latency and off-topic reasoning — **leave it off** (the plugin pins `think: false`)

## Threat model (honest version)

This is a **risk filter, not a security boundary** — same caveat Anthropic gives for auto mode. A local 30B classifier can be fooled by novel obfuscation, and it runs at the same privilege as everything else on your machine. Pair it with Hermes' container backends or `HERMES_WRITE_SAFE_ROOT` for defense-in-depth. What it reliably buys you: caught footguns (the `rm -rf` on the wrong path, the `curl | sh` from a sketchy README), injected-content actions that look abnormal next to your normal workflow, and a written reason for every veto.

## License

MIT
