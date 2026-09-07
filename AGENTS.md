# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Add durable project-specific notes here as they are discovered through real work.

## Essentials

- Layout: the repo-root `__init__.py` is the shipped module; `hermes_classifier_mode/__init__.py` is the package-dir copy Hermes imports. Every code change is made once and copied to both — `make check` enforces byte-identity (see `.hermes.md` for the full gotchas).
- Test: `make test` (no Ollama needed); `tests/test_live.py` auto-skips when Ollama is down. Install: `make install`.
- Config gotcha: `classifier_mode.force_allow_patterns` set in the user's config.yaml REPLACES the plugin's default list (see README) — keep default entries when extending.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
