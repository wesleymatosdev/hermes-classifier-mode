.PHONY: test test-live install check

test:
	python3 -m unittest discover -s tests -p 'test_plugin.py' -v

# Opt-in live suite (real Ollama): HERMES_CLASSIFIER_LIVE=1 gates it, plus a
# 3s /api/tags probe inside the module. Skips (never fails) when unavailable.
test-live:
	HERMES_CLASSIFIER_LIVE=1 python3 -m unittest discover -s tests -v

# The package-dir copy is what Hermes imports; it must stay byte-identical
# to the repo-root module (see .hermes.md).
check:
	diff __init__.py hermes_classifier_mode/__init__.py

install: check
	mkdir -p ~/.hermes/plugins/hermes-classifier-mode/hermes_classifier_mode
	cp __init__.py plugin.yaml README.md LICENSE ~/.hermes/plugins/hermes-classifier-mode/
	cp hermes_classifier_mode/__init__.py ~/.hermes/plugins/hermes-classifier-mode/hermes_classifier_mode/
	@echo "Installed. Enable with: hermes plugins enable hermes-classifier-mode"
